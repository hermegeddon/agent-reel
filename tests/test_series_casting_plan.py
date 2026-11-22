"""Series casting planning, identity hygiene, participation gate, and isolation tests (reel-vln.4)."""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
from PIL import Image

from adapters.hermes import pipeline as pl
from adapters.hermes import reel_cli
from canon.series import SERIES_FILE, load as load_series
from canon.series_cast import CastConfigError

NOW = time.time()


def _png_bytes(color: tuple[int, int, int] = (40, 80, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _make_kit(
    assets_dir: Path,
    slug: str,
    name: str,
    *,
    role: str = "hero",
    voice: str = "",
    with_sheet: bool = True,
) -> Path:
    kit_dir = assets_dir / "kits" / slug
    kit_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "name": name,
        "role": role,
        "real_person": False,
        "description": f"{name} reference sheet.",
        "sheets": {"anime": "anime.png"} if with_sheet else {},
        "licence": "CC0 test fixture",
    }
    if voice:
        manifest["voice"] = voice
    (kit_dir / "kit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if with_sheet:
        (kit_dir / "anime.png").write_bytes(_png_bytes())
    return kit_dir


def _init_state_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "id TEXT PRIMARY KEY, source TEXT, title TEXT, profile_name TEXT, "
        "started_at REAL, ended_at REAL, end_reason TEXT, message_count INTEGER, "
        "tool_call_count INTEGER, input_tokens INTEGER, output_tokens INTEGER, "
        "model TEXT, git_branch TEXT, cwd TEXT, parent_session_id TEXT, "
        "last_activity_at REAL, archived INTEGER)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, "
        "content TEXT, tool_call_id TEXT, tool_calls TEXT, tool_name TEXT, "
        "timestamp REAL, token_count INTEGER, active INTEGER, compacted INTEGER)"
    )
    return conn


def _add_session(
    db_path: Path,
    *,
    session_id: str,
    profile_name: str,
    title: str,
    turns: list[tuple[str, str]],
    base_ts: float,
) -> None:
    conn = _init_state_db(db_path)
    end_ts = base_ts + len(turns) * 60.0
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            session_id,
            "cli",
            title,
            profile_name,
            base_ts,
            end_ts,
            "completed",
            len(turns),
            0,
            100,
            100,
            "test-model",
            "main",
            "/work/app",
            None,
            end_ts,
            0,
        ),
    )
    for idx, (role, content) in enumerate(turns):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, token_count, active, compacted) "
            "VALUES (?,?,?,?,?,?,?)",
            (session_id, role, content, base_ts + idx * 60.0 + 1.0, len(content) // 4, 1, 0),
        )
    conn.commit()
    conn.close()


def _init_kanban_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        "id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT, status TEXT, "
        "priority INTEGER, created_by TEXT, created_at REAL, started_at REAL, completed_at REAL, "
        "tenant TEXT, project_id TEXT, result TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, run_id INTEGER, "
        "kind TEXT, payload TEXT, created_at REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, profile TEXT, step_key TEXT, "
        "status TEXT, started_at REAL, ended_at REAL, outcome TEXT, summary TEXT, error TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS task_comments ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, author TEXT, body TEXT, created_at REAL)"
    )
    return conn


_SYNOPSIS_KEYS = frozenset(
    {"title", "logline", "want", "stakes", "obstacle", "turn", "ending", "meaning", "hook"}
)


def _short_grounded(text: str, max_words: int = 6) -> str:
    """Grounded snippet without template openings so ClaimGate and caption checks accept it."""
    import re

    lines = [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]
    for line in lines:
        cleaned = re.sub(
            r"^[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,2}\s+"
            r"(?:asked\s+for|reported|worked\s+on|wants|described)\s+",
            "",
            line,
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        words = (cleaned or line).split()
        if words:
            return " ".join(words[:max_words])
    return ""


def _default_story_turns(topic: str = "queue") -> list[tuple[str, str]]:
    return [
        ("user", f"We must decide how to fix the stalled {topic} lock right now."),
        ("assistant", f"Checked the stalled {topic} lock and spotted a missing guard."),
        ("user", f"Let us approve adding the safe timeout guard for the {topic} worker."),
        ("user", f"Had to laugh kidding about touching that broken {topic} worker again."),
        ("user", f"Ready to sign off on the clean {topic} limit for release."),
        ("assistant", f"Fixed the stalled {topic} worker cleanly with all tests green."),
    ]


def _setup_hermes_home(tmp_path: Path, *, default_turns: list[tuple[str, str]] | None = None) -> Path:
    home = tmp_path / "hermes_home"
    home.mkdir(parents=True, exist_ok=True)
    if default_turns is not None:
        _add_session(
            home / "state.db",
            session_id="sess-default-1",
            profile_name="default",
            title="Stabilize queue worker",
            turns=default_turns,
            base_ts=NOW - 7200,
        )
    return home


class _RecordingLLM:
    """Fake LLM that records every prompt/payload and returns valid schema JSON."""

    name = "fake-recording"

    def __init__(self, *, vocab_tags: dict[str, str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.vocab_tags = dict(vocab_tags or {})

    def complete_json(
        self,
        *,
        prompt: str,
        payload: Mapping[str, Any],
        schema_hint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "payload": json.loads(json.dumps(payload, default=str)),
                "schema_hint": dict(schema_hint or {}),
            }
        )
        hint = schema_hint or {}
        if "results" in hint:
            candidates = list(payload.get("candidates") or [])
            return {
                "results": [
                    {"name": cand, "category": self.vocab_tags.get(cand, "safe")}
                    for cand in candidates
                ]
            }
        if "spans" in hint:
            return {"spans": []}
        if "supported" in hint:
            return {"supported": True, "unbacked": [], "invented": []}
        if "titles" in hint:
            return {"titles": ["Queue Worker Fixed"]}
        if "score" in hint:
            return {"score": 5, "problems": []}
        if hint and set(hint) <= _SYNOPSIS_KEYS:
            src_lines = [
                ln.strip()
                for ln in str(payload.get("facts") or payload.get("brief") or "").splitlines()
                if ln.strip()
            ] or ["Checked the stalled queue lock."]
            out: dict[str, str] = {}
            for idx, field in enumerate(hint):
                if field == "title":
                    out[field] = "Queue Fixed"
                else:
                    out[field] = _short_grounded(src_lines[idx % len(src_lines)], 6)
            return out
        if "panels" in hint:
            panels_out = []
            for p in payload.get("panels") or []:
                src = str(p.get("evidence") or p.get("facts") or "")
                caption = _short_grounded(src, 6)
                speakers = list(p.get("speakers") or p.get("cast") or ())
                lines = []
                used_speakers: set[str] = set()
                for rl in p.get("real_lines") or ():
                    sp = str(rl.get("speaker") or "")
                    txt = _short_grounded(str(rl.get("text") or ""), 8)
                    if sp in speakers and txt and sp not in used_speakers:
                        lines.append({"speaker": sp, "text": txt})
                        used_speakers.add(sp)
                    if len(lines) >= 2:
                        break
                if not lines and speakers and src:
                    named = next((s for s in speakers if s.lower() in src.lower()), speakers[0])
                    snippet = _short_grounded(src, 6)
                    if snippet:
                        lines.append({"speaker": named, "text": snippet})
                panels_out.append({"key": p["key"], "caption": caption, "lines": lines})
            return {"panels": panels_out}
        if "caption" in hint and "lines" in hint:
            src = str(payload.get("evidence") or payload.get("facts") or "")
            return {"caption": _short_grounded(src, 6), "lines": []}
        return {}


def _all_text_under(directory: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for item in sorted(directory.rglob("*")):
        if item.is_file():
            try:
                files[str(item.relative_to(directory))] = item.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
    return files


def _write_cast_config(
    path: Path,
    *,
    series_id: str = "test-series",
    assets_dir: str = "",
    owner: dict[str, Any] | None = None,
    roles: dict[str, Any] | None = None,
) -> Path:
    data: dict[str, Any] = {
        "schema_version": 1,
        "series_id": series_id,
        "owner": owner or {"name": "Rowan", "kit": "placeholder-lead", "voice": "dry, exact"},
        "roles": (
            roles
            if roles is not None
            else {"hermes:default": {"name": "Kensei", "kit": "kensei"}}
        ),
    }
    if assets_dir:
        data["assets_dir"] = assets_dir
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _bind_hermes_env(monkeypatch: pytest.MonkeyPatch, home: Path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("REEL_ALLOW_CUSTOM_STATE", "1")


def _run_cli_plan(
    data_dir: Path,
    *,
    writer_llm: _RecordingLLM,
    cast_config: Path | None = None,
    assets_dir: Path | None = None,
    all_profiles: bool = False,
    harness: str = "hermes",
    harness_roots: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> int:
    class _StubCtx:
        llm = object()

    class _AdapterWrapper:
        def __init__(self, inner: _RecordingLLM) -> None:
            self._inner = inner

        def complete_json(self, **kwargs):
            return self._inner.complete_json(
                prompt=kwargs["prompt"],
                payload=kwargs["payload"],
                schema_hint=kwargs.get("schema_hint"),
            )

        def usage(self):
            return {"calls": len(self._inner.calls)}

    reel_cli.bind(_StubCtx())
    parser = argparse.ArgumentParser()
    reel_cli.register_cli(parser)
    argv = [
        "plan",
        "--dry-run",
        "--data-dir",
        str(data_dir),
        "--harness",
        harness,
    ]
    if all_profiles:
        argv.append("--all-profiles")
    if cast_config is not None:
        argv.extend(["--cast-config", str(cast_config)])
    if assets_dir is not None:
        argv.extend(["--assets-dir", str(assets_dir)])
    for hr in harness_roots or ():
        argv.extend(["--harness-root", hr])
    argv.extend(extra_args or ())
    args = parser.parse_args(argv)

    # Swap PluginLLMClient with a tiny wrapper routing straight to writer_llm (zero host dependency).
    import adapters.hermes.plugin_llm as pllm

    class _FakePluginClient:
        name = "fake-ctx"

        def __init__(self, _llm, max_calls: int = 150) -> None:
            self._w = _AdapterWrapper(writer_llm)
            self._max = max_calls

        def complete_json(self, **kwargs):
            return self._w.complete_json(**kwargs)

        def usage(self):
            return {
                "calls": len(writer_llm.calls),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "max_prompt_tokens_per_call": 0,
                "budget": self._max,
            }

    orig_cls = pllm.PluginLLMClient
    pllm.PluginLLMClient = _FakePluginClient  # type: ignore[misc]
    try:
        return reel_cli.reel_command(args)
    finally:
        pllm.PluginLLMClient = orig_cls  # type: ignore[misc]
        reel_cli.bind(None)


def test_malformed_cast_config_fails_before_llm_calls(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")

    bad_cases = [
        "{not json",
        json.dumps({"schema_version": 2, "series_id": "s", "owner": {"name": "Rowan", "kit": "placeholder-lead"}, "roles": {}}),
        json.dumps({"schema_version": 1, "series_id": "s", "extra": 1, "owner": {"name": "Rowan", "kit": "placeholder-lead"}, "roles": {}}),
        json.dumps({"schema_version": 1, "series_id": "s", "owner": {"name": "Rowan", "kit": "placeholder-lead"}, "roles": {"owner": {"name": "Alt", "kit": "kensei"}}}),
        json.dumps({"schema_version": 1, "series_id": "BAD ID!", "owner": {"name": "Rowan", "kit": "placeholder-lead"}, "roles": {}}),
        json.dumps({"schema_version": 1, "series_id": "s", "owner": {"name": "Rowan", "kit": "placeholder-lead"}, "roles": {"hermes:default": {"name": "  rowan ", "kit": "kensei"}}}),
        json.dumps({"schema_version": 1, "series_id": "s", "owner": {"name": "Assistant", "kit": "placeholder-lead"}, "roles": {}}),
        json.dumps({"schema_version": 1, "series_id": "s", "owner": {"name": "lead@example.com", "kit": "placeholder-lead"}, "roles": {}}),
    ]

    for idx, content in enumerate(bad_cases):
        cfg_path = tmp_path / f"bad-{idx}.json"
        cfg_path.write_text(content, encoding="utf-8")
        redactor_llm = _RecordingLLM()
        writer_llm = _RecordingLLM()
        with pytest.raises(CastConfigError):
            pl.run_pipeline(
                pl.PipelineConfig(
                    data_dir=tmp_path / f"out-{idx}",
                    state_dir=tmp_path / f"state-{idx}",
                    assets_dir=assets,
                    cast_config=cfg_path,
                    use_model=True,
                    llm=redactor_llm,
                    writer_llm=writer_llm,
                )
            )
        rc = _run_cli_plan(
            tmp_path / f"cli-out-{idx}",
            writer_llm=writer_llm,
            cast_config=cfg_path,
            assets_dir=assets,
        )
        assert rc == 2
        assert redactor_llm.calls == []
        assert writer_llm.calls == []


def test_missing_or_draft_kit_fails_preflight_zero_llm_calls(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)
    assets = tmp_path / "assets"
    _make_kit(assets, "draft-lead", "Rowan", with_sheet=False)

    missing_cfg = _write_cast_config(
        tmp_path / "missing-kit.json",
        owner={"name": "Rowan", "kit": "nonexistent-kit"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    draft_cfg = _write_cast_config(
        tmp_path / "draft-kit.json",
        owner={"name": "Rowan", "kit": "draft-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )

    for label, cfg_path in (("missing", missing_cfg), ("draft", draft_cfg)):
        redactor_llm = _RecordingLLM()
        writer_llm = _RecordingLLM()
        with pytest.raises(CastConfigError):
            pl.run_pipeline(
                pl.PipelineConfig(
                    data_dir=tmp_path / f"data-{label}",
                    state_dir=tmp_path / f"state-{label}",
                    assets_dir=assets,
                    cast_config=cfg_path,
                    use_model=True,
                    llm=redactor_llm,
                    writer_llm=writer_llm,
                )
            )
        rc = _run_cli_plan(
            tmp_path / f"cli-{label}",
            writer_llm=writer_llm,
            cast_config=cfg_path,
            assets_dir=assets,
        )
        assert rc == 2
        assert redactor_llm.calls == []
        assert writer_llm.calls == []


def test_owner_rename_replaces_sahil_and_respects_config_voice(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", voice="kit voice should lose to config voice")
    cfg_path = _write_cast_config(
        tmp_path / "cast.json",
        series_id="pilot-series",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead", "voice": "dry, practical watchmaker"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    data_dir = tmp_path / "data"
    writer_llm = _RecordingLLM()
    result = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=tmp_path / "state",
            cast_config=cfg_path,
            wording="meaning",
            min_strip_panels=4,
            use_model=True,
            writer_llm=writer_llm,
        )
    )
    assert result.problems == [], result.problems
    assert result.ir is not None
    assert len(result.episodes) >= 1

    # Config voice took precedence over kit voice in writer prompts.
    writer_prompts = json.dumps(writer_llm.calls)
    assert "dry, practical watchmaker" in writer_prompts
    assert "kit voice should lose to config voice" not in writer_prompts
    assert "Sahil" not in writer_prompts

    # No legacy "Sahil" or raw role keys anywhere in generated artifacts.
    files = _all_text_under(data_dir)
    for rel, text in files.items():
        assert "Sahil" not in text, f"Sahil leaked into {rel}"
        assert "hermes:default" not in text, f"raw role key leaked into {rel}"

    ir_data = json.loads((data_dir / "script-ir-1.json").read_text(encoding="utf-8"))
    binding = ir_data["cast_binding"]
    assert binding["series_id"] == "pilot-series"
    assert binding["characters"]["Rowan"] == {"kit": "placeholder-lead", "lead": True}
    assert binding["characters"]["Kensei"] == {"kit": "kensei", "lead": False}


def test_hermes_claude_profile_and_harness_claude_code_and_codex_stay_distinct(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path)
    _bind_hermes_env(monkeypatch, home)
    # Hermes profile named literally "claude" -> "Cinder"
    _add_session(
        home / "profiles" / "claude" / "state.db",
        session_id="sess-hermes-claude",
        profile_name="claude",
        title="Claude profile scheduler work",
        turns=_default_story_turns("scheduler"),
        base_ts=NOW - 10800,
    )
    # External Claude Code harness log -> "Tally"
    cc_root = tmp_path / "claude_code_logs" / "proj-a"
    cc_root.mkdir(parents=True, exist_ok=True)
    cc_rows = []
    for idx, (role, text) in enumerate(_default_story_turns("parser")):
        cc_rows.append(
            json.dumps(
                {
                    "type": role,
                    "timestamp": NOW - 7200 + idx * 60.0,
                    "sessionId": "cc-sess-1",
                    "message": {"role": role, "content": text},
                }
            )
        )
    (cc_root / "session.jsonl").write_text("\n".join(cc_rows) + "\n", encoding="utf-8")

    # External Codex harness log -> "Rook"
    codex_root = tmp_path / "codex_logs" / "2026"
    codex_root.mkdir(parents=True, exist_ok=True)
    codex_rows = []
    for idx, (role, text) in enumerate(_default_story_turns("router")):
        codex_rows.append(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": NOW - 3600 + idx * 60.0,
                    "payload": {
                        "type": "message",
                        "role": role,
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
    (codex_root / "rollout.jsonl").write_text("\n".join(codex_rows) + "\n", encoding="utf-8")

    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    _make_kit(assets, "placeholder-claude", "Cinder")
    _make_kit(assets, "placeholder-cc", "Tally")
    _make_kit(assets, "placeholder-codex", "Rook")
    cfg_path = _write_cast_config(
        tmp_path / "cast.json",
        series_id="multi-harness",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={
            "hermes:claude": {"name": "Cinder", "kit": "placeholder-claude"},
            "harness:claude-code": {"name": "Tally", "kit": "placeholder-cc"},
            "harness:codex": {"name": "Rook", "kit": "placeholder-codex"},
        },
    )

    data_dir = tmp_path / "data"
    writer_llm = _RecordingLLM()
    result = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=tmp_path / "state",
            cast_config=cfg_path,
            harnesses=("hermes", "claude-code", "codex"),
            harness_options={
                "claude-code": {"root": str(tmp_path / "claude_code_logs")},
                "codex": {"root": str(tmp_path / "codex_logs")},
            },
            all_profiles=True,
            episodes=3,
            min_strip_panels=4,
            use_model=True,
            writer_llm=writer_llm,
        )
    )
    assert result.problems == [], result.problems
    assert len(result.episodes) == 3

    episode_agents: set[str] = set()
    for num in (1, 2, 3):
        ir_data = json.loads((data_dir / f"script-ir-{num}.json").read_text(encoding="utf-8"))
        chars = set(ir_data["cast_binding"]["characters"].keys())
        episode_agents |= (chars - {"Rowan"})
    assert episode_agents == {"Cinder", "Tally", "Rook"}

    # No raw role keys anywhere on disk.
    for rel, text in _all_text_under(data_dir).items():
        for raw_key in ("hermes:claude", "harness:claude-code", "harness:codex"):
            assert raw_key not in text, f"{raw_key} leaked into {rel}"


def test_unmapped_bystander_in_all_profiles_does_not_abort_and_never_leaks(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns("pipeline"))
    _bind_hermes_env(monkeypatch, home)

    secret_profile = "acmecorp-secret-prod"
    # Bystander profile has its own session, its own kanban task/comment, and MEMORY.md.
    _add_session(
        home / "profiles" / secret_profile / "state.db",
        session_id="sess-secret-1",
        profile_name=secret_profile,
        title="Secret internal project",
        turns=_default_story_turns("secret"),
        base_ts=NOW - 10800,
    )
    mem_dir = home / "profiles" / secret_profile / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text("Bystander memory note about acmecorp.\n", encoding="utf-8")

    kconn = _init_kanban_db(home / "kanban.db")
    kconn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t-sec", "Internal bystander task", "", secret_profile, "completed", 1, secret_profile, NOW - 5000, NOW - 4900, NOW - 4800),
    )
    kconn.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, started_at, ended_at, outcome, summary, error) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("t-sec", secret_profile, "run", "completed", NOW - 4900, NOW - 4800, "ok", "done", ""),
    )
    kconn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        ("t-sec", secret_profile, "Checked bystander step.", NOW - 4850),
    )
    kconn.commit()
    kconn.close()

    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    cfg_path = _write_cast_config(
        tmp_path / "cast.json",
        series_id="bystander-test",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )

    data_dir = tmp_path / "data"
    redactor_llm = _RecordingLLM()
    writer_llm = _RecordingLLM()
    result = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=tmp_path / "state",
            cast_config=cfg_path,
            all_profiles=True,
            episodes=1,
            min_strip_panels=4,
            use_model=True,
            llm=redactor_llm,
            writer_llm=writer_llm,
        )
    )
    # Run still succeeds for the mapped default profile episode!
    assert result.ir is not None
    assert len(result.episodes) == 1
    assert not result.problems

    # Verify zero bystander leakage across ALL LLM payloads (redactor + writer) and ALL files on disk.
    for provider_name, calls in (("redactor", redactor_llm.calls), ("writer", writer_llm.calls)):
        blob = json.dumps(calls)
        assert secret_profile not in blob, f"{secret_profile} leaked to {provider_name} LLM"
        assert f"hermes:{secret_profile}" not in blob
        assert "hermes:default" not in blob

    for rel, text in _all_text_under(data_dir).items():
        assert secret_profile not in text, f"{secret_profile} leaked into {rel}"
        assert f"hermes:{secret_profile}" not in text
        assert "hermes:default" not in text

    # Canon must NOT synthesize an agent entity for the unmapped bystander.
    canon_data = json.loads((data_dir / "canon.json").read_text(encoding="utf-8"))
    entity_names = {e["name"] for e in canon_data.get("entities", [])}
    assert all(not name.startswith("unmapped-role-") for name in entity_names)
    assert secret_profile not in entity_names


def test_mapped_sensitive_profile_writes_episode_with_zero_leaks(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path)
    _bind_hermes_env(monkeypatch, home)
    secret_profile = "acmecorp-secret-prod"
    _add_session(
        home / "profiles" / secret_profile / "state.db",
        session_id="sess-sec-mapped",
        profile_name=secret_profile,
        title="Cache repair session",
        turns=_default_story_turns("cache"),
        base_ts=NOW - 5000,
    )

    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    _make_kit(assets, "placeholder-vex", "Vex")
    cfg_path = _write_cast_config(
        tmp_path / "cast.json",
        series_id="mapped-sensitive",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={f"hermes:{secret_profile}": {"name": "Vex", "kit": "placeholder-vex"}},
    )

    data_dir = tmp_path / "data"
    redactor_llm = _RecordingLLM()
    writer_llm = _RecordingLLM()
    result = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=tmp_path / "state",
            cast_config=cfg_path,
            all_profiles=True,
            episodes=1,
            min_strip_panels=4,
            use_model=True,
            llm=redactor_llm,
            writer_llm=writer_llm,
        )
    )
    assert result.problems == [], result.problems
    assert result.ir is not None
    ir_data = json.loads((data_dir / "script-ir-1.json").read_text(encoding="utf-8"))
    assert "Vex" in ir_data["cast_binding"]["characters"]
    assert ir_data["cast_binding"]["characters"]["Vex"]["kit"] == "placeholder-vex"

    for blob in (json.dumps(redactor_llm.calls), json.dumps(writer_llm.calls)):
        assert secret_profile not in blob
        assert f"hermes:{secret_profile}" not in blob
    for rel, text in _all_text_under(data_dir).items():
        assert secret_profile not in text, f"{secret_profile} leaked into {rel}"


def test_unmapped_participating_role_stops_episode_before_story_writing(tmp_path, monkeypatch, capsys):
    # Case A: Candidate session comes solely from an unmapped profile -> 0 writer calls.
    home_a = _setup_hermes_home(tmp_path / "case_a")
    _bind_hermes_env(monkeypatch, home_a)
    _add_session(
        home_a / "profiles" / "secret-guest" / "state.db",
        session_id="sess-guest",
        profile_name="secret-guest",
        title="Guest fix",
        turns=_default_story_turns("guest"),
        base_ts=NOW - 4000,
    )
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    cfg_a = _write_cast_config(
        tmp_path / "cast_a.json",
        series_id="stop-unmapped-a",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    writer_a = _RecordingLLM()
    data_a = tmp_path / "data_a"
    rc = _run_cli_plan(
        data_a,
        writer_llm=writer_a,
        cast_config=cfg_a,
        all_profiles=True,
    )
    out_a = capsys.readouterr().out
    assert rc == 1
    assert writer_a.calls == [], "writer model must not be called when participating role is unmapped"
    assert "hermes:secret-guest" in out_a
    metrics_a = json.loads((data_a / "metrics.json").read_text(encoding="utf-8"))
    assert metrics_a["episodes"] == []
    assert any("unmapped-role-" in row["reason"] for row in metrics_a["skipped_stories"])
    for rel, text in _all_text_under(data_a).items():
        assert "secret-guest" not in text, f"raw profile leaked into {rel}"

    # Case B: Mapped task (`hermes:default`) has a relied-on unmapped kanban comment (`outcome`) -> stopped before writing.
    home_b = _setup_hermes_home(tmp_path / "case_b")
    _bind_hermes_env(monkeypatch, home_b)
    kconn = _init_kanban_db(home_b / "kanban.db")
    kconn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("t-collab", "Fix build script cleanly", "Fix build script cleanly.", "default", "completed", 1, "owner", NOW - 5000, NOW - 4800, NOW - 4000),
    )
    for kind_val, ts_val, note in (
        ("blocked", NOW - 4700, "step 0"),
        ("unblocked", NOW - 4640, "step 1"),
        ("blocked", NOW - 4200, "step 2"),
        ("completed", NOW - 4100, "step 3"),
    ):
        kconn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,?)",
            ("t-collab", kind_val, json.dumps({"note": note}), ts_val),
        )
    kconn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        ("t-collab", "default", "Checked build lock timeout guard.", NOW - 4550),
    )
    kconn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        ("t-collab", "default", "Patched build retry threshold limit.", NOW - 4450),
    )
    kconn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        ("t-collab", "secret-reviewer", "Hold release until retry check finishes.", NOW - 3900),
    )
    kconn.commit()
    kconn.close()

    writer_b = _RecordingLLM()
    data_b = tmp_path / "data_b"
    res_b = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_b,
            state_dir=tmp_path / "state_b",
            assets_dir=assets,
            cast_config=cfg_a,
            min_strip_panels=1,
            use_model=True,
            writer_llm=writer_b,
        )
    )
    assert res_b.ir is None
    assert writer_b.calls == []
    assert any("hermes:secret-reviewer" in diag for diag in res_b.cast_diagnostics), res_b.skipped_stories
    assert any("unmapped participating role" in row["reason"] for row in res_b.skipped_stories)
    for rel, text in _all_text_under(data_b).items():
        assert "secret-reviewer" not in text, f"raw commenter leaked into {rel}"

    # Case C: Same-thread unmapped event whose contribution is NOT used in the story does NOT block.
    home_c = _setup_hermes_home(tmp_path / "case_c")
    _bind_hermes_env(monkeypatch, home_c)
    kconn_c = _init_kanban_db(home_c / "kanban.db")
    kconn_c.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at, result) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "t-clean",
            "Stabilize queue lock guard",
            "Stabilize queue lock guard.",
            "default",
            "completed",
            1,
            "owner",
            NOW - 5000,
            NOW - 4800,
            NOW - 3800,
            "Fixed the queue lock cleanly.",
        ),
    )
    for kind_val, ts_val, note in (
        ("blocked", NOW - 4700, "worker step 0"),
        ("unblocked", NOW - 4640, "worker step 1"),
        ("blocked", NOW - 4200, "worker step 2"),
        ("completed", NOW - 4100, "worker step 3"),
    ):
        kconn_c.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?,?,?,?)",
            ("t-clean", kind_val, json.dumps({"note": note}), ts_val),
        )
    for msg, ts_val in (
        ("Checked the queue lock guard and found no timeout.", NOW - 4550),
        ("Added clean retry guard across the queue worker.", NOW - 4450),
        ("Verified queue tests stay green after restart.", NOW - 4350),
    ):
        kconn_c.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
            ("t-clean", "default", msg, ts_val),
        )
    # Same-thread unmapped background run and short ack comment (unused by brief/arc/facts).
    kconn_c.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
        ("t-clean", "secret-unused", "ack", NOW - 4310),
    )
    kconn_c.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, started_at, ended_at, outcome, summary, error) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        ("t-clean", "secret-unused", "bg-probe", "ok", NOW - 4300, NOW - 4290, "ok", "", ""),
    )
    kconn_c.commit()
    kconn_c.close()

    writer_c = _RecordingLLM()
    data_c = tmp_path / "data_c"
    res_c = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_c,
            state_dir=tmp_path / "state_c",
            assets_dir=assets,
            cast_config=cfg_a,
            all_profiles=True,
            wording="meaning",
            min_strip_panels=1,
            use_model=True,
            writer_llm=writer_c,
        )
    )
    assert res_c.events_ingested >= 12
    assert res_c.problems == [], res_c.problems
    assert len(res_c.episodes) == 1
    assert res_c.cast_diagnostics == []
    assert "secret-unused" not in json.dumps(writer_c.calls)
    assert "unmapped-role-" not in json.dumps(writer_c.calls)
    assert "unmapped-role-" not in (data_c / "script-ir-1.json").read_text(encoding="utf-8")
    assert "unmapped-role-" not in (data_c / "episode-1-script.md").read_text(encoding="utf-8")
    for rel, text in _all_text_under(data_c).items():
        assert "secret-unused" not in text, f"unused same-thread profile leaked into {rel}"

    # Case D: Relied-on unmapped episode skips ONLY that episode before writing and continues with eligible ones.
    home_d = _setup_hermes_home(tmp_path / "case_d", default_turns=_default_story_turns("worker"))
    _bind_hermes_env(monkeypatch, home_d)
    _add_session(
        home_d / "profiles" / "secret-guest" / "state.db",
        session_id="sess-guest-newest",
        profile_name="secret-guest",
        title="Guest urgent incident",
        turns=_default_story_turns("zebra-incident"),
        base_ts=NOW - 1800,
    )
    writer_d = _RecordingLLM()
    data_d = tmp_path / "data_d"
    rc_d = _run_cli_plan(
        data_d,
        writer_llm=writer_d,
        cast_config=cfg_a,
        all_profiles=True,
        extra_args=["--episodes", "2"],
    )
    out_d = capsys.readouterr().out
    assert rc_d == 0
    assert "hermes:secret-guest" in out_d
    assert "zebra-incident" not in json.dumps(writer_d.calls)
    assert "secret-guest" not in json.dumps(writer_d.calls)
    metrics_d = json.loads((data_d / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics_d["episodes"]) == 1
    assert any("unmapped-role-" in row["reason"] for row in metrics_d["skipped_stories"])
    for rel, text in _all_text_under(data_d).items():
        assert "secret-guest" not in text, f"raw profile leaked into {rel}"


def test_series_and_cache_isolation_across_cast_revisions(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    _make_kit(assets, "placeholder-alt", "Maren")
    _make_kit(assets, "placeholder-claude", "Cinder")

    cfg_v1 = _write_cast_config(
        tmp_path / "cast-v1.json",
        series_id="continuity",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    writer_v1 = _RecordingLLM()
    res1 = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=state_dir,
            cast_config=cfg_v1,
            min_strip_panels=4,
            use_model=True,
            writer_llm=writer_v1,
        )
    )
    assert res1.problems == []
    series_path = data_dir / "series-continuity.json"
    assert series_path.is_file()
    assert not (data_dir / SERIES_FILE).exists()
    assert "hermes:default" not in series_path.read_text(encoding="utf-8")
    loaded = load_series(series_path)
    assert loaded.series_id == "continuity"
    assert loaded.cast_digest
    assert loaded.cast_roles
    assert len(loaded.chapters) == 1
    assert (data_dir / "cache" / f"cast-continuity-{loaded.cast_digest}").is_dir()

    # Adding a co-star later in the same series is allowed cleanly (new cache dir, preserved series).
    _add_session(
        home / "profiles" / "claude" / "state.db",
        session_id="sess-claude-later",
        profile_name="claude",
        title="Second episode with Cinder",
        turns=_default_story_turns("router"),
        base_ts=NOW - 3600,
    )
    cfg_added_costar = _write_cast_config(
        tmp_path / "cast-v1b.json",
        series_id="continuity",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={
            "hermes:default": {"name": "Kensei", "kit": "kensei"},
            "hermes:claude": {"name": "Cinder", "kit": "placeholder-claude"},
        },
    )
    res1b = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=state_dir,
            cast_config=cfg_added_costar,
            all_profiles=True,
            min_strip_panels=4,
            use_model=True,
            writer_llm=_RecordingLLM(),
        )
    )
    assert res1b.problems == []
    loaded_b = load_series(series_path)
    assert loaded_b.cast_digest != loaded.cast_digest
    assert len(loaded_b.chapters) == 2
    assert (data_dir / "cache" / f"cast-continuity-{loaded_b.cast_digest}").is_dir()
    assert "hermes:claude" not in series_path.read_text(encoding="utf-8")

    # Removing an existing role under the same series_id must fail preflight with 0 LLM calls.
    cfg_removed = _write_cast_config(
        tmp_path / "cast-removed.json",
        series_id="continuity",
        assets_dir=str(assets),
        owner={"name": "Rowan", "kit": "placeholder-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    writer_removed = _RecordingLLM()
    with pytest.raises(CastConfigError, match="removed"):
        pl.run_pipeline(
            pl.PipelineConfig(
                data_dir=data_dir,
                state_dir=state_dir,
                cast_config=cfg_removed,
                min_strip_panels=4,
                use_model=True,
                writer_llm=writer_removed,
            )
        )
    assert writer_removed.calls == []

    # Changing owner/existing role under the same series_id + data_dir must fail preflight with 0 LLM calls.
    cfg_v2_same_id = _write_cast_config(
        tmp_path / "cast-v2.json",
        series_id="continuity",
        assets_dir=str(assets),
        owner={"name": "Maren", "kit": "placeholder-alt"},
        roles={
            "hermes:default": {"name": "Kensei", "kit": "kensei"},
            "hermes:claude": {"name": "Cinder", "kit": "placeholder-claude"},
        },
    )
    writer_blocked = _RecordingLLM()
    with pytest.raises(CastConfigError, match="cast binding changed"):
        pl.run_pipeline(
            pl.PipelineConfig(
                data_dir=data_dir,
                state_dir=state_dir,
                cast_config=cfg_v2_same_id,
                min_strip_panels=4,
                use_model=True,
                writer_llm=writer_blocked,
            )
        )
    assert writer_blocked.calls == []

    # Bumping series_id creates an isolated series ledger and cache directory cleanly.
    cfg_v2_new_id = _write_cast_config(
        tmp_path / "cast-s2.json",
        series_id="continuity-s2",
        assets_dir=str(assets),
        owner={"name": "Maren", "kit": "placeholder-alt"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    res2 = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=state_dir,
            cast_config=cfg_v2_new_id,
            min_strip_panels=4,
            use_model=True,
            writer_llm=_RecordingLLM(),
        )
    )
    assert res2.problems == []
    s2 = load_series(data_dir / "series-continuity-s2.json")
    assert s2.series_id == "continuity-s2"
    assert (data_dir / "cache" / f"cast-continuity-s2-{s2.cast_digest}").is_dir()


def test_assets_dir_precedence_cli_over_config_over_data_dir(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)

    data_dir = tmp_path / "data"
    cfg_assets = tmp_path / "cfg_assets"
    cli_assets = tmp_path / "cli_assets"
    _make_kit(data_dir, "data-only-lead", "Rowan")
    _make_kit(cfg_assets, "cfg-only-lead", "Rowan")
    _make_kit(cli_assets, "cli-only-lead", "Rowan", voice="cli precedence voice")

    # Config assets_dir overrides data_dir (data-only-lead would fail; cfg-only-lead succeeds).
    cfg_needs_cfg = _write_cast_config(
        tmp_path / "cast_cfg.json",
        series_id="prec-1",
        assets_dir=str(cfg_assets),
        owner={"name": "Rowan", "kit": "cfg-only-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    rc1 = _run_cli_plan(
        tmp_path / "out_prec1",
        writer_llm=_RecordingLLM(),
        cast_config=cfg_needs_cfg,
    )
    assert rc1 == 0

    # CLI --assets-dir overrides cast.json assets_dir.
    cfg_overridden = _write_cast_config(
        tmp_path / "cast_cli.json",
        series_id="prec-2",
        assets_dir=str(cfg_assets),
        owner={"name": "Rowan", "kit": "cli-only-lead"},
        roles={"hermes:default": {"name": "Kensei", "kit": "kensei"}},
    )
    rc_without_cli = _run_cli_plan(
        tmp_path / "out_prec2_fail",
        writer_llm=_RecordingLLM(),
        cast_config=cfg_overridden,
    )
    assert rc_without_cli == 2
    rc_with_cli = _run_cli_plan(
        tmp_path / "out_prec2_ok",
        writer_llm=_RecordingLLM(),
        cast_config=cfg_overridden,
        assets_dir=cli_assets,
    )
    assert rc_with_cli == 0


def test_legacy_run_without_cast_config_unchanged(tmp_path, monkeypatch):
    home = _setup_hermes_home(tmp_path, default_turns=_default_story_turns())
    _bind_hermes_env(monkeypatch, home)
    data_dir = tmp_path / "legacy_data"
    result = pl.run_pipeline(
        pl.PipelineConfig(
            data_dir=data_dir,
            state_dir=tmp_path / "legacy_state",
            min_strip_panels=4,
            use_model=True,
            writer_llm=_RecordingLLM(),
        )
    )
    assert result.problems == [], result.problems
    assert result.ir is not None
    assert (data_dir / SERIES_FILE).is_file()
    ir_data = json.loads((data_dir / "script-ir-1.json").read_text(encoding="utf-8"))
    assert "cast_binding" not in ir_data
    script_md = (data_dir / "episode-1-script.md").read_text(encoding="utf-8")
    assert "Sahil" in script_md
