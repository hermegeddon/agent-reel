"""End-to-end CLI qualification tests for opt-in series casting across plan and render (reel-vln.6)."""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Mapping

import pytest
from PIL import Image

from adapters.hermes import reel_cli, render_cli
from canon.assets import Approvals, load_kits
from canon.series import SERIES_FILE

REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = time.time()

_SYNOPSIS_KEYS = frozenset(
    {"title", "logline", "want", "stakes", "obstacle", "turn", "ending", "meaning", "hook"}
)


def _png_bytes(size: tuple[int, int] = (320, 220), color: tuple[int, int, int] = (50, 95, 150)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _short_grounded(text: str, max_words: int = 6) -> str:
    for line in str(text or "").splitlines():
        cleaned = re.sub(
            r"^[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,2}\s+"
            r"(?:asked\s+for|reported|worked\s+on|wants|described)\s+",
            "",
            line.strip(),
            count=1,
            flags=re.IGNORECASE,
        ).strip()
        words = (cleaned or line).split()
        if words:
            return " ".join(words[:max_words])
    return ""


def _story_turns(topic: str) -> list[tuple[str, str]]:
    return [
        ("user", f"We must decide how to fix the stalled {topic} lock right now."),
        ("assistant", f"Checked the stalled {topic} lock and spotted a missing guard."),
        ("user", f"Let us approve adding the safe timeout guard for the {topic} worker."),
        ("user", f"Had to laugh kidding about touching that broken {topic} worker again."),
        ("user", f"Ready to sign off on the clean {topic} limit for release."),
        ("assistant", f"Fixed the stalled {topic} worker cleanly with all tests green."),
    ]


def _add_hermes_session(db_path: Path, *, session_id: str, profile: str, topic: str, base_ts: float) -> None:
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
    turns = _story_turns(topic)
    end_ts = base_ts + len(turns) * 60.0
    conn.execute(
        "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            session_id,
            "cli",
            f"Stabilize {topic} worker",
            profile,
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


def _make_kit(assets_dir: Path, slug: str, name: str, *, role: str = "agent") -> None:
    kit_dir = assets_dir / "kits" / slug
    kit_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "role": role,
        "real_person": False,
        "owner": role == "hero",
        "public_consent": True,
        "description": f"{name} ({slug}) reference sheet.",
        "sheets": {"anime": "anime.png"},
        "licence": "CC0 E2E test fixture",
    }
    (kit_dir / "kit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (kit_dir / "anime.png").write_bytes(_png_bytes((64, 64), color=(30, 100, 170)))


class _FakeE2EHarness:
    """Self-contained fake LLM + vision + ComfyUI/OpenAI HTTP transports (zero network calls)."""

    def __init__(self) -> None:
        self.llm_calls: list[dict[str, Any]] = []
        self.image_calls: list[dict[str, Any]] = []
        self.vision_calls: list[dict[str, Any]] = []
        self._png = _png_bytes((320, 220), color=(70, 120, 170))
        self._comfy_counter = 0

    def llm_complete_json(
        self,
        *,
        prompt: str,
        payload: Mapping[str, Any],
        schema_hint: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.llm_calls.append({"prompt": prompt, "payload": json.loads(json.dumps(payload, default=str))})
        hint = schema_hint or {}
        if "results" in hint:
            return {"results": [{"name": c, "category": "safe"} for c in payload.get("candidates") or ()]}
        if "spans" in hint:
            return {"spans": []}
        if "supported" in hint:
            return {"supported": True, "unbacked": [], "invented": []}
        if "titles" in hint:
            return {"titles": ["Worker Queue Repaired"]}
        if "score" in hint:
            return {"score": 5, "problems": []}
        if hint and set(hint) <= _SYNOPSIS_KEYS:
            src_lines = [
                ln.strip()
                for ln in str(payload.get("facts") or payload.get("brief") or "").splitlines()
                if ln.strip()
            ] or ["Checked the stalled queue lock."]
            return {
                field: ("Worker Fixed" if field == "title" else _short_grounded(src_lines[i % len(src_lines)], 6))
                for i, field in enumerate(hint)
            }
        if "panels" in hint:
            panels_out = []
            for p in payload.get("panels") or []:
                src = str(p.get("evidence") or p.get("facts") or "")
                caption = _short_grounded(src, 6)
                speakers = list(p.get("speakers") or p.get("cast") or ())
                lines = []
                used: set[str] = set()
                for rl in p.get("real_lines") or ():
                    sp = str(rl.get("speaker") or "")
                    txt = _short_grounded(str(rl.get("text") or ""), 8)
                    if sp in speakers and txt and sp not in used:
                        lines.append({"speaker": sp, "text": txt})
                        used.add(sp)
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

    def transport(self, method: str, url: str, headers: dict[str, str], body: bytes | None):
        if "api.openai.com" in url:
            self.image_calls.append({"source": "openai", "url": url})
            return 200, json.dumps({"data": [{"b64_json": base64.b64encode(self._png).decode()}]}).encode()
        if "/upload/image" in url:
            return 200, json.dumps({"name": "ref.png"}).encode()
        if "/prompt" in url:
            self._comfy_counter += 1
            job_id = f"comfy-{self._comfy_counter}"
            self.image_calls.append({"source": "comfyui", "url": url, "job_id": job_id})
            return 200, json.dumps({"prompt_id": job_id}).encode()
        if "/history/" in url:
            job_id = url.rsplit("/", 1)[-1]
            return 200, json.dumps(
                {
                    job_id: {
                        "status": {"completed": True},
                        "outputs": {"9": {"images": [{"filename": "panel.png", "subfolder": "", "type": "output"}]}},
                    }
                }
            ).encode()
        if "/view?" in url:
            return 200, self._png
        raise AssertionError(f"unexpected HTTP call to {url}")

    def dispatch_tool(self, tool_name: str, payload: dict[str, Any]) -> str:
        if tool_name != "vision_analyze":
            raise AssertionError(f"unexpected dispatch tool: {tool_name}")
        question = str(payload.get("question", ""))
        self.vision_calls.append(dict(payload))
        if "tail points at the face" in question:
            return json.dumps({"success": True, "analysis": json.dumps({"misses": []})})
        names: list[str] = []
        for line in question.splitlines():
            if line.startswith("Reference descriptions: "):
                raw_desc = line[len("Reference descriptions: "):].strip().rstrip(".")
                for entry in raw_desc.split(";"):
                    part = entry.split(":", 1)[0].strip()
                    if part and part != "none":
                        names.append(part)
                break
        heads = {n: [round(0.10 + (i % 3) * 0.28, 2), 0.25, round(0.22 + (i % 3) * 0.28, 2), 0.40] for i, n in enumerate(names)}
        on_model = {n: True for n in names}
        return json.dumps(
            {
                "success": True,
                "analysis": json.dumps(
                    {
                        "on_model": on_model,
                        "extra_people": 0,
                        "text_in_image": False,
                        "heads": heads,
                        "free": [[0.04, 0.02, 0.96, 0.22]],
                        "keep_out": [],
                    }
                ),
            }
        )


def _run_plan_and_render_with_harness(
    monkeypatch: pytest.MonkeyPatch,
    harness: _FakeE2EHarness,
    *,
    argv: list[str],
    default_dir: Path,
) -> int:
    import adapters.hermes.plugin_llm as pllm
    from canon import providers

    orig_plugin_cls = pllm.PluginLLMClient
    orig_make_gen = providers.make_generator

    class _FakePluginClient:
        name = "fake-e2e-llm"

        def __init__(self, _llm, max_calls: int = 150) -> None:
            self._max = max_calls
            self._start_calls = len(harness.llm_calls)

        def complete_json(self, **kwargs):
            return harness.llm_complete_json(
                prompt=kwargs["prompt"],
                payload=kwargs["payload"],
                schema_hint=kwargs.get("schema_hint"),
            )

        def usage(self):
            return {
                "calls": len(harness.llm_calls) - self._start_calls,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "max_prompt_tokens_per_call": 0,
                "budget": self._max,
            }

    def _wrapped_make_generator(src, *, out_dir, model="", host=None, options=None, env=None, transport=None):
        return orig_make_gen(
            src,
            out_dir=out_dir,
            model=model,
            host=host,
            options=options,
            env=env,
            transport=harness.transport,
        )

    class _Ctx:
        llm = object()

        def dispatch_tool(self, name: str, payload: dict[str, Any]) -> str:
            return harness.dispatch_tool(name, payload)

    with monkeypatch.context() as mp:
        mp.setattr(pllm, "PluginLLMClient", _FakePluginClient)
        mp.setattr(providers, "make_generator", _wrapped_make_generator)
        mp.setattr("urllib.request.urlopen", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("no network")))
        mp.setattr(reel_cli, "_default_data_dir", lambda: default_dir)
        mp.setenv("OPENAI_API_KEY", "sk-test-fake")
        reel_cli.bind(_Ctx())
        try:
            parser = argparse.ArgumentParser()
            reel_cli.register_cli(parser)
            args = parser.parse_args(argv)
            return reel_cli.reel_command(args)
        finally:
            reel_cli.bind(None)


def test_e2e_cli_multi_harness_plan_and_dual_source_render_with_zero_leaks(tmp_path, monkeypatch):
    # 1) Build multi-harness synthetic history with an unmapped sensitive bystander.
    home = tmp_path / "hermes_home"
    _add_hermes_session(home / "state.db", session_id="sess-def", profile="default", topic="queue", base_ts=NOW - 14400)
    _add_hermes_session(home / "profiles" / "claude" / "state.db", session_id="sess-hc", profile="claude", topic="scheduler", base_ts=NOW - 10800)

    bystander = "acmecorp-secret-prod"
    _add_hermes_session(home / "profiles" / bystander / "state.db", session_id="sess-sec", profile=bystander, topic="secret", base_ts=NOW - 18000)
    mem_dir = home / "profiles" / bystander / "memories"
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "MEMORY.md").write_text("Bystander confidential note.\n", encoding="utf-8")

    cc_root = tmp_path / "claude_code_logs" / "proj-a"
    cc_root.mkdir(parents=True, exist_ok=True)
    (cc_root / "session.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "type": role,
                    "timestamp": NOW - 7200 + i * 60.0,
                    "sessionId": "cc-1",
                    "message": {"role": role, "content": text},
                }
            )
            for i, (role, text) in enumerate(_story_turns("parser"))
        ) + "\n",
        encoding="utf-8",
    )

    codex_root = tmp_path / "codex_logs" / "2026"
    codex_root.mkdir(parents=True, exist_ok=True)
    (codex_root / "rollout.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "response_item",
                    "timestamp": NOW - 3600 + i * 60.0,
                    "payload": {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]},
                }
            )
            for i, (role, text) in enumerate(_story_turns("router"))
        ) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("REEL_ALLOW_CUSTOM_STATE", "1")

    # 2) Build character kits + approvals.
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", role="hero")
    _make_kit(assets, "placeholder-claude", "Cinder")
    _make_kit(assets, "placeholder-cc", "Tally")
    _make_kit(assets, "placeholder-codex", "Rook")
    kits = load_kits(REPO_ROOT / "kits", assets / "kits")
    approvals = Approvals.load(assets / "approvals.json")
    for slug in ("placeholder-lead", "kensei", "placeholder-claude", "placeholder-cc", "placeholder-codex"):
        approvals.approve(kits[slug], "anime")

    # 3) Preflight rejects malformed cast.json with 0 LLM calls.
    bad_cast = tmp_path / "bad_cast.json"
    bad_cast.write_text('{"schema_version": 99}', encoding="utf-8")
    harness = _FakeE2EHarness()
    rc_bad = _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=["plan", "--dry-run", "--data-dir", str(tmp_path / "bad_out"), "--cast-config", str(bad_cast)],
        default_dir=tmp_path / "default_data",
    )
    assert rc_bad == 2
    assert harness.llm_calls == []

    # 4) Valid cast.json plans all 4 distinct mapped roles across harnesses.
    cast_file = tmp_path / "cast.json"
    cast_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series_id": "e2e-pilot",
                "assets_dir": str(assets),
                "owner": {"name": "Rowan", "kit": "placeholder-lead", "voice": "dry, practical"},
                "roles": {
                    "hermes:default": {"name": "Kensei", "kit": "kensei"},
                    "hermes:claude": {"name": "Cinder", "kit": "placeholder-claude"},
                    "harness:claude-code": {"name": "Tally", "kit": "placeholder-cc"},
                    "harness:codex": {"name": "Rook", "kit": "placeholder-codex"},
                },
            }
        ),
        encoding="utf-8",
    )

    plan_dir = tmp_path / "plan_1"
    rc_plan = _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "plan", "--dry-run",
            "--data-dir", str(plan_dir),
            "--cast-config", str(cast_file),
            "--all-profiles",
            "--harness", "hermes,claude-code,codex",
            "--harness-root", f"claude-code={tmp_path / 'claude_code_logs'}",
            "--harness-root", f"codex={tmp_path / 'codex_logs'}",
            "--episodes", "4",
        ],
        default_dir=tmp_path / "default_data",
    )
    assert rc_plan == 0
    assert (plan_dir / "series-e2e-pilot.json").is_file()
    series_data = json.loads((plan_dir / "series-e2e-pilot.json").read_text(encoding="utf-8"))
    cast_digest = series_data["cast_digest"]
    assert (plan_dir / "cache" / f"cast-e2e-pilot-{cast_digest}").is_dir()

    seen_agents: set[str] = set()
    first_ir_bytes = (plan_dir / "script-ir-1.json").read_bytes()
    for ep_num in range(1, 5):
        ir_path = plan_dir / f"script-ir-{ep_num}.json"
        assert ir_path.is_file()
        ir_json = json.loads(ir_path.read_text(encoding="utf-8"))
        chars = ir_json["cast_binding"]["characters"]
        if "Rowan" in chars:
            assert chars["Rowan"]["lead"] is True
        seen_agents |= (set(chars) - {"Rowan"})
    assert seen_agents == {"Kensei", "Cinder", "Tally", "Rook"}

    # Warm-cache determinism check: second plan run with warm cache dir makes 0 LLM calls and matches episode 1.
    plan_dir_2 = tmp_path / "plan_2"
    shutil.copytree(plan_dir / "cache", plan_dir_2 / "cache")
    calls_before = len(harness.llm_calls)
    rc_plan_2 = _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "plan", "--dry-run",
            "--data-dir", str(plan_dir_2),
            "--cast-config", str(cast_file),
            "--all-profiles",
            "--harness", "hermes,claude-code,codex",
            "--harness-root", f"claude-code={tmp_path / 'claude_code_logs'}",
            "--harness-root", f"codex={tmp_path / 'codex_logs'}",
            "--episodes", "4",
        ],
        default_dir=tmp_path / "default_data",
    )
    assert rc_plan_2 == 0
    assert len(harness.llm_calls) == calls_before
    assert (plan_dir_2 / "script-ir-1.json").read_bytes() == first_ir_bytes

    # 5) Render episode 1 through both image sources: comfyui (twice -> cache hit) and openai (twice -> cache hit).
    render_dirs: list[Path] = []
    for source_name in ("comfyui", "openai"):
        out_dir = tmp_path / f"render_{source_name}"
        render_dirs.append(out_dir)
        img_before = len(harness.image_calls)
        rc_r1 = _run_plan_and_render_with_harness(
            monkeypatch,
            harness,
            argv=[
                "render",
                "--plan-dir", str(plan_dir),
                "--episode", "1",
                "--style", "anime",
                "--assets-dir", str(assets),
                "--out", str(out_dir),
                "--source", source_name,
                "--no-fillers",
            ],
            default_dir=assets,
        )
        assert rc_r1 == 0
        img_after_r1 = len(harness.image_calls)
        assert img_after_r1 > img_before
        assert all(call["source"] == source_name for call in harness.image_calls[img_before:img_after_r1])
        assert (out_dir / "strip.jpg").is_file()
        assert (out_dir / "render-manifest.json").is_file()
        assert (out_dir / "sidecar.json").is_file()

        manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
        assert manifest["requests"]
        assert all("bindings" in req for req in manifest["requests"])
        sidecar = json.loads((out_dir / "sidecar.json").read_text(encoding="utf-8"))
        assert sidecar["cast_binding"] == json.loads(first_ir_bytes.decode("utf-8"))["cast_binding"]

        # Repeat render for the same source hits cache with 0 new image HTTP calls.
        rc_r2 = _run_plan_and_render_with_harness(
            monkeypatch,
            harness,
            argv=[
                "render",
                "--plan-dir", str(plan_dir),
                "--episode", "1",
                "--style", "anime",
                "--assets-dir", str(assets),
                "--out", str(out_dir),
                "--source", source_name,
                "--no-fillers",
            ],
            default_dir=assets,
        )
        assert rc_r2 == 0
        assert len(harness.image_calls) == img_after_r1

    # 6) Full recursive leak scan across every output artifact (plan_dir + render_dirs).
    forbidden_tokens = (
        bystander,
        f"hermes:{bystander}",
        "hermes:default",
        "hermes:claude",
        "harness:claude-code",
        "harness:codex",
        "Sahil",
    )
    for root_dir in [plan_dir, plan_dir_2, *render_dirs]:
        for path in sorted(root_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() in {".png", ".jpg", ".pdf"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden_tokens:
                assert token not in text, f"{token!r} leaked into {path}"

    # 7) Legacy no-config plan -> render still works unchanged.
    legacy_plan = tmp_path / "legacy_plan"
    legacy_out = tmp_path / "legacy_out"
    _make_kit(assets, "sahil", "Sahil", role="hero")
    kits = load_kits(REPO_ROOT / "kits", assets / "kits")
    Approvals.load(assets / "approvals.json").approve(kits["sahil"], "anime")

    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=["plan", "--dry-run", "--data-dir", str(legacy_plan), "--episodes", "1"],
        default_dir=tmp_path / "default_data_legacy",
    ) == 0
    assert (legacy_plan / SERIES_FILE).is_file()
    legacy_ir = json.loads((legacy_plan / "script-ir-1.json").read_text(encoding="utf-8"))
    assert "cast_binding" not in legacy_ir
    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "render",
            "--plan-dir", str(legacy_plan),
            "--episode", "1",
            "--style", "anime",
            "--assets-dir", str(assets),
            "--out", str(legacy_out),
            "--source", "comfyui",
            "--no-fillers",
        ],
        default_dir=assets,
    ) == 0
    legacy_sidecar = json.loads((legacy_out / "sidecar.json").read_text(encoding="utf-8"))
    assert "cast_binding" not in legacy_sidecar


def test_adding_costar_preserves_chapter_one_renderability(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    kdb = home / "kanban.db"
    kdb.parent.mkdir(parents=True, exist_ok=True)
    kconn = sqlite3.connect(kdb)
    kconn.execute(
        "CREATE TABLE IF NOT EXISTS tasks ("
        "id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT, status TEXT, "
        "priority INTEGER, created_by TEXT, created_at REAL, started_at REAL, completed_at REAL, "
        "tenant TEXT, project_id TEXT, result TEXT)"
    )
    kconn.execute(
        "CREATE TABLE IF NOT EXISTS task_events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, run_id INTEGER, "
        "kind TEXT, payload TEXT, created_at REAL)"
    )
    kconn.execute(
        "CREATE TABLE IF NOT EXISTS task_runs ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, profile TEXT, step_key TEXT, "
        "status TEXT, started_at REAL, ended_at REAL, outcome TEXT, summary TEXT, error TEXT)"
    )
    kconn.execute(
        "CREATE TABLE IF NOT EXISTS task_comments ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, author TEXT, body TEXT, created_at REAL)"
    )
    kconn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, created_by, created_at, started_at, completed_at, result) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "t-ch1",
            "Stabilize queue lock guard",
            "Stabilize queue lock guard.",
            "default",
            "completed",
            1,
            "owner",
            NOW - 7200,
            NOW - 7000,
            NOW - 6000,
            "Fixed the queue lock cleanly with 18 tests passing.",
        ),
    )
    for msg, ts_val in (
        ("Decided to audit the stalled queue lock timeout.", NOW - 6900),
        ("Found a blocked lock timeout across worker retries.", NOW - 6800),
        ("Diagnosed flaky retry failure inside queue worker.", NOW - 6700),
        ("Approved adding clean retry guard and timeout cap.", NOW - 6600),
    ):
        kconn.execute(
            "INSERT INTO task_comments (task_id, author, body, created_at) VALUES (?,?,?,?)",
            ("t-ch1", "default", msg, ts_val),
        )
    kconn.commit()
    kconn.close()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("REEL_ALLOW_CUSTOM_STATE", "1")

    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", role="hero")
    _make_kit(assets, "placeholder-claude", "Cinder")
    kits = load_kits(REPO_ROOT / "kits", assets / "kits")
    approvals = Approvals.load(assets / "approvals.json")
    for slug in ("placeholder-lead", "kensei", "placeholder-claude"):
        approvals.approve(kits[slug], "anime")

    cast_file = tmp_path / "cast.json"
    cast_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series_id": "preserve-ch1",
                "assets_dir": str(assets),
                "owner": {"name": "Rowan", "kit": "placeholder-lead"},
                "roles": {"hermes:default": {"name": "Kensei", "kit": "kensei"}},
            }
        ),
        encoding="utf-8",
    )

    plan_dir = tmp_path / "plan"
    harness = _FakeE2EHarness()
    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "plan", "--dry-run",
            "--data-dir", str(plan_dir),
            "--cast-config", str(cast_file),
            "--episodes", "1",
        ],
        default_dir=tmp_path / "default_data",
    ) == 0
    ir1_before = (plan_dir / "script-ir-1.json").read_bytes()

    # Add co-star Cinder (`hermes:claude`) and plan Chapter 2 in the SAME plan_dir.
    _add_hermes_session(
        home / "profiles" / "claude" / "state.db",
        session_id="sess-ch2",
        profile="claude",
        topic="scheduler",
        base_ts=NOW - 3600,
    )
    cast_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series_id": "preserve-ch1",
                "assets_dir": str(assets),
                "owner": {"name": "Rowan", "kit": "placeholder-lead"},
                "roles": {
                    "hermes:default": {"name": "Kensei", "kit": "kensei"},
                    "hermes:claude": {"name": "Cinder", "kit": "placeholder-claude"},
                },
            }
        ),
        encoding="utf-8",
    )
    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "plan", "--dry-run",
            "--data-dir", str(plan_dir),
            "--cast-config", str(cast_file),
            "--all-profiles",
            "--episodes", "1",
        ],
        default_dir=tmp_path / "default_data",
    ) == 0

    assert (plan_dir / "script-ir-1.json").read_bytes() == ir1_before
    assert (plan_dir / "script-ir-2.json").is_file()

    out1 = tmp_path / "render_ep1"
    out2 = tmp_path / "render_ep2"
    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "render",
            "--plan-dir", str(plan_dir),
            "--episode", "1",
            "--style", "anime",
            "--assets-dir", str(assets),
            "--out", str(out1),
            "--source", "comfyui",
            "--no-fillers",
        ],
        default_dir=assets,
    ) == 0
    assert _run_plan_and_render_with_harness(
        monkeypatch,
        harness,
        argv=[
            "render",
            "--plan-dir", str(plan_dir),
            "--episode", "2",
            "--style", "anime",
            "--assets-dir", str(assets),
            "--out", str(out2),
            "--source", "comfyui",
            "--no-fillers",
        ],
        default_dir=assets,
    ) == 0

    m1 = json.loads((out1 / "render-manifest.json").read_text(encoding="utf-8"))
    m2 = json.loads((out2 / "render-manifest.json").read_text(encoding="utf-8"))
    assert all(req["bindings"].get("Kensei") == "kensei" and "Cinder" not in req["bindings"] for req in m1["requests"])
    assert any(req["bindings"].get("Cinder") == "placeholder-claude" for req in m2["requests"])
