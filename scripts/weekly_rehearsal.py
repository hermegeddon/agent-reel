"""Weekly claim/receipt controller shared by the native CLI and inert tests.

No subprocess, credentials, or scheduler is discovered here. The caller supplies
an invocation boundary; only synthetic transports have been qualified.
"""
from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
import json
import re
from pathlib import Path
from typing import Callable

JOB = "weekly-reel"


def explicit_route(value: str) -> bool:
    """Do not let inherited/automatic route selectors masquerade as pinned names."""
    return (bool(value.strip()) and value == value.strip() and
            value.casefold() not in {'auto', 'default', 'inherit', 'inherited', 'main', 'none'})


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _render_inputs(plan: Path) -> dict[str, str | None]:
    """Inputs read by render episode 1 with --literal; bind optional absence too.

    render_command reads the IR and trace (episode_time). Its director-only
    _story_text reads that same trace, and story JSON is only read without
    --literal. Keep this list aligned with the fixed invocation below.
    """
    return {name: _digest(plan / name) if (plan / name).is_file() else None
            for name in ("script-ir-1.json", "episode-1-trace.json")}


def _asset_inputs(assets: Path) -> dict[str, str]:
    """Freeze kit/style/settings bytes, but allow explicit approval repair and caches."""
    repo = Path(__file__).resolve().parents[1]
    inputs = {}
    for label, base in (('user', assets), ('builtin', repo)):
        paths = [p for name in ('kits', 'styles') for p in (base / name).rglob('*') if p.is_file()]
        if label == 'user' and (base / 'settings.json').is_file():
            paths.append(base / 'settings.json')
        for path in sorted(paths):
            inputs[label + '/' + path.relative_to(base).as_posix()] = _digest(path)
    return inputs


def _save(path: Path, state: dict) -> dict:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return state


def run_weekly(*, root: Path, run_key: str, cast: Path, assets: Path,
               home: Path, occurrence: str, since: str, until: str,
               source: str, model: str, writer_provider: str, writer_model: str,
               invoke: Callable[[list[str]], int]) -> dict:
    """One job/week, fail closed on interrupted planning or ambiguous rendering.

    A mkdir claim stays behind on hard termination. Never automatically delete a
    stale claim: operator investigation, not a timeout, determines ownership.
    """
    from datetime import date, datetime, timedelta, timezone

    start, end = date.fromisoformat(since), date.fromisoformat(until)
    instant = datetime.fromisoformat(occurrence.replace('Z', '+00:00'))
    if instant.utcoffset() is None:
        raise ValueError('occurrence requires timezone')
    instant = instant.astimezone(timezone.utc)
    if (start.isoformat() != since or end.isoformat() != until or
            end - start != timedelta(days=7) or end != instant.date() or
            run_key != instant.strftime('%G-W%V')):
        raise ValueError('require seven UTC dates ending on the scheduled occurrence date')
    if source not in {'openai', 'gemini', 'fal'} or not all(
            explicit_route(x) for x in (model, writer_provider, writer_model)):
        raise ValueError('explicit supported source and model bindings required')
    occurrence = instant.isoformat()
    if not re.fullmatch(r"\d{4}-W(?:0[1-9]|[1-4]\d|5[0-3])", run_key):
        raise ValueError("run_key must be an explicit ISO week label")
    root, assets, home, cast = (p.resolve() for p in (root, assets, home, cast))
    for left, right in ((root, home), (root, assets), (assets, home)):
        if left == right or left in right.parents or right in left.parents:
            raise ValueError('weekly-only state and assets must be disjoint from history and each other')
    if root in cast.parents or cast == root:
        raise ValueError('weekly-only state cannot contain cast input')
    if root.exists() and any(p.name != JOB for p in root.iterdir()):
        raise ValueError('weekly-only root contains unrelated state')
    # Stable across weeks, outside every occurrence's evidence directory. Never
    # inherit REEL_STATE_DIR. Refuse redirected job/state paths before any write.
    planner_state = root / JOB / 'planner-state'
    if any(path.resolve() != path for path in
           (planner_state, planner_state / 'memory-facts.json', planner_state / 'aliases.json')):
        raise ValueError('weekly-only planner state cannot be redirected')
    from adapters.hermes.snapshot import DEFAULT_STATE_DIR, state_dir
    for foreign in (DEFAULT_STATE_DIR.resolve(), state_dir().resolve()):
        if root == foreign or root in foreign.parents or foreign in root.parents:
            raise ValueError('weekly-only state must be disjoint from legacy planner state')
    run = root / JOB / run_key
    run.mkdir(parents=True, exist_ok=True)
    lock = run / "claim"
    try:
        lock.mkdir()
    except FileExistsError:
        return {"phase": "busy", "job": JOB, "run_key": run_key}
    status = run / "status.json"
    try:
        binding = {"job": JOB, "run_key": run_key, "cast_sha256": _digest(cast),
                   "assets": str(assets.resolve()), "asset_inputs": _asset_inputs(assets),
                   "home": str(home.resolve()),
                   "planner_state": str(planner_state),
                   "occurrence": occurrence, "since": since, "until": until,
                   "source": source, "model": model,
                   "writer_provider": writer_provider, "writer_model": writer_model}
        if status.exists():
            state = json.loads(status.read_text(encoding="utf-8"))
            if state["binding"] != binding:
                raise ValueError("run inputs changed; refusing replay")
            if state["phase"] in {"complete", "no_story"}:
                return state
            if state["phase"] not in {"planned", "approval_refused"}:
                raise ValueError("incomplete or ambiguous run; operator reconciliation required")
        else:
            if any(p.name != "claim" for p in run.iterdir()):
                raise ValueError("untracked run artifacts; refusing replay")
            state = {"binding": binding, "phase": "planning"}
            _save(status, state)
            captured = io.StringIO()
            with redirect_stdout(captured):
                rc = invoke(["plan", "--dry-run", "--since", since, "--until", until, "--episodes", "1",
                             "--format", "strip", "--tone", "auto", "--cast-config", str(cast),
                             "--assets-dir", str(assets), "--data-dir", str(run / "plan"),
                             "--state-dir", str(planner_state)])
            plan_stdout = captured.getvalue()
            (run / "plan.stdout.txt").write_text(plan_stdout, encoding="utf-8")
            state["plan_rc"] = rc
            ir_path = run / "plan/script-ir-1.json"
            if not ir_path.exists():
                # The current CLI returns before metrics.json on an empty window.
                # Match its WHOLE observed zero-call response, never just rc=1 or
                # missing IR. Any wording change fails closed as plan_failed.
                empty_response = (
                    "episodes: 0  panels: 0\n"
                    "writer: ctx.llm, 0 calls, 0 prompt tokens, 0 completion tokens, "
                    "max 0 prompt tokens in one call\nproblems:\n"
                    "  - no arc could be built from this window\n")
                empty = rc == 1 and plan_stdout == empty_response
                state["phase"] = "no_story" if empty else "plan_failed"
                return _save(status, state)
            if rc != 0:
                state["phase"] = "plan_failed"
                return _save(status, state)
            state.update(phase="planned", ir_sha256=_digest(ir_path),
                         render_inputs=_render_inputs(run / "plan"))
            _save(status, state)

        ir_path = run / "plan/script-ir-1.json"
        if _render_inputs(run / "plan") != state.get("render_inputs"):
            raise ValueError("saved episode changed; refusing render")
        ir = json.loads(ir_path.read_text(encoding="utf-8"))
        # Reuse shipped bound-kit resolver and hash-sensitive approvals before
        # crossing the render-start boundary. The render CLI repeats this gate.
        from canon.assets import Approvals, clear_to_render, load_kits, load_styles
        from canon.panel_prompt import episode_cast
        from canon.series_cast import resolve_bound_kits

        repo = Path(__file__).resolve().parents[1]
        kits = load_kits(repo / "kits", assets / "kits")
        resolved, problems = resolve_bound_kits(ir["cast_binding"], episode_cast(ir), kits)
        problems += clear_to_render(style=load_styles(repo / "styles", assets / "styles")["anime"],
                                    kits=list(resolved.values()),
                                    approvals=Approvals.load(assets / "approvals.json"),
                                    public=False, allow_untested_style=True)
        if problems:
            state.update(phase="approval_refused", problems=problems)
            return _save(status, state)
        state.pop("problems", None)
        state["phase"] = "render_started"
        _save(status, state)
        rc = invoke(["render", "--plan-dir", str(run / "plan"), "--episode", "1",
                     "--style", "anime", "--assets-dir", str(assets), "--source", source, "--model", model,
                     "--out", str(run / "render"), "--budget", "6", "--retries", "0",
                     "--no-fillers", "--literal"])
        state["render_rc"] = rc
        required = ("strip.jpg", "sidecar.json", "render-manifest.json")
        state["phase"] = "complete" if rc == 0 and all(
            (run / "render" / name).is_file() for name in required) else "render_failed"
        return _save(status, state)
    finally:
        lock.rmdir()
