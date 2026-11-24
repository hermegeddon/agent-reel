"""Offline scheduling-equivalent qualification; no cron/gateway is invoked."""
from __future__ import annotations

import json
import socket

import pytest

from canon.assets import Approvals, load_kits
from test_series_casting_e2e import (
    NOW, REPO_ROOT, _FakeE2EHarness, _add_hermes_session, _make_kit,
    _run_plan_and_render_with_harness,
)


@pytest.fixture
def weekly(tmp_path, monkeypatch):
    def deny_network(*_args, **_kwargs):
        raise AssertionError("offline qualification forbids socket connections")

    # Never import the installed registry, even for the failing-before control.
    import sys
    from types import ModuleType

    discovery_calls = []
    tools = ModuleType("tools")
    registry = ModuleType("tools.registry")
    registry.discover_builtin_tools = lambda: discovery_calls.append("discovery")
    tools.registry = registry
    monkeypatch.setitem(sys.modules, "tools", tools)
    monkeypatch.setitem(sys.modules, "tools.registry", registry)
    # Fake dispatch needs no installed host registry or its initialization effects.
    from adapters.hermes import render_cli
    monkeypatch.setattr(render_cli, "prepare_host_tools", lambda: None)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    home = tmp_path / "synthetic-home"
    from datetime import datetime, timezone
    fixture_time = datetime(2026, 11, 26, 12, tzinfo=timezone.utc).timestamp()
    _add_hermes_session(home / "state.db", session_id="weekly-fixture", profile="default",
                        topic="queue", base_ts=fixture_time)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assets = tmp_path / "weekly-assets"
    _make_kit(assets, "weekly-lead", "Rowan", role="hero")
    _make_kit(assets, "weekly-agent", "Cinder")
    kits = load_kits(REPO_ROOT / "kits", assets / "kits")
    for slug in ("weekly-lead", "weekly-agent"):
        Approvals.load(assets / "approvals.json").approve(kits[slug], "anime")
    cast = tmp_path / "weekly-cast.json"
    cast.write_text(json.dumps({
        "schema_version": 1, "series_id": "weekly-synthetic", "assets_dir": str(assets),
        "owner": {"name": "Rowan", "kit": "weekly-lead"},
        "roles": {"hermes:default": {"name": "Cinder", "kit": "weekly-agent"}},
    }), encoding="utf-8")
    harness = _FakeE2EHarness()
    harness.discovery_calls = discovery_calls
    calls = []

    def invoke(argv):
        calls.append(argv)
        return _run_plan_and_render_with_harness(
            monkeypatch, harness, argv=argv, default_dir=tmp_path / "unused-default")

    yield dict(root=tmp_path / "runs", run_key="2026-W48", cast=cast, assets=assets,
               home=home, invoke=invoke, occurrence='2026-11-27T23:00:00+00:00',
               since='2026-11-20', until='2026-11-27', source='openai', model='synthetic-image',
               writer_provider='synthetic', writer_model='synthetic-writer'), harness, calls
    (tmp_path / "offline-evidence.json").write_text(json.dumps({
        "mode": "offline-fake-provider-not-live-cron", "source_root": str(REPO_ROOT),
        "argv": calls, "llm_calls": len(harness.llm_calls),
        "image_calls": len(harness.image_calls), "vision_calls": len(harness.vision_calls),
    }, indent=2) + "\n", encoding="utf-8")


def test_weekly_private_state_is_explicit_and_foreign_bytes_unchanged(weekly, monkeypatch):
    from adapters.hermes import snapshot
    from scripts.weekly_rehearsal import JOB, run_weekly
    args, harness, calls = weekly
    foreign = args['home'].parent / 'foreign-planner'
    default = args['home'].parent / 'legacy-default'
    saved = {}
    for directory in (foreign, default):
        directory.mkdir()
        for name in ('memory-facts.json', 'aliases.json'):
            path = directory / name
            path.write_bytes(b'{"foreign":"must remain unchanged"}')
            saved[path] = path.read_bytes()
    monkeypatch.setenv('REEL_STATE_DIR', str(foreign))
    monkeypatch.setattr(snapshot, 'DEFAULT_STATE_DIR', default)
    roots = []
    original = snapshot.SnapshotStore.__init__
    def observe(self, root=None):
        original(self, root)
        roots.append(self.root.resolve())
    monkeypatch.setattr(snapshot.SnapshotStore, '__init__', observe)
    result = run_weekly(**args)
    assert {p: p.read_bytes() for p in saved} == saved
    private = args['root'] / JOB / 'planner-state'
    assert roots == [private.resolve()]
    assert result['binding']['planner_state'] == str(private.resolve())
    assert (private / 'memory-facts.json').is_file()
    assert (private / 'aliases.json').is_file()
    assert result['phase'] == 'complete'
    assert len(calls) == 2


@pytest.mark.parametrize('target', ['planner-state', 'memory-facts.json', 'aliases.json', 'job'])
def test_private_state_aliases_refuse_before_invocation(weekly, target):
    from scripts.weekly_rehearsal import JOB, run_weekly
    args, harness, calls = weekly
    foreign = args['home'].parent / 'foreign-state'
    foreign.mkdir()
    sentinel = foreign / 'sentinel'
    sentinel.write_bytes(b'foreign data')
    job = args['root'] / JOB
    if target == 'job':
        args['root'].mkdir()
        job.symlink_to(foreign, target_is_directory=True)
    elif target == 'planner-state':
        job.mkdir(parents=True)
        (job / target).symlink_to(foreign, target_is_directory=True)
    else:
        private = job / 'planner-state'
        private.mkdir(parents=True)
        (private / target).symlink_to(sentinel)
    with pytest.raises(ValueError, match='redirected'):
        run_weekly(**args)
    assert calls == []
    assert sentinel.read_bytes() == b'foreign data'
    assert sorted(p.name for p in foreign.iterdir()) == ['sentinel']


@pytest.mark.parametrize('ambient', [False, True])
def test_normal_legacy_cli_state_selection_unchanged(weekly, monkeypatch, ambient):
    from adapters.hermes import snapshot
    args, harness, calls = weekly
    default = args['home'].parent / 'legacy-default'
    env = args['home'].parent / 'legacy-env'
    monkeypatch.setattr(snapshot, 'DEFAULT_STATE_DIR', default)
    if ambient:
        monkeypatch.setenv('REEL_STATE_DIR', str(env))
    else:
        monkeypatch.delenv('REEL_STATE_DIR', raising=False)
    roots = []
    original = snapshot.SnapshotStore.__init__
    def observe(self, root=None):
        original(self, root)
        roots.append((root, self.root.resolve()))
    monkeypatch.setattr(snapshot.SnapshotStore, '__init__', observe)
    rc = args['invoke'](['plan', '--dry-run', '--since', args['since'], '--until', args['until'],
                         '--episodes', '1', '--tone', 'auto', '--cast-config', str(args['cast']),
                         '--assets-dir', str(args['assets']), '--data-dir', str(args['home'].parent / 'legacy-evidence/plan')])
    assert rc == 0
    selected = env if ambient else default
    assert roots == [(None, selected.resolve())]
    assert (selected / 'memory-facts.json').is_file()
    assert (selected / 'aliases.json').is_file()
    assert not args['root'].exists()


def test_planner_state_binding_tamper_refuses_replay(weekly):
    from scripts.weekly_rehearsal import JOB, run_weekly
    args, harness, calls = weekly
    (args['assets'] / 'approvals.json').unlink()
    result = run_weekly(**args)
    assert result['phase'] == 'approval_refused'
    status = args['root'] / JOB / args['run_key'] / 'status.json'
    result['binding']['planner_state'] += '-foreign'
    status.write_text(json.dumps(result), encoding='utf-8')
    with pytest.raises(ValueError, match='run inputs changed'):
        run_weekly(**args)
    assert len(calls) == 1
    assert harness.image_calls == []


def test_representative_weekly_cli_path(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    result = run_weekly(**args)
    assert result["phase"] == "complete"
    assert [argv[0] for argv in calls] == ["plan", "render"]
    plan, render = calls
    assert '--days' not in plan
    assert plan[plan.index('--since') + 1] == args['since']
    assert plan[plan.index('--until') + 1] == args['until']
    assert plan[plan.index("--tone") + 1] == "auto"
    assert render[render.index("--budget") + 1] == "6"
    assert 0 < len(harness.image_calls) <= 6
    run = args["root"] / "weekly-reel" / args["run_key"]
    ir = json.loads((run / "plan/script-ir-1.json").read_text())
    sidecar = json.loads((run / "render/sidecar.json").read_text())
    assert sidecar["cast_binding"] == ir["cast_binding"]
    assert set(ir["cast_binding"]["characters"]) == {"Rowan", "Cinder"}
    assert json.loads((run / "status.json").read_text())["phase"] == "complete"
    before = len(harness.llm_calls), len(harness.image_calls)
    assert run_weekly(**args) == result
    assert (len(harness.llm_calls), len(harness.image_calls)) == before
    assert len(calls) == 2


def test_empty_week_is_no_story_without_render(weekly):
    import sqlite3
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    with sqlite3.connect(args["home"] / "state.db") as conn:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
    result = run_weekly(**args)
    assert result["phase"] == "no_story"
    assert [argv[0] for argv in calls] == ["plan"]
    assert harness.image_calls == []
    assert harness.llm_calls == []
    assert run_weekly(**args) == result
    assert len(calls) == 1


def test_unapproved_sheet_refuses_then_recovers_same_saved_plan(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    approvals = args["assets"] / "approvals.json"
    saved = approvals.read_bytes()
    approvals.unlink()
    result = run_weekly(**args)
    assert result["phase"] == "approval_refused"
    assert harness.image_calls == []
    ir_path = args["root"] / "weekly-reel/2026-W48/plan/script-ir-1.json"
    saved_ir = ir_path.read_bytes()
    trace_path = ir_path.with_name("episode-1-trace.json")
    saved_trace = trace_path.read_bytes()
    trace_arc = json.loads(saved_trace)["arc"]
    import hashlib
    assert result["render_inputs"] == {
        "script-ir-1.json": hashlib.sha256(saved_ir).hexdigest(),
        "episode-1-trace.json": hashlib.sha256(saved_trace).hexdigest(),
    }
    llm_calls = len(harness.llm_calls)
    approvals.write_bytes(saved)
    assert run_weekly(**args)["phase"] == "complete"
    assert len(harness.llm_calls) == llm_calls
    assert ir_path.read_bytes() == saved_ir
    assert trace_path.read_bytes() == saved_trace
    sidecar = json.loads((ir_path.parent.parent / "render/sidecar.json").read_text())
    assert sidecar["episode_time"] == (trace_arc.get("resolved") or trace_arc["opened"])
    assert [argv[0] for argv in calls] == ["plan", "render"]


def test_changed_approved_sheet_is_not_silently_accepted(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    sheet = args["assets"] / "kits/weekly-lead/anime.png"
    sheet.write_bytes(sheet.read_bytes() + b"changed")
    assert run_weekly(**args)["phase"] == "approval_refused"
    assert harness.image_calls == []
    assert len(calls) == 1


def test_concurrent_invocation_has_one_winner(weekly):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    entered, release = Event(), Event()
    invoke = args["invoke"]

    def pause_before_cli(argv):
        entered.set()
        assert release.wait(30), "test contender failed to release owner"
        return invoke(argv)

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(run_weekly, **dict(args, invoke=pause_before_cli))
        try:
            assert entered.wait(10)
            import subprocess
            import sys
            child_code = (
                "import json,sys; from pathlib import Path; "
                "from scripts.weekly_rehearsal import run_weekly; "
                "data=json.loads(sys.argv[1]); "
                "data.update({k:Path(data[k]) for k in ('root','cast','assets','home')}); "
                "print(json.dumps(run_weekly(**data, invoke=lambda argv: sys.exit(99))))")
            child_args = {k: str(v) for k, v in args.items() if k != "invoke"}
            contender = subprocess.run(
                [sys.executable, "-B", "-c", child_code, json.dumps(child_args)],
                cwd=REPO_ROOT, capture_output=True, text=True, timeout=20, check=True)
            assert json.loads(contender.stdout)["phase"] == "busy"
        finally:
            release.set()
        assert first.result(timeout=90)["phase"] == "complete"
    assert [argv[0] for argv in calls] == ["plan", "render"]
    assert 0 < len(harness.image_calls) <= 6


@pytest.mark.parametrize("stage", ["plan", "render"])
def test_interrupted_stage_refuses_ambiguous_replay(weekly, stage):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    invoke = args["invoke"]

    def interrupt(argv):
        if argv[0] == stage:
            raise RuntimeError("synthetic process interruption")
        return invoke(argv)

    with pytest.raises(RuntimeError, match="synthetic process interruption"):
        run_weekly(**dict(args, invoke=interrupt))
    before = len(harness.llm_calls), len(harness.image_calls), len(calls)
    with pytest.raises(ValueError, match="operator reconciliation"):
        run_weekly(**args)
    assert (len(harness.llm_calls), len(harness.image_calls), len(calls)) == before


def test_failed_render_refuses_retry_even_after_output_was_written(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    invoke = args["invoke"]

    def fail_after_render(argv):
        rc = invoke(argv)
        return 1 if argv[0] == "render" else rc

    result = run_weekly(**dict(args, invoke=fail_after_render))
    assert result["phase"] == "render_failed"
    assert harness.image_calls
    before = len(harness.image_calls), len(calls)
    with pytest.raises(ValueError, match="operator reconciliation"):
        run_weekly(**args)
    assert (len(harness.image_calls), len(calls)) == before


def test_malformed_cast_refusal_is_not_a_no_story_success(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    args["cast"].write_text('{"schema_version":99}', encoding="utf-8")
    assert run_weekly(**args)["phase"] == "plan_failed"
    assert harness.llm_calls == harness.image_calls == []
    assert len(calls) == 1


def test_stale_claim_is_never_stolen(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    claim = args["root"] / "weekly-reel/2026-W48/claim"
    claim.mkdir(parents=True)
    assert run_weekly(**args)["phase"] == "busy"
    assert claim.is_dir()
    assert calls == []


def test_changed_plan_refuses_recovery(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    (args["assets"] / "approvals.json").unlink()
    assert run_weekly(**args)["phase"] == "approval_refused"
    ir = args["root"] / "weekly-reel/2026-W48/plan/script-ir-1.json"
    ir.write_bytes(ir.read_bytes() + b" ")
    with pytest.raises(ValueError, match="saved episode changed"):
        run_weekly(**args)
    assert harness.image_calls == []
    assert len(calls) == 1


def test_offline_render_never_discovers_host_tools(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    assert run_weekly(**args)["phase"] == "complete"
    assert [argv[0] for argv in calls] == ["plan", "render"]
    assert harness.discovery_calls == []


@pytest.mark.parametrize("filename,mutation", [
    ("script-ir-1.json", "delete"),
    ("episode-1-trace.json", "delete"),
    ("episode-1-trace.json", "bytes"),
    ("episode-1-trace.json", "resolved"),
])
def test_render_input_drift_refuses_before_render(weekly, filename, mutation):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    approvals = args["assets"] / "approvals.json"
    saved_approvals = approvals.read_bytes()
    approvals.unlink()
    assert run_weekly(**args)["phase"] == "approval_refused"
    run = args["root"] / "weekly-reel" / args["run_key"]
    path = run / "plan" / filename
    original = path.read_bytes()
    if mutation == "delete":
        path.unlink()
    elif mutation == "bytes":
        path.write_bytes(original + b" ")
    else:
        trace = json.loads(original)
        trace["arc"]["resolved"] = 946684800.0
        path.write_text(json.dumps(trace), encoding="utf-8")
    approvals.write_bytes(saved_approvals)
    status = (run / "status.json").read_bytes()
    before = len(harness.llm_calls), len(harness.image_calls), len(calls)
    with pytest.raises(ValueError, match="saved episode changed"):
        run_weekly(**args)
    assert (len(harness.llm_calls), len(harness.image_calls), len(calls)) == before
    assert (run / "status.json").read_bytes() == status
    assert not (run / "render").exists()
    path.write_bytes(original)
    assert run_weekly(**args)["phase"] == "complete"


def test_absent_trace_cannot_appear_on_recovery(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    invoke = args["invoke"]
    run = args["root"] / "weekly-reel" / args["run_key"]
    trace = run / "plan/episode-1-trace.json"

    def plan_without_trace(argv):
        rc = invoke(argv)
        if argv[0] == "plan":
            trace.unlink()
        return rc

    approvals = args["assets"] / "approvals.json"
    saved = approvals.read_bytes()
    approvals.unlink()
    assert run_weekly(**dict(args, invoke=plan_without_trace))["phase"] == "approval_refused"
    trace.write_text('{"arc":{"resolved":946684800.0}}', encoding="utf-8")
    approvals.write_bytes(saved)
    with pytest.raises(ValueError, match="saved episode changed"):
        run_weekly(**args)
    assert [argv[0] for argv in calls] == ["plan"]
    assert harness.image_calls == []
    trace.unlink()
    assert run_weekly(**args)["phase"] == "complete"


def test_changed_cast_refuses_completed_run_replay(weekly):
    from scripts.weekly_rehearsal import run_weekly

    args, harness, calls = weekly
    assert run_weekly(**args)["phase"] == "complete"
    args["cast"].write_bytes(args["cast"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="run inputs changed"):
        run_weekly(**args)
    assert len(calls) == 2
