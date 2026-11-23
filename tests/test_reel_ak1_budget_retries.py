"""Regression tests for reel-ak1: two-pass budget scheduling, retry metrics, and isolated villain cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from adapters.hermes import reel_cli
from canon.export import caption_draft
from canon.panel_prompt import PanelRequest
from canon.render import PanelRenderer, RenderError, RenderReport


def _make_approved_assets(assets_dir: Path) -> None:
    kit_dir = assets_dir / "kits" / "maren"
    kit_dir.mkdir(parents=True, exist_ok=True)
    sheet = kit_dir / "anime.png"
    Image.new("RGB", (64, 64), (80, 120, 160)).save(sheet)
    (kit_dir / "kit.json").write_text(
        json.dumps({
            "name": "Maren",
            "role": "cast",
            "description": "compact cartographer",
            "sheets": {"anime": "anime.png"},
            "real_person": False,
            "licence": "test",
        }),
        encoding="utf-8",
    )
    from canon.assets import Approvals, load_kits

    kits = load_kits(assets_dir / "kits")
    approvals = Approvals.load(assets_dir / "approvals.json")
    approvals.approve(kits["maren"], "anime")


def _write_plan(plan_dir: Path, panels: list[dict[str, object]], *, with_villain: bool = True) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    ir: dict[str, object] = {
        "episode_id": "ep_ak1",
        "logline": "Queue Rescue",
        "cast_binding": {
            "schema_version": 1,
            "series_id": "ak1-series",
            "cast_digest": "deadbeef12345678",
            "characters": {"Maren": {"kit": "maren", "lead": True}},
        },
        "panels": panels,
    }
    if with_villain:
        ir["frame"] = {
            "conflict": {"key": "two-truths", "label": "conflict"},
            "world": {"key": "observatory", "label": "an observatory"},
            "genre": {"key": "sport", "label": "underdog"},
            "motif": {"key": "chair", "label": "an empty chair"},
            "shot": {"key": "wide", "label": "wide shot"},
            "villain": {
                "key": "the-even-swap",
                "label": "The Even Swap",
                "look": "a crystal two-faced phantom",
            },
        }
    (plan_dir / "script-ir-1.json").write_text(json.dumps(ir, indent=1), encoding="utf-8")


def _four_panels() -> list[dict[str, object]]:
    roles = ("setup", "hidden_problem", "climax", "win")
    captions = ("Setup caption.", "Problem caption.", "Climax caption.", "Win caption.")
    return [
        {
            "id": f"p{i}",
            "role": role,
            "caption": cap,
            "caption_source_event_ids": [f"e{i}"],
            "cast": ["Maren"],
            "lines": [{"speaker": "Maren", "text": f"Line {i}.", "source_event_ids": [f"e{i}"]}],
            "shot": "wide",
            "setting": "control room",
        }
        for i, (role, cap) in enumerate(zip(roles, captions), start=1)
    ]


def _run_render(
    monkeypatch,
    *,
    plan_dir: Path,
    assets_dir: Path,
    out_dir: Path,
    budget: int,
    retries: int = 1,
    literal: bool = False,
    save_villain_kit: bool = False,
    vision_fail_fn=None,
    provider_fail_fn=None,
):
    from canon import providers

    calls_log: list[str] = []
    call_counter = {"n": 0}

    def fake_generator(prompt: str, refs, aspect: str) -> Path:
        call_counter["n"] += 1
        idx = call_counter["n"]
        if "Character reference sheet of a comic villain" in prompt:
            tag = "villain"
        else:
            tag = f"panel-{idx}"
        calls_log.append(tag)
        if provider_fail_fn is not None:
            provider_fail_fn(idx, prompt)
        out = out_dir.parent / f"_fake_gen_{idx}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 480), (20 + (idx * 25) % 200, 40, 60)).save(out)
        return out

    def fake_make_generator(source, *, out_dir, model="", host=None, options=None, env=None, transport=None):
        return fake_generator

    def fake_dispatch(tool_name: str, payload: dict):
        assert tool_name == "vision_analyze"
        q = str(payload.get("question") or "")
        if "Answer with JSON only. This comic panel has speech balloons" in q:
            return json.dumps({"success": True, "analysis": json.dumps({"misses": [], "occluded": []})})
        img_name = Path(str(payload.get("image_url") or "")).name
        fail = bool(vision_fail_fn(img_name)) if vision_fail_fn is not None else False
        return json.dumps({
            "success": True,
            "analysis": json.dumps({
                "on_model": {"Maren": True, "The Even Swap": True},
                "extra_people": 0,
                "text_in_image": fail,
                "boxes": [
                    {"label": "head:Maren", "box_2d": [320, 200, 480, 320]},
                    {"label": "protect", "box_2d": [320, 150, 920, 420]},
                ],
            }),
        })

    monkeypatch.setattr(providers, "make_generator", fake_make_generator)
    monkeypatch.setattr("adapters.hermes.render_cli.prepare_host_tools", lambda: None)

    argv = [
        "render",
        "--plan-dir", str(plan_dir),
        "--assets-dir", str(assets_dir),
        "--out", str(out_dir),
        "--style", "anime",
        "--source", "gemini",
        "--budget", str(budget),
        "--retries", str(retries),
        "--no-fillers",
    ]
    if literal:
        argv.append("--literal")
    if save_villain_kit:
        argv.append("--save-villain-kit")

    parser = argparse.ArgumentParser()
    reel_cli.register_cli(parser)
    args = parser.parse_args(argv)
    ctx = SimpleNamespace(llm=None, dispatch_tool=fake_dispatch)
    reel_cli.bind(ctx)
    try:
        rc = int(reel_cli.reel_command(args))
    finally:
        reel_cli.bind(None)
    return rc, calls_log


def test_two_pass_budget_renders_all_panels_before_quality_retries(tmp_path, monkeypatch, capsys):
    """p2 and p3 fail Pass 1 under budget=6 (1 villain + 4 panels + 1 retry): p4 still renders cleanly."""
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_assets(assets_dir)
    _write_plan(plan_dir, _four_panels(), with_villain=True)

    # First attempt of p2 and p3 fail vision check; retry1 of p2 passes.
    rc, calls_log = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_dir,
        budget=6,
        retries=1,
        vision_fail_fn=lambda name: name in {"p2.png", "p3.png"},
    )
    out_text = capsys.readouterr().out
    assert rc == 1  # p3 remained flagged for text_in_image after retry budget was exhausted on p2
    assert calls_log[0] == "villain"
    assert len(calls_log) == 6

    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    sidecar = json.loads((out_dir / "sidecar.json").read_text(encoding="utf-8"))
    report = manifest["report"]
    assert set(report["rendered"].keys()) == {"p1", "p2", "p3", "p4"}
    assert Path(report["rendered"]["p2"]).name == "p2-retry1.png"
    assert Path(report["rendered"]["p3"]).name == "p3.png"
    assert report["calls"] == 6
    assert report["retries"] == 1
    assert report["quality_retries"] == 1
    assert report["transient_retries"] == 0
    assert report["failures"] == []
    assert sidecar["render"]["quality_retries"] == 1
    assert sidecar["render"]["retries"] == 1
    assert "1 retries (1 quality, 0 transient)" in out_text
    assert "Win caption." in (out_dir / "caption.txt").read_text(encoding="utf-8")


def test_villain_sheet_deferred_when_budget_equals_panel_count(tmp_path, monkeypatch):
    """When budget == planned panel count, skip generating a new villain sheet so all panels render."""
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_assets(assets_dir)
    _write_plan(plan_dir, _four_panels(), with_villain=True)

    rc, calls_log = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_dir,
        budget=4,
        retries=1,
    )
    assert rc == 0
    assert calls_log[0] != "villain" and len(calls_log) == 4
    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["report"]["rendered"].keys()) == {"p1", "p2", "p3", "p4"}
    assert not (assets_dir / "kits" / "villain-the-even-swap").exists()
    assert not (assets_dir / "cache" / "villains" / "villain-the-even-swap").exists()


def test_caption_uses_last_rendered_panel_when_budget_cuts_episode_short(tmp_path, monkeypatch):
    """Unrendered tail panels must not leak their win caption into caption.txt."""
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_assets(assets_dir)
    _write_plan(plan_dir, _four_panels(), with_villain=False)

    rc, calls_log = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_dir,
        budget=2,
        retries=0,
        literal=True,
    )
    assert rc == 1 and len(calls_log) == 2
    caption_text = (out_dir / "caption.txt").read_text(encoding="utf-8")
    assert "Problem caption." in caption_text
    assert "Win caption." not in caption_text

    # Standalone caption_draft without rendered_ids preserves full-episode fallback behavior.
    assert "Win caption." in caption_draft("Queue Rescue", {"panels": _four_panels()})


def test_transient_and_quality_retries_both_accounted_in_render_report(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_assets(assets_dir)
    _write_plan(plan_dir, _four_panels()[:1], with_villain=False)

    def transient_once(idx: int, _prompt: str) -> None:
        if idx == 1:
            raise RenderError("503 upstream busy", transient=True)

    rc, _calls = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_dir,
        budget=4,
        retries=1,
        literal=True,
        provider_fail_fn=transient_once,
        vision_fail_fn=lambda name: name == "p1.png",
    )
    assert rc == 0
    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    report = manifest["report"]
    assert report["calls"] == 3  # 1 failed 503 + 1 initial p1 + 1 quality retry p1-retry1
    assert report["transient_retries"] == 1
    assert report["quality_retries"] == 1
    assert report["retries"] == 2

    # Also verify PanelRenderer.can_render respects cache hits vs budget.
    ref = tmp_path / "ref.png"
    ref.write_bytes(b"r")
    req = PanelRequest(panel_id="x", prompt="p", references=(ref,), cast=("Maren",), positions={}, aspect="landscape")
    renderer = PanelRenderer(lambda *_: ref, cache_dir=tmp_path / "cache", budget=1)
    rep = RenderReport()
    renderer.render_one(req, tmp_path / "out", rep)
    assert renderer.calls_made == 1
    assert renderer.is_cached(req) and renderer.can_render(req)
    assert not renderer.can_render(req, attempt_tag="retry1")


def test_villain_sheet_isolates_cache_from_curated_kits_unless_opted_in(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    _make_approved_assets(assets_dir)
    _write_plan(plan_dir, _four_panels()[:1], with_villain=True)

    out1 = tmp_path / "render1"
    rc1, calls1 = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out1,
        budget=6,
        retries=0,
    )
    assert rc1 == 0 and calls1[0] == "villain"
    cached_villain = assets_dir / "cache" / "villains" / "villain-the-even-swap" / "anime.png"
    curated_kit = assets_dir / "kits" / "villain-the-even-swap" / "anime.png"
    assert cached_villain.is_file()
    assert not curated_kit.exists()

    # Re-run reuses cache/villains without generating a new villain image.
    out2 = tmp_path / "render2"
    rc2, calls2 = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out2,
        budget=6,
        retries=0,
    )
    assert rc2 == 0 and calls2 == []

    # Opt-in --save-villain-kit persists the villain sheet + kit.json into assets/kits/.
    out3 = tmp_path / "render3"
    rc3, calls3 = _run_render(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out3,
        budget=6,
        retries=0,
        save_villain_kit=True,
    )
    assert rc3 == 0 and calls3 == []
    assert curated_kit.is_file()
    kit_meta = json.loads((curated_kit.parent / "kit.json").read_text(encoding="utf-8"))
    assert kit_meta["generated"] is True and kit_meta["role"] == "villain"
