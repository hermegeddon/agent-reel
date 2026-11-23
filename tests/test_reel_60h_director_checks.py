"""Regression tests for reel-60h: shot diversity, explicit shot plans, screen hygiene, and retry alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from adapters.hermes import reel_cli
from canon.director import (
    CLOSE_SHOTS,
    DIRECTOR_PROMPT,
    WORLD_DEFAULT,
    direct,
    no_text,
    parse_shot_plan,
    resolve_shot_plan,
)
from canon.panel_check import CHECK_QUESTION, interpret
from canon.panel_prompt import (
    SCREEN_HYGIENE_RULE,
    PanelRequest,
    build_panel_request,
    build_retry_request,
)


def _pilot_panels(*, include_diagnosis: bool = False, extra_closeup_cue_on: str = "") -> list[dict]:
    if include_diagnosis:
        roles = ("setup", "hidden_problem", "diagnosis", "climax", "aftermath", "win")
    else:
        roles = ("setup", "hidden_problem", "climax", "win")
    panels = []
    for idx, role in enumerate(roles, start=1):
        lines = []
        if role == "hidden_problem" and not include_diagnosis:
            lines = [{"speaker": "Kensei", "text": "My closeup scan found the blocked mutex."}]
        if extra_closeup_cue_on and role == extra_closeup_cue_on:
            lines = [{"speaker": "Kensei", "text": "Let's take a close-up look at this wire."}]
        panels.append({
            "id": f"8631b9fa519b-p{idx}",
            "role": role,
            "caption": f"Step {idx}.",
            "lines": lines,
            "cast": ["Maren", "Kensei"],
            "shot": "wide",
            "setting": "the observatory",
        })
    return panels


def _four_panel_ir(**extra_panel_kwargs) -> dict:
    return {"logline": "Clear Jammed Map Queue Lock", "panels": _pilot_panels(**extra_panel_kwargs)}


def test_short_strip_ensures_closeup_in_rules_and_model_modes():
    ir = _four_panel_ir()
    rules_t = direct(ir, story_text="worker timed out")
    rules_shots = [p.shot for p in rules_t.panels.values()]
    assert any(shot in CLOSE_SHOTS for shot in rules_shots)
    assert rules_t.panels["8631b9fa519b-p2"].shot == "close"
    assert rules_t.panels["8631b9fa519b-p3"].shot in {"full", "long"}

    # Model returning long/medium/full/medium gets a close-up without breaking adjacency or climax.
    def no_closeup_model(**kwargs):
        assert kwargs["prompt"].endswith('"villain_presence": "...", "shot": "...", "angle": "..."}]}')
        assert "shot_plan" not in kwargs["payload"]
        return {
            "villain": {"name": "The Even Swap", "look": "two-faced phantom"},
            "world": "an observatory",
            "panels": [
                {"id": "8631b9fa519b-p1", "location": "control room", "action": "scan", "villain_presence": "lurking", "shot": "long", "angle": "eye"},
                {"id": "8631b9fa519b-p2", "location": "server room", "action": "inspect", "villain_presence": "lurking", "shot": "medium", "angle": "high"},
                {"id": "8631b9fa519b-p3", "location": "core", "action": "clash", "villain_presence": "attacking", "shot": "full", "angle": "low"},
                {"id": "8631b9fa519b-p4", "location": "control room", "action": "celebrate", "villain_presence": "defeated", "shot": "medium", "angle": "eye"},
            ],
            "fillers": [],
        }

    model_t = direct(ir, story_text="worker timed out", complete_json=no_closeup_model)
    assert model_t.panels["8631b9fa519b-p2"].shot == "close"
    assert any("promoted to close-up" in r for r in model_t.repairs)
    pairs = [(p.shot, p.angle) for p in model_t.panels.values()]
    assert all(a != b for a, b in zip(pairs, pairs[1:]))

    # 2-panel strip is left untouched; 6-panel strip already has diagnosis=close.
    two_ir = {"logline": "Short", "panels": _four_panel_ir()["panels"][:2]}
    two_t = direct(two_ir, story_text="timeout")
    assert all(p.shot not in CLOSE_SHOTS for p in two_t.panels.values())
    six_ir = {"logline": "Six", "panels": _pilot_panels(include_diagnosis=True)}
    six_t = direct(six_ir, story_text="timeout")
    assert six_t.panels["8631b9fa519b-p3"].shot == "close"


def test_closeup_cue_prioritizes_cued_panel_over_other_candidates():
    ir = {
        "logline": "Close-up cue priority",
        "panels": [
            {"id": "p1", "role": "setup", "caption": "Opening.", "lines": [], "cast": ["Maren"]},
            {"id": "p2", "role": "hidden_problem", "caption": "Quiet fault.", "lines": [], "cast": ["Maren"]},
            {"id": "p3", "role": "climax", "caption": "Strike.", "lines": [], "cast": ["Maren"]},
            {"id": "p4", "role": "aftermath", "caption": "Calm.", "lines": [{"speaker": "Maren", "text": "Let's zoom in on the seal."}], "cast": ["Maren"]},
        ],
    }
    t = direct(ir, story_text="worker failed")
    assert t.panels["p4"].shot == "close"


def test_parse_and_resolve_shot_plan_variants_and_errors():
    panels = _pilot_panels()
    assert parse_shot_plan("") == {}
    assert parse_shot_plan("long, close, full, medium") == ["long", "close", "full", "medium"]
    assert parse_shot_plan(",close,,") == ["", "close", "", ""]
    assert parse_shot_plan("p2=close, 4=extreme-close") == {"p2": "close", "4": "extreme-close"}

    assert resolve_shot_plan("long,close,full,extreme-close", panels) == {
        "8631b9fa519b-p1": "long",
        "8631b9fa519b-p2": "close",
        "8631b9fa519b-p3": "full",
        "8631b9fa519b-p4": "extreme-close",
    }
    assert resolve_shot_plan(
        "8631b9fa519b-p1=long, p2=close, 3=long, win=extreme-close",
        panels,
    ) == {
        "8631b9fa519b-p1": "long",
        "8631b9fa519b-p2": "close",
        "8631b9fa519b-p3": "long",
        "8631b9fa519b-p4": "extreme-close",
    }

    for bad in ("badshot", "p2=badshot", "p2=close, full", "p99=close", "long,close,full,medium,close"):
        with pytest.raises(ValueError):
            resolve_shot_plan(bad, panels)


def test_shot_plan_overrides_disobeying_model_and_ir_field():
    seen: dict[str, object] = {}

    def disobeying_model(**kwargs):
        seen["prompt"] = kwargs["prompt"]
        seen["payload"] = kwargs["payload"]
        return {
            "villain": {"name": "The Even Swap", "look": "crystal phantom"},
            "world": "an observatory",
            "panels": [
                {"id": f"8631b9fa519b-p{i}", "location": "room", "action": "act", "villain_presence": "lurking", "shot": "medium", "angle": "eye"}
                for i in range(1, 5)
            ],
            "fillers": [],
        }

    ir = _four_panel_ir()
    ir["shot_plan"] = {"p2": "extreme-close", "win": "long"}
    t = direct(ir, story_text="timeout", complete_json=disobeying_model)
    assert seen["payload"]["shot_plan"] == {"8631b9fa519b-p2": "extreme-close", "8631b9fa519b-p4": "long"}
    assert "Follow shot_plan exactly" in str(seen["prompt"])
    assert t.panels["8631b9fa519b-p2"].shot == "extreme-close"
    assert t.panels["8631b9fa519b-p4"].shot == "long"
    # Unpinned climax still follows dynamic climax rules; pinned climax keeps shot and dynamic angle.
    assert t.panels["8631b9fa519b-p3"].shot == "full" and t.panels["8631b9fa519b-p3"].angle == "low"

    t_climax_pin = direct(_four_panel_ir(), story_text="timeout", complete_json=disobeying_model, shot_plan={"p3": "close"})
    assert t_climax_pin.panels["8631b9fa519b-p3"].shot == "close"
    assert t_climax_pin.panels["8631b9fa519b-p3"].angle in {"low", "dutch"}


def test_director_and_screen_prompt_hygiene():
    lowered = DIRECTOR_PROMPT.lower()
    assert "lines of code" in lowered
    assert "terminal logs" in lowered
    assert "error messages" in lowered
    assert "close or extreme-close" in lowered

    assert no_text(WORLD_DEFAULT) == WORLD_DEFAULT
    assert no_text('as the holographic displays show "ALL CLEAR".') == "as the holographic displays."
    assert no_text('a glowing display showing "429".') == "a glowing display."

    pilot_p2 = (
        "Kensei, bathed in the green glow of a diagnostic screen, points a gloved finger at a "
        "specific line of code, his expression grim as Maren peers over his shoulder, her eyes "
        "wide with dawning understanding."
    )
    cleaned_p2 = no_text(pilot_p2)
    assert "diagnostic screen" not in cleaned_p2.lower()
    assert "line of code" not in cleaned_p2.lower()
    assert "luminous abstract screen" in cleaned_p2 and "bright signal pulse" in cleaned_p2

    for sample, forbidden in (
        ("a glowing monitor full of scrolling green code", "scrolling"),
        ("wall displays streaming terminal logs and error messages", "error messages"),
        ("Maren inspects his stack trace and source code", "source code"),
    ):
        result = no_text(sample)
        assert forbidden not in result.lower()

    style = SimpleNamespace(
        key="anime",
        name="anime",
        prompt_block="anime art",
        staging_rules="",
        negative_block="No text.",
        raw_template="",
        composition="",
        impact="",
    )
    screen_req = build_panel_request(
        {
            "id": "p2",
            "role": "hidden_problem",
            "cast": [],
            "caption": "Timeout.",
            "lines": [],
            "setting": "the rig",
        },
        style=style,
        kits={},
    )
    assert SCREEN_HYGIENE_RULE in screen_req.prompt

    outdoor_req = build_panel_request(
        {
            "id": "p1",
            "role": "setup",
            "cast": [],
            "caption": "Quiet morning.",
            "lines": [],
            "setting": "a stone bridge in fog",
        },
        style=style,
        kits={},
    )
    assert SCREEN_HYGIENE_RULE not in outdoor_req.prompt


def test_vision_prompt_alignment_and_retry_request_feedback():
    assert "Scripted characters, left to right as staged: {cast}." in CHECK_QUESTION
    assert "Reference descriptions: {descriptions}." in CHECK_QUESTION
    assert "0..1000" in CHECK_QUESTION and "[ymin, xmin, ymax, xmax]" in CHECK_QUESTION
    assert "outstretched" in CHECK_QUESTION and "weapon" in CHECK_QUESTION.lower()
    assert "signature props" in CHECK_QUESTION and "both faces" in CHECK_QUESTION
    assert "unreadable code squiggles" in CHECK_QUESTION and "MUST return false" in CHECK_QUESTION

    req = PanelRequest(
        panel_id="p2",
        prompt="Base prompt.",
        references=(Path("maren.png"), Path("villain.png")),
        cast=("Maren", "The Even Swap"),
        positions={"Maren": "LEFT"},
        aspect="square",
    )
    bad = interpret(
        "p2",
        ["Maren", "The Even Swap"],
        {
            "on_model": {"Maren": True, "The Even Swap": False},
            "extra_people": 1,
            "text_in_image": True,
            "boxes": [{"label": "protect", "box_2d": [0, 0, 1000, 1000]}],
        },
        require_heads=["Maren"],
    )
    retry_req = build_retry_request(
        req,
        bad,
        {
            "Maren": "matte-charcoal map tube on a signal-vermilion cross-body strap",
            "The Even Swap": "a two-faced phantom of glass shards, one face grinning, one weeping",
        },
    )
    assert "RETRY CORRECTION:" in retry_req.prompt
    assert "Remove all text" in retry_req.prompt
    assert "scripted characters (Maren, The Even Swap)" in retry_req.prompt
    assert "Match The Even Swap accurately" in retry_req.prompt
    assert "face and head clearly visible" in retry_req.prompt
    assert "top 25 percent" in retry_req.prompt
    assert "matte-charcoal map tube" in retry_req.prompt
    assert "one face grinning, one weeping" in retry_req.prompt

    clean = SimpleNamespace(on_model={"Maren": True}, extra_people=0, text_in_image=False, problems=[])
    assert build_retry_request(req, clean) == req


def _make_approved_cast(assets_dir: Path) -> None:
    kit_dir = assets_dir / "kits" / "maren-lead"
    kit_dir.mkdir(parents=True, exist_ok=True)
    sheet = kit_dir / "anime.png"
    Image.new("RGB", (64, 64), (80, 120, 160)).save(sheet)
    (kit_dir / "kit.json").write_text(
        json.dumps({
            "name": "Maren",
            "role": "cast",
            "description": "compact cartographer with matte-charcoal map tube on a red strap",
            "sheets": {"anime": "anime.png"},
            "real_person": False,
            "licence": "test",
        }),
        encoding="utf-8",
    )
    from canon.assets import Approvals, load_kits

    kits = load_kits(assets_dir / "kits")
    Approvals.load(assets_dir / "approvals.json").approve(kits["maren-lead"], "anime")


def _write_test_plan(plan_dir: Path) -> None:
    plan_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for idx, role in enumerate(("setup", "hidden_problem", "climax", "win"), start=1):
        panels.append({
            "id": f"8631b9fa519b-p{idx}",
            "role": role,
            "caption": f"Step {idx}.",
            "caption_source_event_ids": [f"e{idx}"],
            "cast": ["Maren"],
            "lines": [{"speaker": "Maren", "text": f"Check {idx}.", "source_event_ids": [f"e{idx}"]}],
            "shot": "wide",
            "setting": "observatory",
        })
    ir = {
        "episode_id": "ep_60h",
        "logline": "Queue Rescue",
        "cast_binding": {
            "schema_version": 1,
            "series_id": "pilot-60h",
            "cast_digest": "deadbeef12345678",
            "characters": {"Maren": {"kit": "maren-lead", "lead": True}},
        },
        "frame": {
            "conflict": {"key": "two-truths", "label": "conflict"},
            "world": {"key": "observatory", "label": "an observatory"},
            "genre": {"key": "sport", "label": "underdog"},
            "motif": {"key": "chair", "label": "an empty chair"},
            "shot": {"key": "wide", "label": "wide shot"},
            "villain": {
                "key": "the-even-swap",
                "label": "The Even Swap",
                "look": "a two-faced phantom, one face grinning, one weeping",
            },
        },
        "panels": panels,
    }
    (plan_dir / "script-ir-1.json").write_text(json.dumps(ir, indent=1), encoding="utf-8")


def _run_render_cli(
    monkeypatch,
    *,
    plan_dir: Path,
    assets_dir: Path,
    out_dir: Path,
    shots: str = "",
    retries: int = 1,
    vision_eval_fn=None,
):
    from canon import providers

    recorded_prompts: list[str] = []
    call_idx = {"n": 0}

    def fake_generator(prompt: str, refs, aspect: str) -> Path:
        call_idx["n"] += 1
        recorded_prompts.append(prompt)
        out = out_dir.parent / f"_gen_{call_idx['n']}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (640, 480), (40 + (call_idx["n"] * 20) % 200, 60, 80)).save(out)
        return out

    monkeypatch.setattr(providers, "make_generator", lambda *a, **kw: fake_generator)
    monkeypatch.setattr("adapters.hermes.render_cli.prepare_host_tools", lambda: None)

    def fake_dispatch(tool_name: str, payload: dict):
        assert tool_name == "vision_analyze"
        q = str(payload.get("question") or "")
        if "Answer with JSON only. This comic panel has speech balloons" in q:
            return json.dumps({"success": True, "analysis": json.dumps({"misses": [], "occluded": []})})
        img_name = Path(str(payload.get("image_url") or "")).name
        if vision_eval_fn is not None:
            result = vision_eval_fn(img_name)
            return json.dumps({"success": True, "analysis": json.dumps(result)})
        return json.dumps({
            "success": True,
            "analysis": json.dumps({
                "on_model": {"Maren": True, "The Even Swap": True},
                "extra_people": 0,
                "text_in_image": False,
                "boxes": [
                    {"label": "head:Maren", "box_2d": [320, 200, 480, 320]},
                    {"label": "protect", "box_2d": [320, 150, 920, 420]},
                ],
            }),
        })

    argv = [
        "render",
        "--plan-dir", str(plan_dir),
        "--assets-dir", str(assets_dir),
        "--out", str(out_dir),
        "--style", "anime",
        "--source", "gemini",
        "--budget", "12",
        "--retries", str(retries),
        "--no-fillers",
    ]
    if shots:
        argv.extend(["--shots", shots])

    parser = argparse.ArgumentParser()
    reel_cli.register_cli(parser)
    args = parser.parse_args(argv)
    ctx = SimpleNamespace(llm=None, dispatch_tool=fake_dispatch)
    reel_cli.bind(ctx)
    try:
        rc = int(reel_cli.reel_command(args))
    finally:
        reel_cli.bind(None)
    return rc, recorded_prompts


def test_render_cli_shots_flag_and_invalid_validation(tmp_path, monkeypatch, capsys):
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    _make_approved_cast(assets_dir)
    _write_test_plan(plan_dir)

    rc_bad, prompts_bad = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=tmp_path / "bad",
        shots="p2=notashot",
    )
    assert rc_bad == 2 and prompts_bad == []
    assert "invalid --shots" in capsys.readouterr().out

    out_ok = tmp_path / "ok"
    rc_ok, _prompts_ok = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_ok,
        shots="p2=close,p4=extreme-close",
    )
    assert rc_ok == 0
    treatment = json.loads((out_ok / "treatment.json").read_text(encoding="utf-8"))
    by_id = {p["panel_id"]: p["shot"] for p in treatment["panels"]}
    assert by_id["8631b9fa519b-p2"] == "close"
    assert by_id["8631b9fa519b-p4"] == "extreme-close"
    manifest = json.loads((out_ok / "render-manifest.json").read_text(encoding="utf-8"))
    req_by_id = {r["panel_id"]: r["prompt"] for r in manifest["requests"]}
    assert req_by_id["8631b9fa519b-p2"].startswith("Close-up,")
    assert req_by_id["8631b9fa519b-p4"].startswith("Extreme close-up,")


def test_render_cli_retry_feedback_and_no_unimproved_replacement(tmp_path, monkeypatch):
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_cast(assets_dir)
    _write_test_plan(plan_dir)

    def vision_eval(img_name: str) -> dict:
        if img_name == "8631b9fa519b-p2.png":
            return {
                "on_model": {"Maren": True, "The Even Swap": True},
                "extra_people": 0,
                "text_in_image": True,  # 1 problem
                "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
            }
        if img_name.startswith("8631b9fa519b-p2-retry"):
            return {
                "on_model": {"Maren": False, "The Even Swap": True},  # swapped 1 problem -> not better
                "extra_people": 0,
                "text_in_image": False,
                "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
            }
        if img_name == "8631b9fa519b-p3.png":
            return {
                "on_model": {"Maren": True, "The Even Swap": False},
                "extra_people": 0,
                "text_in_image": False,
                "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
            }
        if img_name == "8631b9fa519b-p4.png":
            return {
                "on_model": {"Maren": True, "The Even Swap": True},
                "extra_people": 1,
                "text_in_image": True,  # 2 problems
                "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
            }
        if img_name.startswith("8631b9fa519b-p4-retry"):
            return {
                "on_model": {"Maren": True, "The Even Swap": True},
                "extra_people": 1,
                "text_in_image": False,  # reduced to 1 problem -> retry1 replaces p4, retry2 does not
                "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
            }
        return {
            "on_model": {"Maren": True, "The Even Swap": True},
            "extra_people": 0,
            "text_in_image": False,
            "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
        }

    rc, prompts = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets_dir,
        out_dir=out_dir,
        retries=2,
        vision_eval_fn=vision_eval,
    )
    assert rc == 1  # p2 and p4 never passed
    retry_prompts = [p for p in prompts if "RETRY CORRECTION:" in p]
    assert retry_prompts
    assert len(retry_prompts) == 5
    # Base request stays clean across multiple retry attempts (no stacked suffixes).
    assert all(p.count("RETRY CORRECTION:") == 1 for p in retry_prompts)
    p2_retries = [p for p in retry_prompts if p.startswith("Close-up,") and "Remove all text" in p and "matte-charcoal map tube" in p]
    assert len(p2_retries) == 2
    p3_retries = [p for p in retry_prompts if "Match The Even Swap accurately" in p]
    assert len(p3_retries) == 1 and "one face grinning, one weeping" in p3_retries[0]
    p4_retries = [p for p in retry_prompts if "zero extra background people" in p]
    assert len(p4_retries) == 2 and "Remove all text" in p4_retries[0] and "Remove all text" not in p4_retries[1]

    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    rendered = manifest["report"]["rendered"]
    # Unimproved p2 retry must NOT replace p2.png; passing p3-retry1 and strictly-improved
    # p4-retry1 DO replace their initial attempts (while equal-score p4-retry2 does not).
    assert Path(rendered["8631b9fa519b-p2"]).name == "8631b9fa519b-p2.png"
    assert Path(rendered["8631b9fa519b-p3"]).name == "8631b9fa519b-p3-retry1.png"
    assert Path(rendered["8631b9fa519b-p4"]).name == "8631b9fa519b-p4-retry1.png"


@pytest.mark.parametrize("off_model_verdict", [False, "false"], ids=["boolean", "malformed-string"])
def test_render_cli_rejects_fewer_problems_if_retry_loses_cast_fidelity(tmp_path, monkeypatch, off_model_verdict):
    assets_dir = tmp_path / "assets"
    plan_dir = tmp_path / "plan"
    out_dir = tmp_path / "render"
    _make_approved_cast(assets_dir)
    _write_test_plan(plan_dir)
    panel_id = "8631b9fa519b-p2"

    def vision_eval(img_name: str) -> dict:
        first_attempt = img_name == f"{panel_id}.png"
        retry = img_name == f"{panel_id}-retry1.png"
        return {
            "on_model": {"Maren": off_model_verdict if retry else True, "The Even Swap": True},
            "extra_people": int(first_attempt),
            "text_in_image": first_attempt,
            "boxes": [{"label": "head:Maren", "box_2d": [320, 200, 480, 320]}],
        }

    rc, _ = _run_render_cli(
        monkeypatch, plan_dir=plan_dir, assets_dir=assets_dir, out_dir=out_dir, vision_eval_fn=vision_eval,
    )

    report = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))["report"]
    assert (out_dir / "raw" / f"{panel_id}-retry1.png").is_file()
    assert Path(report["rendered"][panel_id]).name == f"{panel_id}.png"
    assert report["quality_retries"] == 1
    assert rc == 1  # The retained initial image still has two review flags.
