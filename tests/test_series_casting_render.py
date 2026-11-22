"""Render-side exact kit-slug binding, preflight approval/consent, and export tests (reel-vln.5)."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from adapters.hermes import reel_cli, render_cli
from canon.assets import Approvals, load_kits, load_styles
from canon.director import Treatment, _rule_fillers
from canon.ideas import draw as draw_frame
from canon.panel_prompt import PanelRequest, build_episode_requests
from canon.render import request_key

REPO_ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(size: tuple[int, int] = (320, 220), color: tuple[int, int, int] = (60, 90, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


def _make_kit(
    assets_dir: Path,
    slug: str,
    name: str,
    *,
    role: str = "hero",
    color: tuple[int, int, int] = (60, 90, 140),
    real_person: bool = False,
    owner: bool = False,
    public_consent: bool = False,
    sheets: dict[str, str] | None = None,
) -> Path:
    kit_dir = assets_dir / "kits" / slug
    kit_dir.mkdir(parents=True, exist_ok=True)
    sheet_map = dict(sheets or {"anime": "anime.png"})
    manifest = {
        "name": name,
        "role": role,
        "real_person": real_person,
        "owner": owner,
        "public_consent": public_consent,
        "description": f"{name} ({slug}) reference description.",
        "sheets": sheet_map,
        "licence": "CC0 test fixture",
    }
    (kit_dir / "kit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for rel in sheet_map.values():
        (kit_dir / rel).write_bytes(_png_bytes((64, 64), color=color))
    return kit_dir


def _approve_all(assets_dir: Path, entries: list[tuple[str, str]]) -> None:
    kits = load_kits(REPO_ROOT / "kits", assets_dir / "kits")
    approvals = Approvals.load(assets_dir / "approvals.json")
    for slug, style_key in entries:
        approvals.approve(kits[slug], style_key)


def _sample_ir(
    *,
    characters: tuple[str, ...] = ("Rowan", "Kensei"),
    cast_binding: dict[str, Any] | None = None,
    with_cast_binding: bool = True,
) -> dict[str, Any]:
    hero = characters[0]
    agent = characters[1] if len(characters) > 1 else characters[0]
    roles = ("setup", "hidden_problem", "diagnosis", "climax")
    panels = []
    for idx, role in enumerate(roles, start=1):
        speaker = hero if idx % 2 else agent
        panels.append(
            {
                "id": f"p{idx}",
                "role": role,
                "shot": "medium",
                "pose": "standing",
                "expression": "calm",
                "setting": "the rig at night",
                "cast": list(characters),
                "caption": f"{speaker} checked step {idx}.",
                "lines": [{"speaker": speaker, "text": f"Step {idx} looks steady."}],
                "event_ids": [f"ev-{idx}"],
                "caption_source_event_ids": [f"ev-{idx}"],
            }
        )
    data: dict[str, Any] = {
        "episode_id": "ep-test-1",
        "season": 1,
        "number": 1,
        "format": "strip",
        "logline": "Queue Fixed",
        "episode_time": 1700000000.0,
        "frame": draw_frame(hero, 1),
        "panels": panels,
    }
    if with_cast_binding:
        data["cast_binding"] = (
            cast_binding
            if cast_binding is not None
            else {
                "schema_version": 1,
                "series_id": "render-series",
                "cast_digest": "abcd1234ef567890",
                "characters": {
                    hero: {"kit": "placeholder-lead", "lead": True},
                    **({agent: {"kit": "kensei", "lead": False}} if len(characters) > 1 else {}),
                },
            }
        )
    return data


class _FakeVisionAndProviders:
    """Offline fake vision + fake OpenAI/ComfyUI HTTP transport recording every call."""

    def __init__(self) -> None:
        self.image_calls: list[dict[str, Any]] = []
        self.vision_calls: list[dict[str, Any]] = []
        self._png = _png_bytes((320, 220), color=(90, 120, 160))

    def transport(self, method: str, url: str, headers: dict[str, str], body: bytes | None):
        if "api.openai.com" in url:
            self.image_calls.append({"source": "openai", "url": url, "body": body or b""})
            payload = {"data": [{"b64_json": base64.b64encode(self._png).decode()}]}
            return 200, json.dumps(payload).encode()
        if "/upload/image" in url:
            return 200, json.dumps({"name": "uploaded.png"}).encode()
        if "/prompt" in url:
            self.image_calls.append({"source": "comfyui", "url": url, "body": body or b""})
            return 200, json.dumps({"prompt_id": "job-1"}).encode()
        if "/history/job-1" in url:
            return 200, json.dumps(
                {
                    "job-1": {
                        "status": {"completed": True},
                        "outputs": {"9": {"images": [{"filename": "panel.png", "subfolder": "", "type": "output"}]}},
                    }
                }
            ).encode()
        if "/view" in url:
            return 200, self._png
        raise AssertionError(f"unexpected network attempt to {url}")

    def dispatch(self, tool_name: str, payload: dict[str, Any]) -> str:
        if tool_name != "vision_analyze":
            raise AssertionError(f"unexpected dispatch tool: {tool_name}")
        question = str(payload.get("question", ""))
        self.vision_calls.append(dict(payload))
        if "tail points at the face" in question:
            return json.dumps({"success": True, "analysis": json.dumps({"misses": []})})
        # Extract exact cast/villain keys from CHECK_QUESTION's Reference descriptions dict.
        heads: dict[str, list[float]] = {}
        on_model: dict[str, bool] = {}
        parsed_names: list[str] = []
        desc_prefix = "Reference descriptions: "
        marker_prefix = "Scripted characters, left to right as staged: "
        for line in question.splitlines():
            if line.startswith(desc_prefix):
                raw_desc = line[len(desc_prefix):].strip().rstrip(".")
                for entry in raw_desc.split(";"):
                    name_part = entry.split(":", 1)[0].strip()
                    if name_part and name_part != "none":
                        parsed_names.append(name_part)
                break
        if not parsed_names:
            for line in question.splitlines():
                if line.startswith(marker_prefix):
                    raw = line[len(marker_prefix):].strip().rstrip(".")
                    parsed_names = [part.strip() for part in raw.split(",") if part.strip()]
                    break
        for idx, char_name in enumerate(parsed_names):
            x0 = round(0.10 + (idx % 3) * 0.28, 2)
            heads[char_name] = [x0, 0.25, round(x0 + 0.12, 2), 0.40]
            on_model[char_name] = True
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


def _run_render_cli(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan_dir: Path,
    assets_dir: Path,
    out_dir: Path,
    style: str = "anime",
    source: str = "openai",
    extra_args: list[str] | None = None,
) -> tuple[int, _FakeVisionAndProviders]:
    from canon import providers

    fake = _FakeVisionAndProviders()
    orig_make_generator = providers.make_generator

    def _wrapped_make_generator(src, *, out_dir, model="", host=None, options=None, env=None, transport=None):
        return orig_make_generator(
            src,
            out_dir=out_dir,
            model=model,
            host=host,
            options=options,
            env=env,
            transport=fake.transport,
        )

    def _forbid_urlopen(*_a, **_kw):
        raise AssertionError("network access forbidden in tests")

    class _Ctx:
        llm = None

        def dispatch_tool(self, name: str, payload: dict[str, Any]) -> str:
            return fake.dispatch(name, payload)

    with monkeypatch.context() as mp:
        mp.setenv("OPENAI_API_KEY", "sk-test-fake")
        mp.setattr("urllib.request.urlopen", _forbid_urlopen)
        mp.setattr(providers, "make_generator", _wrapped_make_generator)

        parser = argparse.ArgumentParser()
        reel_cli.register_cli(parser)
        argv = [
            "render",
            "--plan-dir",
            str(plan_dir),
            "--episode",
            "1",
            "--style",
            style,
            "--assets-dir",
            str(assets_dir),
            "--out",
            str(out_dir),
            "--source",
            source,
            "--no-fillers",
        ] + list(extra_args or ())
        args = parser.parse_args(argv)
        rc = render_cli.render_command(args, ctx=_Ctx(), default_dir=lambda: assets_dir)
    return rc, fake


def test_exact_slug_binding_beats_display_name_decoy(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", color=(10, 120, 200))
    _make_kit(assets, "rowan-decoy", "Rowan", color=(220, 20, 20))
    _approve_all(assets, [("placeholder-lead", "anime"), ("rowan-decoy", "anime"), ("kensei", "anime")])

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(_sample_ir()), encoding="utf-8")

    out_dir = tmp_path / "render_out"
    rc, fake = _run_render_cli(monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=out_dir)
    assert rc == 0
    assert len(fake.image_calls) >= 4

    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    assert manifest["requests"]
    for req in manifest["requests"]:
        assert req["bindings"] == {"Rowan": "placeholder-lead", "Kensei": "kensei"}
        assert any("placeholder-lead" in ref for ref in req["references"])
        assert all("rowan-decoy" not in ref for ref in req["references"])


def test_editing_cast_json_after_plan_does_not_alter_rendered_kits(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", color=(10, 120, 200))
    _make_kit(assets, "placeholder-alt", "Rowan", color=(200, 180, 10))
    _approve_all(assets, [("placeholder-lead", "anime"), ("placeholder-alt", "anime"), ("kensei", "anime")])

    cast_file = tmp_path / "cast.json"
    cast_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series_id": "render-series",
                "owner": {"name": "Rowan", "kit": "placeholder-lead"},
                "roles": {"hermes:default": {"name": "Kensei", "kit": "kensei"}},
            }
        ),
        encoding="utf-8",
    )
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(_sample_ir()), encoding="utf-8")

    # Mutate cast.json after plan: must NOT alter what saved script-ir-1.json renders.
    cast_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "series_id": "render-series",
                "owner": {"name": "Rowan", "kit": "placeholder-alt"},
                "roles": {"hermes:default": {"name": "Kensei", "kit": "kensei"}},
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "render_out"
    rc, _fake = _run_render_cli(monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=out_dir)
    assert rc == 0
    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    assert all(req["bindings"]["Rowan"] == "placeholder-lead" for req in manifest["requests"])


def test_malformed_cast_binding_or_missing_kit_fails_preflight_zero_image_calls(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan")
    _approve_all(assets, [("placeholder-lead", "anime"), ("kensei", "anime")])

    bad_bindings = [
        {"schema_version": 2, "series_id": "s", "cast_digest": "abc", "characters": {"Rowan": {"kit": "placeholder-lead", "lead": True}}},
        {"schema_version": 1, "series_id": "s", "cast_digest": "abc", "extra": 1, "characters": {"Rowan": {"kit": "placeholder-lead", "lead": True}}},
        {"schema_version": 1, "series_id": "s", "cast_digest": "abc", "characters": {"Rowan": {"kit": "placeholder-lead", "owner": True}}},
        {"schema_version": 1, "series_id": "s", "cast_digest": "abc", "characters": {"hermes:default": {"kit": "kensei", "lead": False}}},
        {"schema_version": 1, "series_id": "s", "cast_digest": "", "characters": {"Rowan": {"kit": "placeholder-lead", "lead": True}}},
        {"schema_version": 1, "series_id": "s", "cast_digest": "abc", "characters": {"Rowan": {"kit": "placeholder-lead", "lead": True}}},  # missing Kensei
        {"schema_version": 1, "series_id": "s", "cast_digest": "abc", "characters": {"Rowan": {"kit": "nonexistent-kit", "lead": True}, "Kensei": {"kit": "kensei", "lead": False}}},
    ]

    for idx, binding in enumerate(bad_bindings):
        plan_dir = tmp_path / f"plan-{idx}"
        plan_dir.mkdir()
        (plan_dir / "script-ir-1.json").write_text(
            json.dumps(_sample_ir(cast_binding=binding)), encoding="utf-8"
        )
        rc, fake = _run_render_cli(
            monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=tmp_path / f"out-{idx}"
        )
        assert rc == 1
        assert fake.image_calls == []
        assert fake.vision_calls == []


def test_unapproved_or_byte_changed_sheet_fails_preflight_zero_image_calls(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    kit_dir = _make_kit(assets, "placeholder-lead", "Rowan")
    _approve_all(assets, [("kensei", "anime")])

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(_sample_ir()), encoding="utf-8")

    # 1) Unapproved placeholder-lead fails preflight.
    rc1, fake1 = _run_render_cli(
        monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=tmp_path / "out-unapproved"
    )
    assert rc1 == 1
    assert fake1.image_calls == []

    # 2) Approve, then tamper sheet bytes -> SHA-256 mismatch fails preflight.
    _approve_all(assets, [("placeholder-lead", "anime")])
    (kit_dir / "anime.png").write_bytes(_png_bytes((64, 64), color=(250, 10, 10)))
    rc2, fake2 = _run_render_cli(
        monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=tmp_path / "out-tampered"
    )
    assert rc2 == 1
    assert fake2.image_calls == []


def test_approved_identity_fallback_clears_on_experimental_style(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-lead", "Rowan", sheets={"anime": "anime.png"})
    _make_kit(assets, "placeholder-vex", "Vex", role="agent", sheets={"anime": "anime.png"})
    _approve_all(assets, [("placeholder-lead", "anime"), ("placeholder-vex", "anime")])

    ir = _sample_ir(
        characters=("Rowan", "Vex"),
        cast_binding={
            "schema_version": 1,
            "series_id": "exp-style",
            "cast_digest": "1122334455667788",
            "characters": {
                "Rowan": {"kit": "placeholder-lead", "lead": True},
                "Vex": {"kit": "placeholder-vex", "lead": False},
            },
        },
    )
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(ir), encoding="utf-8")

    out_dir = tmp_path / "out_ligne_claire"
    rc, fake = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets,
        out_dir=out_dir,
        style="ligne-claire",
    )
    assert rc == 0
    assert len(fake.image_calls) >= 4


def test_public_flag_refuses_real_person_kit_without_consent(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-real", "Rowan", real_person=True, owner=False, public_consent=False)
    _approve_all(assets, [("placeholder-real", "anime"), ("kensei", "anime")])

    ir = _sample_ir(
        cast_binding={
            "schema_version": 1,
            "series_id": "pub-gate",
            "cast_digest": "deadbeef12345678",
            "characters": {
                "Rowan": {"kit": "placeholder-real", "lead": True},
                "Kensei": {"kit": "kensei", "lead": False},
            },
        }
    )
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(ir), encoding="utf-8")

    rc_pub, fake_pub = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets,
        out_dir=tmp_path / "out_pub",
        extra_args=["--public"],
    )
    assert rc_pub == 1
    assert fake_pub.image_calls == []

    rc_priv, fake_priv = _run_render_cli(
        monkeypatch,
        plan_dir=plan_dir,
        assets_dir=assets,
        out_dir=tmp_path / "out_priv",
    )
    assert rc_priv == 0
    assert len(fake_priv.image_calls) >= 4


def test_request_key_manifest_sidecar_and_rule_fillers(tmp_path, monkeypatch):
    # 1) request_key changes when bindings change even with identical prompt/aspect/references.
    ref = tmp_path / "shared.png"
    ref.write_bytes(_png_bytes((32, 32)))
    req_a = PanelRequest("p1", "prompt", (ref,), ("Rowan",), {"Rowan": "LEFT"}, "landscape", bindings={"Rowan": "kit-a"})
    req_b = PanelRequest("p1", "prompt", (ref,), ("Rowan",), {"Rowan": "LEFT"}, "landscape", bindings={"Rowan": "kit-b"})
    assert request_key(req_a) != request_key(req_b)

    # 2) _rule_fillers picks the cast_binding narrative lead (`Zed`) over earlier-alphabetical `Astra` or hardcoded `Sahil`.
    ir = _sample_ir(
        characters=("Astra", "Zed"),
        cast_binding={
            "schema_version": 1,
            "series_id": "fillers",
            "cast_digest": "9988776655443322",
            "characters": {
                "Zed": {"kit": "placeholder-zed", "lead": True},
                "Astra": {"kit": "placeholder-astra", "lead": False},
            },
        },
    )
    treatment = Treatment(
        world="workshop",
        villain_name="Clockwork Gremlin",
        villain_look="brass gears",
        panels=[],
    )
    # Adjust roles so index 1's upcoming panel is 'climax', triggering the hero closeup filler.
    ir_filler = dict(ir, panels=[dict(p, role=r) for p, r in zip(ir["panels"], ("setup", "hidden_problem", "climax", "win"))])
    fillers = _rule_fillers(treatment, ir_filler)
    assert fillers and any(f.cast == ("Zed",) and "Zed's eyes narrow" in f.action for f in fillers)

    # 3) Manifest bindings & sidecar cast_binding contain zero raw role keys.
    assets = tmp_path / "assets"
    _make_kit(assets, "placeholder-zed", "Zed")
    _make_kit(assets, "placeholder-astra", "Astra", role="agent")
    _approve_all(assets, [("placeholder-zed", "anime"), ("placeholder-astra", "anime")])

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(json.dumps(ir), encoding="utf-8")

    out_dir = tmp_path / "out_manifest"
    rc, _fake = _run_render_cli(monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=out_dir)
    assert rc == 0

    manifest_text = (out_dir / "render-manifest.json").read_text(encoding="utf-8")
    sidecar_text = (out_dir / "sidecar.json").read_text(encoding="utf-8")
    assert "hermes:" not in manifest_text and "harness:" not in manifest_text
    assert "hermes:" not in sidecar_text and "harness:" not in sidecar_text
    sidecar = json.loads(sidecar_text)
    assert sidecar["cast_binding"] == ir["cast_binding"]


def test_legacy_plan_without_cast_binding_renders_unchanged(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    _make_kit(assets, "sahil", "Sahil")
    _approve_all(assets, [("sahil", "anime"), ("kensei", "anime")])

    plan_dir = tmp_path / "legacy_plan"
    plan_dir.mkdir()
    (plan_dir / "script-ir-1.json").write_text(
        json.dumps(_sample_ir(characters=("Sahil", "Kensei"), with_cast_binding=False)),
        encoding="utf-8",
    )
    out_dir = tmp_path / "legacy_out"
    rc, fake = _run_render_cli(monkeypatch, plan_dir=plan_dir, assets_dir=assets, out_dir=out_dir)
    assert rc == 0
    assert len(fake.image_calls) >= 4
    manifest = json.loads((out_dir / "render-manifest.json").read_text(encoding="utf-8"))
    assert all("bindings" not in req for req in manifest["requests"])
    sidecar = json.loads((out_dir / "sidecar.json").read_text(encoding="utf-8"))
    assert "cast_binding" not in sidecar
