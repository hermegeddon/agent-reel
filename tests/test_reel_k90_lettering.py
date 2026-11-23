"""Regression tests for reel-k90: protected acting zones, compaction, headroom, strip aspect, and trimming."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from canon.caption import trim_at_boundary
from canon.director import direct, no_text
from canon.episode import LLMDialogueWriter
from canon.export import compose
from canon.letterer import _inter, compact_panel_dialogue, letter_panel
from canon.panel_check import CHECK_QUESTION, LETTER_QUESTION, TIGHT_QUESTION, interpret, parse_head_box


def _px_box(norm, w, h):
    return [norm[0] * w, norm[1] * h, norm[2] * w, norm[3] * h]


def _blank(w=1264, h=848):
    return Image.new("RGB", (w, h), (70, 90, 120))


def test_keep_out_and_heads_are_never_covered_in_crowded_acting_panels():
    """Panel 1 (face + outstretched hand) & Panel 3 (pose + blade + impact) keep zero box overlap."""
    w, h = 1264, 848
    heads_p1 = {"Maren": [0.17, 0.27, 0.27, 0.42], "Kensei": [0.64, 0.25, 0.80, 0.44]}
    keep_out_p1 = [
        [0.10, 0.40, 0.43, 0.88],  # Maren body & outstretched hand
        [0.35, 0.25, 0.62, 0.60],  # central hologram
        [0.58, 0.40, 0.88, 0.92],  # Kensei armor & console
    ]
    lines_p1 = [
        ("Maren", "Kensei, we must clear this jammed map queue."),
        ("Maren", "Runaway workers could crash the relay."),
        ("Kensei", "Understood."),
        ("Kensei", "I will decide how to clear the map queue and arm the safety timer."),
    ]
    im1, rep1 = letter_panel(
        _blank(w, h),
        heads=heads_p1,
        free=[[0.02, 0.01, 0.96, 0.23]],
        keep_out=keep_out_p1,
        caption="The main relay lock is blocked, threatening a system crash.",
        lines=lines_p1,
        font_scale=2.0,
    )
    assert rep1.complete, rep1.unplaced
    pad_top = rep1.expanded_top
    protected_p1 = [
        [b[0], b[1] + pad_top, b[2], b[3] + pad_top]
        for b in ([_px_box(v, w, h) for v in heads_p1.values()] + [_px_box(k, w, h) for k in keep_out_p1])
    ]
    drawn_p1 = ([rep1.caption_box] if rep1.caption_box else []) + [b["box"] for b in rep1.balloons]
    for dbox in drawn_p1:
        for pbox in protected_p1:
            assert _inter(dbox, pbox) == 0, f"overlap between {dbox} and protected {pbox}"
    assert im1.size == (w, h + pad_top)

    heads_p3 = {"Kensei": [0.27, 0.37, 0.40, 0.49], "Maren": [0.81, 0.44, 0.90, 0.56]}
    keep_out_p3 = [
        [0.03, 0.42, 0.48, 0.95],  # Kensei lunge pose & glowing blade
        [0.38, 0.22, 0.78, 0.88],  # villain & vortex impact burst
    ]
    im3, rep3 = letter_panel(
        _blank(w, h),
        heads=heads_p3,
        free=[[0.02, 0.01, 0.44, 0.34], [0.76, 0.01, 0.98, 0.38]],
        keep_out=keep_out_p3,
        caption="The stalled worker crashed, blocking the main relay lock.",
        lines=[(
            "Kensei",
            "The stalled orphan worker just crashed and blocked the main relay lock—striking clean across the cor...",
        )],
        font_scale=2.0,
    )
    assert rep3.complete, rep3.unplaced
    assert rep3.caption_omitted is True
    assert rep3.caption_box is None
    pad3 = rep3.expanded_top
    protected_p3 = [
        [b[0], b[1] + pad3, b[2], b[3] + pad3]
        for b in ([_px_box(v, w, h) for v in heads_p3.values()] + [_px_box(k, w, h) for k in keep_out_p3])
    ]
    for b in rep3.balloons:
        for pbox in protected_p3:
            assert _inter(b["box"], pbox) == 0
    assert im3.size == (w, h + pad3)


def test_multiline_search_avoids_bottom_lockout():
    """Placing line 1 at the bottom must not lock line 2 out of open upper space."""
    w, h = 1024, 1024
    heads = {"Kensei": [0.24, 0.42, 0.38, 0.56], "Maren": [0.64, 0.39, 0.77, 0.51]}
    keep_out = [[0.04, 0.52, 0.96, 0.99], [0.44, 0.19, 0.57, 0.40]]
    free = [[0.01, 0.01, 0.98, 0.18], [0.01, 0.18, 0.24, 0.40], [0.78, 0.18, 0.99, 0.38]]
    _, rep = letter_panel(
        _blank(w, h),
        heads=heads,
        free=free,
        keep_out=keep_out,
        caption="A timeout error froze the queue.",
        lines=[("Kensei", "Scan hit a lock."), ("Maren", "Resetting worker now.")],
        font_scale=2.0,
    )
    assert rep.complete, rep.unplaced
    assert len(rep.balloons) == 2


def test_compaction_repairs_truncation_drops_caption_echo_and_omits_duplicate_caption():
    cap, lines, compacted, omitted, _notes = compact_panel_dialogue(
        "A timeout error froze the map export queue.",
        [
            ("Kensei", "My closeup scan found the blocked mutex:"),
            ("Kensei", "a timeout error froze the map export queue."),
        ],
    )
    assert compacted is True and omitted is False
    assert cap == "A timeout error froze the map export queue."
    assert lines == [("Kensei", "My closeup scan found the blocked mutex.")]

    cap3, lines3, compacted3, omitted3, _notes3 = compact_panel_dialogue(
        "The stalled worker crashed, blocking the main relay lock.",
        [(
            "Kensei",
            "The stalled orphan worker just crashed and blocked the main relay lock—striking clean across the cor...",
        )],
    )
    assert compacted3 is True and omitted3 is True and cap3 == ""
    assert not lines3[0][1].endswith("cor...")
    assert lines3[0][1] == "The stalled orphan worker just crashed and blocked the main relay lock."


def test_top_headroom_expands_only_within_bounds_and_flags_when_impossible():
    # Crowded non-degenerate panel expands headroom cleanly instead of overlapping keep_out.
    heads = {"Sahil": [0.1, 0.12, 0.25, 0.28]}
    keep_out = [[0.02, 0.10, 0.98, 0.95]]
    out, rep = letter_panel(
        _blank(800, 600),
        heads=heads,
        free=[],
        keep_out=keep_out,
        caption="Relay blocked.",
        lines=[("Sahil", "Holding all worker locks until the safety timer clears.")],
        font_scale=1.5,
    )
    assert rep.complete and rep.expanded_top > 0
    assert out.size[1] == 600 + rep.expanded_top

    # Impossible expansion cap disables enough headroom -> unplaced is flagged, never dropped.
    _, impossible = letter_panel(
        _blank(400, 300),
        heads={"A": [0.05, 0.05, 0.95, 0.95]},
        free=[],
        lines=[("A", "Long line that cannot possibly fit inside a tiny margin.")],
        allow_expand=False,
    )
    assert not impossible.complete and impossible.unplaced


def test_compose_single_column_strip_preserves_mixed_panel_aspects():
    panels = [
        Image.new("RGB", (1264, 1000), (20, 40, 60)),
        Image.new("RGB", (1024, 1024), (40, 60, 80)),
        Image.new("RGB", (1264, 848), (60, 80, 100)),
    ]
    strip = compose(panels, 1, "Test Strip", "", "Agent Reel")
    # Uniform width with proportional heights: p2 scaled to 1264px is 1264px tall, not shrunk to max height.
    assert strip.size[0] > 1264
    expected_min_height = 1000 + 1264 + 848
    assert strip.size[1] > expected_min_height


def test_trim_at_boundary_and_director_cleanup():
    raw = "The stalled orphan worker just crashed and blocked the main relay lock—striking clean across the core gate!"
    trimmed = trim_at_boundary(raw, 100)
    assert "cor..." not in trimmed and len(trimmed) <= 100
    assert trimmed.endswith(".")

    assert no_text('as the holographic displays show "ALL CLEAR".') == "as the holographic displays."
    assert no_text('a glowing display showing "429".') == "a glowing display."

    prefix = " ".join(["obs"] * 72)  # 287 chars; "cold" spans index 300
    t = direct(
        {"logline": "T", "panels": [{"id": "p1", "role": "setup", "caption": "c", "lines": [], "cast": ["Maren"]}]},
        story_text="worker failed",
        complete_json=lambda **_k: {
            "villain": {"name": "The Static", "look": "static mist " * 40},
            "world": f"{prefix} all bathed in the cold",
            "panels": [{"id": "p1", "location": "room", "action": 'displays show "OK".', "villain_presence": "lurking", "shot": "long", "angle": "eye"}],
            "fillers": [],
        },
    )
    assert len(t.world) <= 300
    assert not t.world.endswith("col") and "the col" not in t.world
    assert "show ." not in t.panels["p1"].action

    writer = LLMDialogueWriter(SimpleNamespace(), cast=("Maren",), max_chars=90)
    _cap, out_lines = writer._validate({
        "caption": "Blocked queue.",
        "lines": [{"speaker": "Maren", "text": raw}],
    }, allowed=["Maren"])
    assert len(out_lines[0][1]) <= 90 and "cor..." not in out_lines[0][1]


def test_panel_check_prompts_enforce_xy_coordinates_and_occlusion_check():
    assert "0..1000" in CHECK_QUESTION and "[ymin, xmin, ymax, xmax]" in CHECK_QUESTION
    assert "outstretched" in CHECK_QUESTION and "weapon" in CHECK_QUESTION.lower()
    assert "0..1000" in TIGHT_QUESTION and "[ymin, xmin, ymax, xmax]" in TIGHT_QUESTION
    assert "occluded" in LETTER_QUESTION


def test_interpret_converts_native_box_2d_and_derives_non_overlapping_free_bands():
    answer = {
        "on_model": {"Kensei": True, "Maren": True},
        "extra_people": 0,
        "text_in_image": False,
        "boxes": [
            {"label": "head:Kensei", "box_2d": [456, 354, 536, 392]},
            {"label": "head:Maren", "box_2d": [464, 864, 560, 935]},
            {"label": "protect", "box_2d": [398, 48, 964, 492]},
            {"label": "protect", "box_2d": [188, 500, 998, 998]},
        ],
    }
    check = interpret("p3", ["Kensei", "Maren"], answer)
    assert check.passed, check.problems
    assert check.heads["Kensei"] == [0.354, 0.456, 0.392, 0.536]
    assert check.heads["Maren"] == [0.864, 0.464, 0.935, 0.56]
    assert len(check.keep_out) == 2
    assert check.free
    blocked = list(check.heads.values()) + check.keep_out
    for f in check.free:
        for b in blocked:
            assert _inter(f, b) == 0, f"free {f} overlapped blocked {b}"

    assert parse_head_box({"box_2d": [296, 175, 420, 296]}) == [0.175, 0.296, 0.296, 0.42]
    assert parse_head_box({"head": [0.2, 0.3, 0.3, 0.4]}) == [0.2, 0.3, 0.3, 0.4]
