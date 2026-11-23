"""The letterer (M2 P5): balloons, captions and tails placed after the art is made.

Places captions, speech balloons, and speaker-aimed tails on finished art without
covering faces or protected action regions:

- **Heads and ``keep_out`` action zones are hard-blocked** across every placement pass.
- **Anti-lockout whole-panel search** scores reading order and remaining lines together across
  readable font sizes (``36`` down to ``24`` at ``REFERENCE_WIDTH``).
- **Grounded compaction & bounded top headroom:** cleans mid-word ``...`` cuts, removes
  caption-duplicate dialogue lines, omits redundant captions, merges consecutive lines from
  the same speaker when crowded, and adds bounded white top headroom before giving up.
- **Anything unplaceable is reported** in ``report.unplaced`` rather than covering acting.
"""

from __future__ import annotations

import math
import re
import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageFont

from canon.caption import trim_at_boundary

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
BODY_FONT = FONT_DIR / "ComicNeue-Bold.ttf"
TITLE_FONT = FONT_DIR / "Bangers-Regular.ttf"
REFERENCE_WIDTH = 1536  # bake-off r2 panel width the sizes were tuned on

BODY_SIZES = (36, 32, 28, 26, 24)
WRAP_WIDTHS = (16, 20, 26, 32, 40, 48)
EXPAND_RATIOS = (0.18, 0.28, 0.38)
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has", "have",
    "he", "her", "his", "i", "in", "is", "it", "its", "my", "no", "not", "of", "on", "or",
    "our", "she", "that", "the", "their", "there", "they", "this", "to", "was", "we", "were",
    "will", "with", "you", "your",
})


@dataclass
class LetterReport:
    placed: list[str] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)
    caption_box: list[int] | None = None
    balloons: list[dict[str, object]] = field(default_factory=list)
    expanded_top: int = 0
    compacted: bool = False
    caption_omitted: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.unplaced

    def as_dict(self) -> dict[str, object]:
        return {
            "placed": self.placed,
            "unplaced": self.unplaced,
            "caption_box": self.caption_box,
            "balloons": self.balloons,
            "expanded_top": self.expanded_top,
            "compacted": self.compacted,
            "caption_omitted": self.caption_omitted,
            "notes": self.notes,
            "complete": self.complete,
        }


def _px(box: Sequence[float], width: int, height: int) -> list[float]:
    return [box[0] * width, box[1] * height, box[2] * width, box[3] * height]


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, w) * max(0.0, h)


def _pad(box: Sequence[float], amount: float) -> list[float]:
    return [box[0] - amount, box[1] - amount, box[2] + amount, box[3] + amount]


def _segment_hits_box(ax: float, ay: float, bx: float, by: float, box: Sequence[float]) -> bool:
    """Liang-Barsky segment-vs-AABB intersection check."""
    dx, dy = bx - ax, by - ay
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, ax - box[0]), (dx, box[2] - ax), (-dy, ay - box[1]), (dy, box[3] - ay)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def _content_tokens(text: str) -> set[str]:
    tokens = {tok for tok in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(tok) >= 3}
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _token_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) >= 5 and (a.startswith(b[:5]) or b.startswith(a[:5])):
        return True
    return False


def _coverage_ratio(needle: str, haystack: str) -> tuple[float, int]:
    need = _content_tokens(needle)
    have = _content_tokens(haystack)
    if not need or not have:
        return 0.0, 0
    hits = sum(1 for tok in need if any(_token_match(tok, h) for h in have))
    return hits / len(need), hits


def _fix_midword_ellipsis(text: str) -> tuple[str, bool]:
    """Repair planner hard-slicing mid-word '...' at a preceding clause/sentence break."""
    clean = " ".join((text or "").split())
    if not (clean.endswith("...") or clean.endswith("…")):
        return clean, False
    stem = clean[:-3] if clean.endswith("...") else clean[:-1]
    stem = stem.rstrip()
    last_word = stem.rsplit(" ", 1)[-1] if " " in stem else stem
    looks_clipped = len(last_word) <= 4 or "—" in stem or len(stem) >= 75
    if not looks_clipped:
        return clean, False
    repaired = trim_at_boundary(stem, len(stem) - max(1, len(last_word)) + 1)
    if repaired and repaired[-1] not in ".!?":
        repaired += "."
    return repaired or clean, repaired != clean


def _merge_same_speaker(lines: Sequence[tuple[str, str]]) -> list[tuple[str, str]]:
    """Combine consecutive lines by the same speaker into one balloon when cramped."""
    merged: list[tuple[str, str]] = []
    for speaker, text in lines:
        if merged and merged[-1][0].lower() == speaker.lower():
            prev_speaker, prev_text = merged[-1]
            glue = prev_text.rstrip()
            if glue and glue[-1] not in ".!?:;":
                glue += "."
            elif glue.endswith(":"):
                glue = glue[:-1] + "."
            merged[-1] = (prev_speaker, f"{glue} {text}".strip())
        else:
            merged.append((speaker, text))
    return merged


def compact_panel_dialogue(
    caption: str,
    lines: Sequence[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]], bool, bool, list[str]]:
    """Ground-preserving cleanup: repair mid-word cuts, drop caption echoes, omit duplicate captions."""
    notes: list[str] = []
    compacted = False
    cleaned: list[tuple[str, str]] = []
    for speaker, text in lines:
        fixed, was_fixed = _fix_midword_ellipsis(text)
        if was_fixed:
            compacted = True
            notes.append(f"repaired mid-word truncation for {speaker}")
        cleaned.append((speaker, fixed))

    cap_clean = " ".join((caption or "").split())
    if cap_clean and len(cleaned) > 1:
        kept: list[tuple[str, str]] = []
        for idx, (speaker, text) in enumerate(cleaned):
            ratio, hits = _coverage_ratio(text, cap_clean)
            if ratio >= 0.85 and hits >= 3 and (len(cleaned) - (idx - len(kept)) > 1 or kept):
                compacted = True
                notes.append(f"dropped caption-duplicate line for {speaker}")
                continue
            kept.append((speaker, text))
        if kept and len(kept) < len(cleaned):
            # A lead-in line before a dropped continuation often ends with a dangling ':'.
            last_spk, last_txt = kept[-1]
            if last_txt.endswith(":"):
                kept[-1] = (last_spk, last_txt[:-1].rstrip() + ".")
            cleaned = kept

    caption_omitted = False
    if cap_clean and len(cleaned) == 1:
        ratio, hits = _coverage_ratio(cap_clean, cleaned[0][1])
        if ratio >= 0.80 and hits >= 4:
            caption_omitted = True
            compacted = True
            cap_clean = ""
            notes.append("omitted caption already stated verbatim by dialogue")

    return cap_clean, cleaned, compacted, caption_omitted, notes


@dataclass(frozen=True, slots=True)
class LetterStyle:
    """How a style letters its panels. The defaults are the original comic look."""

    balloon: str = "ellipse"  # ellipse | rounded | rect
    balloon_fill: str = "white"
    comms_fill: str = "#e8fbff"
    caption_fill: str = "#fff3b0"
    ink: str = "black"
    outline: float = 1.0  # multiplier on the line weight
    font: str = "comic"  # comic | display
    caption_shape: str = "rect"  # rect | rounded

    @classmethod
    def from_style(cls, lettering: Mapping[str, str] | None) -> LetterStyle:
        """A letter style from a style card's ``lettering`` block; unknown values are ignored."""
        data = dict(lettering or {})
        base = cls()
        choices = {"balloon": ("ellipse", "rounded", "rect"), "font": ("comic", "display"),
                   "caption_shape": ("rect", "rounded")}
        kwargs: dict[str, object] = {}
        for key, value in data.items():
            if key == "outline":
                try:
                    kwargs[key] = min(3.0, max(0.5, float(value)))
                except ValueError:
                    pass
            elif key in choices:
                if value in choices[key]:
                    kwargs[key] = value
            elif key in {"balloon_fill", "comms_fill", "caption_fill", "ink"} and _colour(value):
                kwargs[key] = value
        return replace(base, **kwargs)


def _colour(value: str) -> bool:
    try:
        ImageColor.getrgb(value)
    except ValueError:
        return False
    return True


def _font(path: Path, size: float) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), max(10, int(size)))


def _balloon_dims(tw: float, th: float, scale: float, *, is_rect: bool) -> tuple[int, int]:
    if is_rect:
        return int(tw * 1.12 + 28 * scale), int(th * 1.18 + 22 * scale)
    bw = int(tw * 1.26 + 38 * scale)
    bh = int(th * 1.36 + 26 * scale)
    # Keep ellipse corners well inside the curved border on dense multi-line blocks.
    while (tw / max(1, bw)) ** 2 + (th / max(1, bh)) ** 2 > 0.94:
        bw = int(bw * 1.05 + 4)
        bh = int(bh * 1.05 + 4)
    return bw, bh


def _place_caption(
    draw: ImageDraw.ImageDraw,
    caption: str,
    width: int,
    height: int,
    scale: float,
    body_font: Path,
    protected: Sequence[Sequence[float]],
) -> tuple[list[float] | None, str, ImageFont.FreeTypeFont | None]:
    if not caption:
        return None, "", None
    for size in (26, 24, 22):
        cap_font = _font(body_font, size * scale)
        wrap_candidates = (
            max(28, int(width / (15.5 * scale))),
            max(20, int(width / (21.5 * scale))),
        )
        for wrap_chars in dict.fromkeys(wrap_candidates):
            wrapped = textwrap.fill(caption, wrap_chars)
            tb = draw.multiline_textbbox((0, 0), wrapped, font=cap_font, spacing=4)
            cw, ch = tb[2] - tb[0] + 32 * scale, tb[3] - tb[1] + 22 * scale
            if cw >= width - 10 or ch >= height - 10:
                continue
            corners = (
                [0.0, 0.0, cw, ch],
                [width - cw, 0.0, float(width), ch],
                [0.0, height - ch, cw, float(height)],
                [width - cw, height - ch, float(width), float(height)],
            )
            for corner in corners:
                if not any(_inter(corner, b) > 0 for b in protected):
                    return corner, wrapped, cap_font
    return None, "", None


@dataclass
class _Candidate:
    score: float
    box: list[float]
    wrapped: str
    font: ImageFont.FreeTypeFont
    bw: int
    bh: int
    speaker: str
    head: list[float]
    is_comms: bool


def _collect_candidates(
    draw: ImageDraw.ImageDraw,
    *,
    speaker: str,
    text: str,
    line_idx: int,
    total_lines: int,
    head_px: Mapping[str, list[float]],
    comms_lower: set[str],
    width: int,
    height: int,
    scale: float,
    step: int,
    size: int,
    body_font: Path,
    look: LetterStyle,
    free_px: Sequence[Sequence[float]],
    blockers: Sequence[Sequence[float]],
) -> list[_Candidate]:
    raw_head = head_px.get(speaker.lower())
    is_comms = speaker.lower() in comms_lower or raw_head is None
    head = raw_head if raw_head is not None else [width * 0.02, height * 0.02, width * 0.08, height * 0.12]
    hx = (head[0] + head[2]) / 2
    hy = head[1]
    hcy = (head[1] + head[3]) / 2
    other_heads = [h for name, h in head_px.items() if name != speaker.lower()]
    font = _font(body_font, size * scale)
    is_rect = is_comms or look.balloon != "ellipse"
    remaining = total_lines - 1 - line_idx

    best_by_cell: dict[tuple[int, int], _Candidate] = {}
    cell_w = max(48, width // 8)
    cell_h = max(42, height // 6)
    stride = max(6, int(step * scale * 0.85))
    min_margin = 10

    for wrap in WRAP_WIDTHS:
        wrapped = textwrap.fill(text, wrap)
        tb = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=4, align="center")
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        bw, bh = _balloon_dims(tw, th, scale, is_rect=is_rect)
        if bw >= width - 20 or bh >= height - 20:
            continue
        for x in range(min_margin, max(min_margin + 1, width - bw - min_margin), stride):
            for y in range(min_margin, max(min_margin + 1, height - bh - min_margin), stride):
                box = [float(x), float(y), float(x + bw), float(y + bh)]
                if any(_inter(box, b) > 0 for b in blockers):
                    continue
                in_free = sum(_inter(box, f) for f in free_px) / max(1, bw * bh)
                cx, cy = x + bw / 2, y + bh / 2
                dist = math.hypot(cx - hx, (y + bh - hy) * 1.25)
                tail_hits_other = any(_segment_hits_box(cx, cy, hx, hcy, oh) for oh in other_heads)
                below_speaker = y + bh > head[3] + 20 * scale
                starves_next = remaining > 0 and (height - y) < remaining * max(90.0, bh * 0.85)
                score = (
                    1.5 * dist
                    + 950 * (1 - min(in_free, 1))
                    + 0.35 * y
                    + (380 if below_speaker else 0)
                    + (1600 if starves_next else 0)
                    + (700 if tail_hits_other else 0)
                )
                cell = (x // cell_w, y // cell_h)
                prev = best_by_cell.get(cell)
                if prev is None or score < prev.score:
                    best_by_cell[cell] = _Candidate(
                        score=score, box=box, wrapped=wrapped, font=font, bw=bw, bh=bh,
                        speaker=speaker, head=list(head), is_comms=is_comms,
                    )
    return sorted(best_by_cell.values(), key=lambda c: c.score)[:18]


def _reading_order_ok(cand_box: Sequence[float], prev_box: Sequence[float], scale: float) -> bool:
    dy_row = max(30.0, 30 * scale)
    same_row_band = max(60.0, 60 * scale)
    return (
        cand_box[1] >= prev_box[1] + dy_row
        or (abs(cand_box[1] - prev_box[1]) <= same_row_band and cand_box[0] >= prev_box[2] - 16)
    )


def _solve_sequence(
    per_line: Sequence[Sequence[_Candidate]],
    scale: float,
    *,
    idx: int = 0,
    chosen: list[_Candidate] | None = None,
) -> list[_Candidate] | None:
    acc = [] if chosen is None else chosen
    if idx == len(per_line):
        return list(acc)
    for cand in per_line[idx]:
        if acc and not _reading_order_ok(cand.box, acc[-1].box, scale):
            continue
        if any(_inter(cand.box, prev.box) > 0 for prev in acc):
            continue
        acc.append(cand)
        solved = _solve_sequence(per_line, scale, idx=idx + 1, chosen=acc)
        if solved is not None:
            return solved
        acc.pop()
    return None


def _place_all_lines(
    draw: ImageDraw.ImageDraw,
    lines: Sequence[tuple[str, str]],
    *,
    head_px: Mapping[str, list[float]],
    free_px: Sequence[Sequence[float]],
    protected: Sequence[Sequence[float]],
    capbox: Sequence[float] | None,
    width: int,
    height: int,
    scale: float,
    step: int,
    body_font: Path,
    look: LetterStyle,
    comms_lower: set[str],
) -> list[_Candidate] | None:
    if not lines:
        return []
    base_blockers = list(protected)
    if capbox is not None:
        base_blockers.append(capbox)

    for size in BODY_SIZES:
        per_line = [
            _collect_candidates(
                draw, speaker=speaker, text=text, line_idx=i, total_lines=len(lines),
                head_px=head_px, comms_lower=comms_lower, width=width, height=height,
                scale=scale, step=step, size=size, body_font=body_font, look=look,
                free_px=free_px, blockers=base_blockers,
            )
            for i, (speaker, text) in enumerate(lines)
        ]
        if any(not cands for cands in per_line):
            continue
        solved = _solve_sequence(per_line, scale)
        if solved is not None:
            return solved
    return None


def _can_expand(heads: Mapping[str, Sequence[float]]) -> bool:
    for box in heads.values():
        if len(box) == 4 and max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) >= 0.8:
            return False
    return True


def _draw_tail_and_balloon(
    draw: ImageDraw.ImageDraw,
    cand: _Candidate,
    *,
    all_boxes: Sequence[Sequence[float]],
    other_heads: Sequence[Sequence[float]],
    canvas_h: int,
    scale: float,
    look: LetterStyle,
) -> tuple[int, int]:
    box = cand.box
    bw, bh = cand.bw, cand.bh
    head = cand.head
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    hx = (head[0] + head[2]) / 2
    tx, ty = (hx, (head[1] + head[3]) / 2) if cand.is_comms else (hx, head[1] + (head[3] - head[1]) * 0.45)
    angle = math.atan2(ty - cy, tx - cx)
    ex, ey = cx + (bw / 2) * math.cos(angle), cy + (bh / 2) * math.sin(angle)
    length = math.hypot(tx - ex, ty - ey)
    max_tail = max(480 * scale, canvas_h * 0.65)
    head_margin = _pad(head, max(8.0, 10 * scale))

    # March from the balloon edge toward the speaker head, stopping just outside the
    # speaker's padded head box (or right before another balloon/head in the path).
    probe_step = max(3.0, 4 * scale)
    tail = min(length, max_tail)
    others = [b for b in all_boxes if b is not box]
    drawn = 0.0
    probe = probe_step
    while probe <= tail:
        px, py = ex + probe * math.cos(angle), ey + probe * math.sin(angle)
        if any(b[0] <= px <= b[2] and b[1] <= py <= b[3] for b in others):
            break
        if any(h[0] <= px <= h[2] and h[1] <= py <= h[3] for h in other_heads):
            break
        if not cand.is_comms and head_margin[0] <= px <= head_margin[2] and head_margin[1] <= py <= head_margin[3]:
            drawn = max(drawn, probe)
            break
        drawn = probe
        probe += probe_step
    tail = min(max(drawn, min(26 * scale, length)), max_tail)
    tip = (ex + tail * math.cos(angle), ey + tail * math.sin(angle))

    perp = angle + math.pi / 2
    base = min(22 * scale, bw * 0.16, bh * 0.30)
    inset_x = min(24 * scale, bw * 0.25)
    inset_y = min(18 * scale, bh * 0.25)
    b1 = (cx + (bw / 2 - inset_x) * math.cos(angle) + base * math.cos(perp),
          cy + (bh / 2 - inset_y) * math.sin(angle) + base * math.sin(perp))
    b2 = (cx + (bw / 2 - inset_x) * math.cos(angle) - base * math.cos(perp),
          cy + (bh / 2 - inset_y) * math.sin(angle) - base * math.sin(perp))

    fill = look.comms_fill if cand.is_comms else look.balloon_fill
    outline = max(2, int(4 * scale * look.outline))
    draw.polygon([b1, b2, tip], fill=fill, outline=look.ink, width=outline)
    if cand.is_comms or look.balloon == "rounded":
        draw.rounded_rectangle(box, radius=int(18 * scale), fill=fill, outline=look.ink, width=outline + 1)
    elif look.balloon == "rect":
        draw.rectangle(box, fill=fill, outline=look.ink, width=outline + 1)
    else:
        draw.ellipse(box, fill=fill, outline=look.ink, width=outline + 1)
    draw.polygon([(b1[0] * 0.9 + tip[0] * 0.1, b1[1] * 0.9 + tip[1] * 0.1),
                  (b2[0] * 0.9 + tip[0] * 0.1, b2[1] * 0.9 + tip[1] * 0.1), (cx, cy)], fill=fill)
    draw.multiline_text((cx, cy), cand.wrapped, font=cand.font, fill=look.ink, anchor="mm", spacing=4, align="center")
    return int(tip[0]), int(tip[1])


def letter_panel(
    image: Image.Image,
    *,
    heads: Mapping[str, Sequence[float]],
    free: Sequence[Sequence[float]],
    keep_out: Sequence[Sequence[float]] = (),
    caption: str = "",
    lines: Sequence[tuple[str, str]] = (),
    comms: Sequence[str] = (),
    step: int = 16,
    font_scale: float = 1.0,
    look: LetterStyle | None = None,
    allow_expand: bool = True,
    max_expand_ratio: float = 0.40,
    allow_shorten: bool = True,
) -> tuple[Image.Image, LetterReport]:
    """``font_scale``: QAR-M2-07 - panels shown smaller on a multi-panel page get bigger text,
    so dialogue stays readable at whole-page view."""
    look = look or LetterStyle()
    body_font = TITLE_FONT if look.font == "display" else BODY_FONT
    base_rgb = image.convert("RGB")
    width, orig_height = base_rgb.size
    scale = width / REFERENCE_WIDTH * max(1.0, font_scale)
    comms_lower = {name.lower() for name in comms}

    report = LetterReport()
    active_caption = " ".join((caption or "").split())
    active_lines = [(spk, " ".join((txt or "").split())) for spk, txt in lines]
    if allow_shorten:
        active_caption, active_lines, compacted, caption_omitted, notes = compact_panel_dialogue(
            active_caption, active_lines
        )
        report.compacted = compacted
        report.caption_omitted = caption_omitted
        report.notes.extend(notes)

    def _prepare_canvas(pad_top: int):
        canvas_h = orig_height + pad_top
        if pad_top > 0:
            canvas = Image.new("RGB", (width, canvas_h), "white")
            canvas.paste(base_rgb, (0, pad_top))
        else:
            canvas = base_rgb.copy()
        draw = ImageDraw.Draw(canvas)
        if pad_top > 0:
            rule_w = max(2, int(3 * scale * 0.5))
            draw.line([(0, pad_top), (width, pad_top)], fill=look.ink, width=rule_w)

        def _shift(box: Sequence[float]) -> list[float]:
            x0, y0, x1, y1 = _px(box, width, orig_height)
            return [x0, y0 + pad_top, x1, y1 + pad_top]

        head_px = {name.lower(): _shift(box) for name, box in heads.items()}
        head_blocked = [_pad(b, max(10.0, 14 * scale)) for b in head_px.values()]
        action_blocked = [_pad(_shift(b), max(6.0, 8 * scale)) for b in keep_out]
        protected = head_blocked + action_blocked
        free_px = [_shift(b) for b in free]
        if pad_top > 0:
            free_px.insert(0, [0.0, 0.0, float(width), float(pad_top)])
        return canvas, draw, canvas_h, head_px, protected, free_px

    def _try_layout(pad_top: int, candidate_lines: Sequence[tuple[str, str]]):
        canvas, draw, canvas_h, head_px, protected, free_px = _prepare_canvas(pad_top)
        capbox, wrapped_cap, cap_font = _place_caption(
            draw, active_caption, width, canvas_h, scale, body_font, protected
        )
        if active_caption and capbox is None:
            return None
        solved = _place_all_lines(
            draw, candidate_lines, head_px=head_px, free_px=free_px, protected=protected,
            capbox=capbox, width=width, height=canvas_h, scale=scale, step=step,
            body_font=body_font, look=look, comms_lower=comms_lower,
        )
        if solved is None:
            return None
        return canvas, draw, canvas_h, head_px, capbox, wrapped_cap, cap_font, solved

    chosen = _try_layout(0, active_lines)
    merged_lines = _merge_same_speaker(active_lines) if allow_shorten else list(active_lines)
    if chosen is None and allow_shorten and len(merged_lines) < len(active_lines):
        chosen = _try_layout(0, merged_lines)
        if chosen is not None:
            report.compacted = True
            report.notes.append("merged consecutive lines from the same speaker")

    expansions: list[float] = []
    if chosen is None and allow_expand and max_expand_ratio > 0 and _can_expand(heads):
        expansions = [r for r in EXPAND_RATIOS if r <= max_expand_ratio + 1e-6]
        if not expansions or max_expand_ratio - expansions[-1] >= 0.02:
            expansions.append(max_expand_ratio)
        for ratio in expansions:
            pad_top = max(24, int(round(orig_height * ratio)))
            chosen = _try_layout(pad_top, merged_lines if len(merged_lines) < len(active_lines) else active_lines)
            if chosen is not None:
                report.expanded_top = pad_top
                if len(merged_lines) < len(active_lines):
                    report.compacted = True
                    report.notes.append("merged consecutive lines from the same speaker")
                report.notes.append(f"added {pad_top}px top headroom")
                break

    if chosen is None and allow_shorten and any(len(txt) > 56 for _s, txt in merged_lines):
        for limit in (56, 40):
            short_lines = [(s, trim_at_boundary(txt, limit)) for s, txt in merged_lines]
            pad_top = max(24, int(round(orig_height * expansions[-1]))) if expansions else 0
            chosen = _try_layout(pad_top, short_lines)
            if chosen is not None:
                report.expanded_top = pad_top
                report.compacted = True
                report.notes.append(f"trimmed long dialogue lines at {limit} chars")
                break

    if chosen is None:
        # Strict fallback: place whatever caption and balloons fit with zero protected-area
        # overlap, and leave unplaceable lines in report.unplaced so the caller flags review.
        pad_top = max(24, int(round(orig_height * expansions[-1]))) if expansions else 0
        canvas, draw, canvas_h, head_px, protected, free_px = _prepare_canvas(pad_top)
        report.expanded_top = pad_top
        capbox, wrapped_cap, cap_font = _place_caption(
            draw, active_caption, width, canvas_h, scale, body_font, protected
        )
        if active_caption and capbox is None:
            report.unplaced.append(f"caption: {active_caption}")
        placed_boxes = [capbox] if capbox is not None else []
        placed_cands: list[_Candidate] = []
        partial_lines = merged_lines if len(merged_lines) < len(active_lines) else active_lines
        for idx, (speaker, text) in enumerate(partial_lines):
            picked: _Candidate | None = None
            for size in BODY_SIZES:
                cands = _collect_candidates(
                    draw, speaker=speaker, text=text, line_idx=idx, total_lines=len(partial_lines),
                    head_px=head_px, comms_lower=comms_lower, width=width, height=canvas_h,
                    scale=scale, step=step, size=size, body_font=body_font, look=look,
                    free_px=free_px, blockers=list(protected) + placed_boxes,
                )
                valid = [
                    c for c in cands
                    if not placed_cands or _reading_order_ok(c.box, placed_cands[-1].box, scale)
                ]
                if valid:
                    picked = valid[0]
                    break
            if picked is None:
                report.unplaced.append(f"{speaker}: {text}")
            else:
                placed_boxes.append(picked.box)
                placed_cands.append(picked)
        solved_cands = placed_cands
    else:
        canvas, draw, canvas_h, head_px, capbox, wrapped_cap, cap_font, solved_cands = chosen

    all_boxes = ([capbox] if capbox is not None else []) + [c.box for c in solved_cands]
    for cand in solved_cands:
        other_heads = [h for name, h in head_px.items() if name != cand.speaker.lower()]
        tip_x, tip_y = _draw_tail_and_balloon(
            draw, cand, all_boxes=all_boxes, other_heads=other_heads,
            canvas_h=canvas_h, scale=scale, look=look,
        )
        report.placed.append(cand.speaker)
        report.balloons.append({
            "speaker": cand.speaker,
            "box": [int(v) for v in cand.box],
            "tip": [tip_x, tip_y],
            "comms": cand.is_comms,
            "font_px": int(getattr(cand.font, "size", 0)),
        })

    if active_caption and capbox is not None and cap_font is not None:
        cap_outline = max(2, int(4 * scale * look.outline))
        if look.caption_shape == "rounded":
            draw.rounded_rectangle(capbox, radius=int(14 * scale), fill=look.caption_fill, outline=look.ink, width=cap_outline)
        else:
            draw.rectangle(capbox, fill=look.caption_fill, outline=look.ink, width=cap_outline)
        draw.multiline_text((capbox[0] + 16 * scale, capbox[1] + 8 * scale), wrapped_cap, font=cap_font, fill=look.ink, spacing=4)
        report.caption_box = [int(v) for v in capbox]

    return canvas, report
