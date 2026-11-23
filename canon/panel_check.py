"""Per-panel auto-check (M2 P4). Harness-free: the vision call is injected.

One vision question per panel returns:
- **On-model check** for each scripted character (face, hair, outfit, colours).
- **Count of people** drawn who are not in the script. Bake-off r2 saw the model add Sahil to panels he wasn't in.
- **Text check**: whether any lettering appears in the art.
- **Normalised head boxes** for each scripted character. The letterer aims tails at these.
- **Free regions** where balloons may go, and **keep-out boxes** for key objects.

A panel fails when a character is off-model, when anyone unscripted is drawn, or when text appears. Failed panels are regenerated up to N times by the caller; if they still fail, the episode is flagged, never shipped quietly.

A head box covering more than ``COARSE_HEAD_AREA`` of the frame is too coarse to aim a tail at. Bake-off r2 found 3 of 12 boxes like this, so those are re-asked tightly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

COARSE_HEAD_AREA = 0.06

CHECK_QUESTION = (
    "You are checking one comic panel before lettering. Answer with JSON only, no prose.\n"
    "Scripted characters, left to right as staged: {cast}.\n"
    "Reference descriptions: {descriptions}.\n"
    "Return JSON only: {{\"on_model\": {{NAME: true|false}}, \"extra_people\": INT, "
    "\"text_in_image\": true|false, \"boxes\": [{{\"label\": str, \"box_2d\": [ymin, xmin, ymax, xmax]}}]}}.\n"
    "Use 0..1000 integer coordinates [ymin, xmin, ymax, xmax] for every box_2d.\n"
    "Detect in boxes:\n"
    "- \"head:<NAME>\" tight around each visible scripted character's face/head/helmet only.\n"
    "- \"protect\" boxes covering each character's full acting body, outstretched or gesturing hands, "
    "drawn weapons/blades/tools, glowing impact bursts, and focal displays or props so lettering "
    "never covers faces or key acting.\n"
    "on_model[NAME] = true only when the character matches their reference description, including "
    "key outfit items, signature props, and distinctive traits (such as both faces of a two-faced "
    "creature or a signature carried pack, tube, or weapon when visible in framing).\n"
    "extra_people = people or characters drawn who are not in the scripted list.\n"
    "text_in_image = true ONLY if there are legible words, readable letter sequences, speech bubbles, "
    "or captions drawn in the art; abstract glowing monitor stripes, blurred or unreadable code "
    "squiggles, status bars, waveforms, geometric UI lines, numerals on clock faces or dials, "
    "and lone symbols (?, !, arrows) MUST return false."
)

LETTER_QUESTION = (
    "Answer with JSON only. This comic panel has speech balloons and/or a yellow caption box added.\n"
    "1) Check tail aiming for speakers: {speakers}.\n"
    "2) Check whether ANY speech balloon or caption box overlaps a character's face/head, gesturing or "
    "outstretched hand, active body pose, drawn weapon/blade, or key impact/action.\n"
    "Return {{\"misses\": [SPEAKER, ...], \"occluded\": [SHORT_DESCRIPTION, ...]}} using empty lists when "
    "every tail clearly aims at its speaker and all faces and key action remain unobstructed."
)

TIGHT_QUESTION = (
    "Answer with JSON only. Detect ONLY the head/face/helmet of {name} in 0..1000 integer coordinates "
    "[ymin, xmin, ymax, xmax] (head only, not torso): {{\"box_2d\": [ymin, xmin, ymax, xmax]}}."
)


@dataclass
class PanelCheck:
    panel_id: str
    on_model: dict[str, bool] = field(default_factory=dict)
    extra_people: int = 0
    text_in_image: bool = False
    heads: dict[str, list[float]] = field(default_factory=dict)
    free: list[list[float]] = field(default_factory=list)
    keep_out: list[list[float]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id,
            "passed": self.passed,
            "on_model": self.on_model,
            "extra_people": self.extra_people,
            "text_in_image": self.text_in_image,
            "heads": self.heads,
            "free": self.free,
            "keep_out": self.keep_out,
            "problems": self.problems,
        }


def _box(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))
    y0, y1 = sorted((max(0.0, min(1.0, y0)), max(0.0, min(1.0, y1))))
    if x1 - x0 <= 0.005 or y1 - y0 <= 0.005:
        return None
    return [round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4)]


def _box_2d_to_xy(value: Any) -> list[float] | None:
    """Convert Gemini native ``[ymin, xmin, ymax, xmax]`` (``0..1000``) to ``[x0, y0, x1, y1]``."""
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    if max(abs(ymin), abs(xmin), abs(ymax), abs(xmax)) > 1.0:
        ymin, xmin, ymax, xmax = ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0
    return _box((xmin, ymin, xmax, ymax))


def parse_head_box(answer: Any) -> list[float] | None:
    """Parse either ``{"box_2d": ...}``, ``{"boxes": [...]}``, or legacy ``{"head": ...}``."""
    if not isinstance(answer, Mapping):
        return None
    if answer.get("box_2d") is not None:
        return _box_2d_to_xy(answer.get("box_2d"))
    raw_boxes = answer.get("boxes")
    if isinstance(raw_boxes, list):
        candidates = [item for item in raw_boxes if isinstance(item, Mapping)]
        preferred = sorted(
            candidates,
            key=lambda item: (0 if str(item.get("label") or "").lower().startswith("head") else 1),
        )
        for item in preferred:
            parsed = _box_2d_to_xy(item.get("box_2d"))
            if parsed is not None:
                return parsed
    return _box(answer.get("head"))


def _derive_free_regions(blocked: Sequence[Sequence[float]]) -> list[list[float]]:
    """Derive open background rectangles that have zero overlap with ``blocked``."""
    valid = [b for b in blocked if len(b) == 4]
    if not valid:
        return [[0.0, 0.0, 1.0, 0.30]]

    candidates: list[list[float]] = []
    for sx0, sx1 in ((0.0, 1.0), (0.0, 0.48), (0.52, 1.0), (0.0, 0.35), (0.65, 1.0)):
        overlapping = [b for b in valid if b[2] > sx0 and b[0] < sx1]
        y_ceil = min((b[1] for b in overlapping), default=1.0)
        candidates.append([sx0, 0.0, sx1, y_ceil])

    max_y1 = max(b[3] for b in valid)
    if 1.0 - max_y1 >= 0.14:
        candidates.append([0.0, max_y1, 1.0, 1.0])

    free: list[list[float]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for rect in candidates:
        x0, y0, x1, y1 = (round(v, 4) for v in rect)
        if x1 - x0 < 0.20 or y1 - y0 < 0.12:
            continue
        if any(min(x1, b[2]) > max(x0, b[0]) and min(y1, b[3]) > max(y0, b[1]) for b in valid):
            continue
        key = (x0, y0, x1, y1)
        if key not in seen:
            seen.add(key)
            free.append([x0, y0, x1, y1])
    return free


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def interpret(
    panel_id: str,
    cast: Sequence[str],
    answer: Mapping[str, Any],
    *,
    tighten: Callable[[str], Any] | None = None,
    require_heads: Sequence[str] | None = None,
) -> PanelCheck:
    """``require_heads``: who needs a head box (the speakers); default: the whole cast."""
    check = PanelCheck(panel_id=panel_id)
    raw_on_model = answer.get("on_model")
    on_model: Mapping[str, Any] = raw_on_model if isinstance(raw_on_model, dict) else {}
    lowered = {str(k).lower(): v for k, v in on_model.items()}
    for name in cast:
        verdict = lowered.get(name.lower())
        check.on_model[name] = bool(verdict)
        if verdict is not True:
            check.problems.append(f"{name} is off-model or not found")
    try:
        check.extra_people = int(answer.get("extra_people") or 0)
    except (TypeError, ValueError):
        check.extra_people = 1
        check.problems.append("could not read the extra-people count")
    if check.extra_people > 0:
        check.problems.append(f"{check.extra_people} unscripted people drawn")
    check.text_in_image = bool(answer.get("text_in_image"))
    if check.text_in_image:
        check.problems.append("text drawn in the art")

    raw_boxes = answer.get("boxes")
    heads_lower: dict[str, list[float]] = {}
    parsed_keep_out: list[list[float]] = []
    parsed_free: list[list[float]] = []
    used_boxes = False

    if isinstance(raw_boxes, list):
        for item in raw_boxes:
            if not isinstance(item, Mapping):
                continue
            label = str(item.get("label") or "").strip().lower()
            box = _box_2d_to_xy(item.get("box_2d"))
            if box is None:
                continue
            used_boxes = True
            if label.startswith("head:"):
                char_key = label.split(":", 1)[1].strip()
                if char_key:
                    heads_lower[char_key] = box
            elif label == "free":
                parsed_free.append(box)
            else:
                parsed_keep_out.append(box)

    if not used_boxes:
        raw_heads = answer.get("heads")
        heads: Mapping[str, Any] = raw_heads if isinstance(raw_heads, dict) else {}
        for k, v in heads.items():
            box = _box(v)
            if box is not None:
                heads_lower[str(k).lower()] = box
        parsed_free = [b for b in (_box(v) for v in answer.get("free") or []) if b]
        parsed_keep_out = [b for b in (_box(v) for v in answer.get("keep_out") or []) if b]

    needed = {n.lower() for n in (require_heads if require_heads is not None else cast)}
    for name in cast:
        box = heads_lower.get(name.lower())
        if box is None and name.lower() not in needed:
            continue
        if box is not None and area(box) > COARSE_HEAD_AREA and tighten is not None:
            tight_box = parse_head_box(tighten(name))
            if tight_box is not None:
                box = tight_box
        if box is None:
            check.problems.append(f"no head box for {name}")
            continue
        check.heads[name] = box

    check.keep_out = parsed_keep_out
    if used_boxes and not parsed_free:
        check.free = _derive_free_regions(list(check.heads.values()) + check.keep_out)
    else:
        check.free = parsed_free
    if not check.free:
        check.problems.append("no free space for balloons reported")
    return check


def check_question(cast: Sequence[str], descriptions: Mapping[str, str]) -> str:
    described = "; ".join(f"{name}: {descriptions.get(name, 'see reference')}" for name in cast) or "none"
    return CHECK_QUESTION.format(cast=", ".join(cast) or "none", descriptions=described)
