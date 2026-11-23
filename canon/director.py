"""The director (M2 story pass, owner decisions V1/V3, 25/10/26).

The script says what happened; the director decides how it LOOKS. Per panel it picks:
- a dramatised **location**
- an **action**
- the **villain**'s presence
- a **shot size** and **camera angle**

Following docs/STORY-CRAFT.md:
- the episode's problem becomes a villain (orphaned processes -> The Orphans)
- the climax is the biggest, most dynamic panel
- no two consecutive panels repeat a shot

Truth boundary: the treatment only changes the picture.
- Captions and lines are never edited here; they stay grounded by canon.grounding.
- Treatments must not ask for text in the art, so quotes and "sign that says ..." requests are stripped.

Harness-free: the model call is an injected ``complete_json``. Without a model, a rule-based treatment is used, so a render never depends on the director being available.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from canon.caption import trim_at_boundary

SHOTS = ("extreme-long", "long", "full", "medium", "close", "extreme-close")
CLOSE_SHOTS = frozenset({"close", "extreme-close"})
ANGLES = ("eye", "low", "high", "dutch", "overhead")

SHOT_WORDS = {
    "extreme-long": "Extreme long shot",
    "long": "Long shot",
    "full": "Full shot",
    "medium": "Medium shot",
    "close": "Close-up",
    "extreme-close": "Extreme close-up",
}
ANGLE_WORDS = {
    "eye": "eye-level camera",
    "low": "low-angle camera looking up (power)",
    "high": "high-angle camera looking down (vulnerability)",
    "dutch": "tilted Dutch-angle camera (chaos)",
    "overhead": "overhead bird's-eye camera",
}

#: Role defaults from the craft notes. They are used when there is no model, and as the repair target when a model breaks the rules.
ROLE_DEFAULTS = {
    "setup": ("extreme-long", "eye", "absent"),
    "hidden_problem": ("full", "high", "lurking"),
    "diagnosis": ("close", "dutch", "lurking"),
    "climax": ("full", "low", "attacking"),
    "aftermath": ("medium", "eye", "defeated"),
    "win": ("long", "low", "defeated"),
}

#: The problem-kind -> villain library. Fictional creatures only, never a real person.
VILLAINS = {
    "process": ("The Orphans", "small gremlin-like orphaned processes with glowing eyes that hide in server racks and chew cables"),
    "limit": ("The Toll Wraith", "a hooded spectral gatekeeper made of red warning lights, holding a lantern that drains energy"),
    "failure": ("The Crash Golem", "a hulking golem of broken circuit boards and sparking cables"),
    "defect": ("The Glitch Swarm", "a swarm of flickering pixelated insects that corrupt whatever they touch"),
    "blocker": ("The Gatekeeper", "a towering armoured sentinel blocking a glowing doorway"),
    "default": ("The Static", "a shapeless storm of black static with glowing red eyes"),
}

_PATTERNS = (
    ("process", re.compile(r"\b(orphan|zombie|process|gateway|daemon|worker|session)s?\b", re.I)),
    ("limit", re.compile(r"\b(rate[- ]?limit|quota|usage limit|429|403|budget|limit)\b", re.I)),
    ("defect", re.compile(r"\b(bug|regression|leak|corrupt|drift|flaky)\w*\b", re.I)),
    ("blocker", re.compile(r"\b(blocked|stuck|timeout|timed out|refused|hung)\b", re.I)),
    ("failure", re.compile(r"\b(fail|broke|crash|error|exception)\w*\b", re.I)),
)

#: Role -> (motion, impact, reaction, camera) for the rule-based treatment (M3).
ROLE_ACTION = {
    "setup": ("still air, dust drifting across frame", "", "the pair scan the horizon", "slow crane down into the scene"),
    "hidden_problem": ("something moves at the edge of frame", "a thin crack spreads", "the heroes' shoulders tense", "slow push in"),
    "diagnosis": ("a hand snaps out to point", "the surface splits open", "eyes widen", "rapid zoom to the detail"),
    "climax": ("bodies and debris hurtling left to right", "the blow lands and the ground buckles", "the villain recoils, shattered", "whip pan into the strike"),
    "aftermath": ("dust settling, fragments falling", "the last shard drops", "exhausted breathing, a slow nod", "slow pull back"),
    "win": ("the pair straighten, cloaks settling", "the horizon clears", "a shared look of relief", "crane up and away"),
}


#: The default shape. Its treatment is the original one, untouched.
DEFAULT_TYPE = "struggle-to-resolution"
#: Shapes with no creature in them: the opponent is a plan, a number, the clock.
NO_VILLAIN_TYPES = frozenset({"pivot", "training-montage"})

#: Per-shape overrides of ROLE_DEFAULTS: (shot, angle, villain presence).
TYPE_ROLE_DEFAULTS: dict[str, dict[str, tuple[str, str, str]]] = {
    # The cause stays unseen until the reveal, so the camera hunts for it.
    "mystery": {
        "setup": ("close", "dutch", "absent"),
        "hidden_problem": ("extreme-close", "eye", "absent"),
        "diagnosis": ("medium", "high", "absent"),
        "climax": ("full", "low", "attacking"),
        "aftermath": ("medium", "eye", "defeated"),
    },
    # The plan is the opponent, so no creature, and the turn is a quiet decision.
    "pivot": {
        "setup": ("long", "eye", "absent"),
        "hidden_problem": ("medium", "high", "absent"),
        "diagnosis": ("close", "eye", "absent"),
        "climax": ("full", "overhead", "absent"),
        "aftermath": ("medium", "low", "absent"),
        "win": ("long", "eye", "absent"),
    },
    # The rival is in the room from the first panel.
    "rivalry": {
        "setup": ("medium", "eye", "lurking"),
        "hidden_problem": ("full", "low", "attacking"),
        "diagnosis": ("close", "dutch", "lurking"),
        "aftermath": ("medium", "eye", "lurking"),
        "win": ("long", "low", "defeated"),
    },
    # Same hero and place, again and again: the change is in the framing, not a creature.
    "training-montage": {
        "setup": ("medium", "eye", "absent"),
        "hidden_problem": ("medium", "high", "absent"),
        "diagnosis": ("close", "eye", "absent"),
        "climax": ("full", "low", "absent"),
        "aftermath": ("close", "low", "absent"),
        "win": ("long", "low", "absent"),
    },
}

#: Per-shape action lines, `{name}` is the opponent. A role not listed keeps the default.
TYPE_ACTION: dict[str, dict[str, str]] = {
    "mystery": {
        "setup": "the heroes stare at something that makes no sense",
        "hidden_problem": "a clue that does not fit, lit from the side",
        "diagnosis": "the heroes chase a lead and it goes cold",
        "climax": "{name} revealed at last, the heroes see what was behind it all",
    },
    "pivot": {
        "setup": "the hero lays out the plan with confidence",
        "hidden_problem": "the plan bends under its first real test",
        "diagnosis": "the hero looks at the plan and sees that it will not hold",
        "climax": "the old plan swept aside, a new one drawn in its place",
        "aftermath": "the discarded work stacked up behind the heroes",
        "win": "the heroes walk a new road, uncertain but moving",
    },
    "rivalry": {
        "setup": "the heroes face {name} across the room, neither blinking",
        "hidden_problem": "{name} makes the first move and gains ground",
        "diagnosis": "the heroes spot the tell in {name}'s guard",
        "climax": "the heroes and {name} collide head to head",
    },
    "training-montage": {
        "setup": "the hero marks the starting line and the target",
        "hidden_problem": "the first attempt falls short of the mark",
        "diagnosis": "the hero changes grip and tries a different way",
        "climax": "the best attempt yet clears the mark",
        "aftermath": "tired hands, worn tools, a long day behind them",
        "win": "the hero looks back along the road they climbed",
    },
}

#: Per-shape ROLE_ACTION overrides (motion, impact, reaction, camera) where a creature would not fit.
TYPE_MOTION: dict[str, dict[str, tuple[str, str, str, str]]] = {
    shape: {
        "setup": ("still air, a screen glowing", "", "the pair size up the problem", "slow push in"),
        "hidden_problem": ("a small thing slips out of place", "a thin crack spreads", "a frown", "slow push in"),
        "diagnosis": ("a hand stops mid-motion", "the picture shifts", "eyes widen", "slow zoom to the detail"),
        "climax": ("the whole frame turns on one gesture", "the old way falls away", "a long breath out", "wide hold"),
        "aftermath": ("papers settling", "", "a slow nod", "slow pull back"),
        "win": ("the pair move on together", "", "a small, tired smile", "crane up and away"),
    }
    for shape in NO_VILLAIN_TYPES
}

ARC_DIRECTION: dict[str, str] = {
    "mystery": (
        " This is a MYSTERY: keep the cause out of frame (villain_presence absent) until the "
        "climax, which is the reveal; show clues, shadows and close details, not the culprit."
    ),
    "pivot": (
        " This is a PIVOT: there is NO villain and no creature. The opponent is a plan that "
        "stops working. Leave the villain name and look empty and use villain_presence absent "
        "everywhere. The climax is a decision, not a fight."
    ),
    "rivalry": (
        " This is a RIVALRY: the opponent is on the page from the first panel (lurking or "
        "attacking), and two sides stay in frame together."
    ),
    "training-montage": (
        " This is a TRAINING MONTAGE: there is NO villain and no creature. Show the same hero "
        "and place getting better, with each panel a visibly different attempt. Leave the "
        "villain name and look empty and use villain_presence absent everywhere."
    ),
}


def _role_defaults(arc_type: str, role: str) -> tuple[str, str, str]:
    base = ROLE_DEFAULTS.get(role, ("medium", "eye", "absent"))
    return TYPE_ROLE_DEFAULTS.get(arc_type, {}).get(role, base)


def frame_villain(frame: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """The chapter's villain, named from the deck archetype.

    t_ae4a9485: the director used to invent a new villain name every render
    ("the even swap", "the swapping", "the twin-handed thief", "The Ledger Hydra"), so the
    reference sheet, the prompts and the art changed every time and the lettering had to
    chase a moving target. The name now comes from the deck, so the same chapter draws the
    same villain and the existing sheet is reused instead of paid for again.
    """
    if not frame:
        return None
    villain = frame.get("villain")
    if not isinstance(villain, Mapping):
        return None
    key = str(villain.get("key") or "").strip()
    look = str(villain.get("look") or villain.get("label") or "").strip()
    if not key or not look:
        return None
    return key.replace("-", " ").title(), look


def _frame_brief(frame: Mapping[str, Any] | None, *, villainless: bool = False) -> str:
    """A one-line deck brief for the director (empty when there is no deck).

    A shape with no creature in it drops the deck's villain archetype from the brief.
    """
    if not frame:
        return ""
    from canon.ideas import frame_prompt

    brief = frame_prompt(frame)
    if villainless:
        brief = "; ".join(bit for bit in brief.split("; ") if not bit.startswith("villain archetype:"))
    return brief


#: Used only when neither the style nor the deck names a world. Deliberately plain: a style
#: owns its world (StyleCard.world), the director does not impose a sci-fi one.
WORLD_DEFAULT = "the place where the work happens, a desk and a screen in an ordinary lit room"

_TEXT_REQUEST = re.compile(r"(\"[^\"]*\"|'[^']*'|\b(sign|banner|label|screen|text|words?|caption)s? (that )?(says?|reads?|showing)\b[^,.;]*)", re.I)
_CLOSEUP_CUE = re.compile(r"\b(?:close[- ]?up|zoom(?:ed|ing)?\s+in|up\s+close|peer(?:s|ing)?\s+closely)\b", re.I)

_SCREEN_WORDS = r"(?:screens?|monitors?|displays?|consoles?|terminals?|holograms?)"
_SCREEN_ADJ = r"(?:diagnostic|terminal|code|text|log|status|readout|glowing|flickering|holographic|green|red|amber|cyan|main|central|wall|overhead)"
_NON_CLAUSE_WORD = r"(?!and\b|or\b|but\b|as\b|while\b|when\b|where\b|that\b|which\b|who\b|by\b|at\b|on\b|in\b|to\b|from\b|into\b|onto\b|over\b|under\b|near\b|around\b|across\b|through\b|before\b|after\b|between\b)[a-z-]+"
_SCREEN_WITH_TEXT_RE = re.compile(
    rf"\b(?:{_SCREEN_ADJ}\s+)*(?P<dev>{_SCREEN_WORDS})"
    r"\s+(?:that\s+|which\s+)?(?:(?:are|is)\s+)?"
    r"(?:full\s+of|filled\s+with|covered\s+(?:in|with)|packed\s+with|crowded\s+with|flashing|streaming|scrolling|showing|displaying)\s+"
    rf"(?:{_NON_CLAUSE_WORD}\s+){{0,4}}?"
    r"(?:source\s+code|data\s+lines?|lines?\s+of\s+(?:code|text|logs?|output)|code|text|words?|letters?|numbers?|digits?|logs?|readouts?|messages?|errors?|warnings?|characters?|symbols?|output)\b",
    re.I,
)
_CODE_LINES_RE = re.compile(
    r"\b(?:(?P<art>an?|the)\s+)?(?:(?:specific|single|glowing|flashing|red|green|amber|cyan|highlighted|broken|failing|scrolling|streaming|corrupted|faulty)\s+)*"
    r"(?P<unit>lines?|blocks?|snippets?|rows?|columns?|streams?|walls?)\s+of\s+"
    r"(?:(?:source|terminal|console|diagnostic|error|status|log|green|red|glowing)\s+)*"
    r"(?:data\s+text|code|text|logs?|output|errors?|warnings?|numbers?|digits?|characters?)\b",
    re.I,
)
_TECH_TEXT_RE = re.compile(
    r"\b(?:(?P<art>an?|the)\s+)?(?:(?:specific|single|glowing|flashing|red|green|amber|cyan|highlighted|broken|failing|scrolling|streaming|corrupted|faulty)\s+)*"
    r"(?P<phrase>(?:error|warning|status|log|console|terminal|diagnostic|text|debug|system)\s+"
    r"(?:messages?|readouts?|outputs?|logs?|entr(?:y|ies)|text|strings?|alerts?)|"
    r"stack\s+traces?|tracebacks?|source\s+code|code\s+(?:snippets?|listings?|editors?|blocks?|lines?))\b",
    re.I,
)
_TEXTY_SCREEN_ADJ_RE = re.compile(
    rf"\b(?:diagnostic|terminal|code|text|log|status|readout)\s+(?P<dev>{_SCREEN_WORDS})\b",
    re.I,
)
_POSSESSIVE_PREFIX_RE = re.compile(r"(?:\b(?:my|his|her|its|our|their|this|that|another|each|every|any|no|some)|['’]s)\s+$", re.I)

DIRECTOR_PROMPT = (
    "You are the director of a short comic based on true events. Do NOT change any "
    "words: captions and dialogue are fixed. Decide only how each panel LOOKS. The pictures "
    "may dramatise the work as a cinematic adventure: a villain that personifies the "
    "problem, a dramatic world, action, chases and fights. Rules: one villain for the "
    "episode, a creature or force, never a real person; the climax is the most dynamic "
    "panel (low or Dutch angle, full or long shot, impact); vary shot size and angle every "
    "panel; the setup is an establishing extreme-long or long shot; include at least one "
    "close or extreme-close shot on strips of 3 or more panels (for diagnosis, discovery, "
    "or reaction); never ask for text, signs, labels, writing, lines of code, terminal logs, "
    "error messages, or readable readouts in the image; depict monitors or holographic "
    "displays only as blank glow, abstract light, colour blocks, or geometric pulses. "
    "EVERY panel is a frozen moment of ACTION, not a pose: give action as what a body is "
    "doing right now (a verb, in motion: 'hurls the blade through the gate'), motion as the "
    "direction and energy ('left to right, cables whipping'), impact as what takes the hit "
    "or breaks, reaction as how the other side answers in the same frame, and camera as the "
    "movement the shot implies ('push in', 'whip pan', 'crane down'). Two neighbouring "
    "panels never share location, shot and angle together. The climax is a full-bleed "
    "impact frame with speed lines and debris. shot is one of "
    + ", ".join(SHOTS) + "; angle is one of " + ", ".join(ANGLES) + "; villain_presence is "
    "one of absent, lurking, attacking, defeated. Follow visual_tone when it is given. Also add wordless FILLER panels like a "
    "real comic book, about one for every two story panels: establishing shots of the "
    "world, reaction close-ups, the villain lurking, the calm before the fight, the impact "
    "after it, a quiet closing beat. A filler goes after a story panel (never after the "
    "last one), has no words, and may show only the episode's cast or the villain. "
    "Answer JSON only: "
    '{"villain": {"name": "...", "look": "..."}, "world": "...", "panels": '
    '[{"id": "...", "location": "...", "action": "...", "villain_presence": "...", '
    '"shot": "...", "angle": "..."}], "fillers": [{"after": "<story panel id>", '
    '"kind": "...", "location": "...", "action": "...", "cast": ["..."], '
    '"villain_presence": "...", "shot": "...", "angle": "..."}]}'
)


@dataclass
class PanelTreatment:
    panel_id: str
    role: str
    location: str
    action: str
    villain_presence: str
    shot: str
    angle: str
    #: M3 action grammar. A panel is a moment of action, not a pose: what moves, what it
    #: hits, how the other side reacts, and how the camera behaves. Empty means the panel
    #: keeps the older static treatment.
    motion: str = ""
    impact: str = ""
    reaction: str = ""
    camera: str = ""
    #: Filler panels only (V5): the story panel this one follows, and who is drawn.
    after: str = ""
    cast: tuple[str, ...] = ()
    #: V16: the episode's visual tone, repeated on every panel.
    tone: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = {
            "panel_id": self.panel_id,
            "role": self.role,
            "location": self.location,
            "action": self.action,
            "villain_presence": self.villain_presence,
            "shot": self.shot,
            "angle": self.angle,
        }
        for name in ("motion", "impact", "reaction", "camera"):
            value = getattr(self, name)
            if value:
                data[name] = value
        if self.role == "filler":
            data.update(after=self.after, cast=list(self.cast))
        return data


@dataclass
class Treatment:
    villain_name: str
    villain_look: str
    world: str
    panels: dict[str, PanelTreatment] = field(default_factory=dict)
    source: str = "rules"  # rules | model
    repairs: list[str] = field(default_factory=list)
    #: V5: wordless filler panels (establishing, reaction, lurking, impact, quiet beat).
    fillers: list[PanelTreatment] = field(default_factory=list)
    #: The shape of the story this treatment was made for ("" = the default shape).
    arc_type: str = ""

    @property
    def has_villain(self) -> bool:
        return bool(self.villain_name)

    def as_dict(self) -> dict[str, Any]:
        extra = {"arc_type": self.arc_type} if self.arc_type else {}
        return {
            **extra,
            "villain": {"name": self.villain_name, "look": self.villain_look},
            "world": self.world,
            "source": self.source,
            "repairs": self.repairs,
            "panels": [p.as_dict() for p in self.panels.values()],
            "fillers": [f.as_dict() for f in self.fillers],
        }

    def sequence(self, ir: Mapping[str, Any]) -> list[tuple[str, PanelTreatment | None]]:
        """Reading order: every story panel id, each followed by its fillers."""
        order: list[tuple[str, PanelTreatment | None]] = []
        for panel in ir.get("panels", []) or []:
            pid = str(panel.get("id"))
            order.append((pid, self.panels.get(pid)))
            order += [(f.panel_id, f) for f in self.fillers if f.after == pid]
        return order


#: V4: a chapter reaches at least this many panels with its fillers.
MIN_CHAPTER_PANELS = 16
FILLER_KINDS = ("establishing", "reaction", "lurk", "calm", "impact", "quiet")


def _episode_cast(ir: Mapping[str, Any]) -> list[str]:
    seen: list[str] = []
    for panel in ir.get("panels", []) or []:
        for name in panel.get("cast", []) or []:
            if name not in seen:
                seen.append(str(name))
    return seen


def _rule_fillers(treatment: "Treatment", ir: Mapping[str, Any]) -> list[PanelTreatment]:
    """About one wordless panel for every two story panels (V5), chosen by what comes next."""
    panels = list(ir.get("panels", []) or [])
    cast = _episode_cast(ir)
    binding = ir.get("cast_binding")
    chars = binding.get("characters") if isinstance(binding, Mapping) else None
    lead_candidates = [
        name
        for name, entry in (chars.items() if isinstance(chars, Mapping) else ())
        if isinstance(entry, Mapping) and entry.get("lead") is True and name in cast
    ]
    hero = lead_candidates[0] if lead_candidates else ("Sahil" if "Sahil" in cast else (cast[0] if cast else ""))
    name = treatment.villain_name
    fillers: list[PanelTreatment] = []
    toggle = 0
    # Balanced (V5): after every second story panel. A chapter must still reach 16 panels
    # (V4), so a thin chapter gets a filler after more story panels - never more than one each.
    every_other = [i for i in range(len(panels) - 1) if i % 2 == 1]
    slots = set(every_other)
    if len(panels) >= 7:
        for i in range(len(panels) - 1):
            if len(panels) + len(slots) >= MIN_CHAPTER_PANELS:
                break
            slots.add(i)
    for index, panel in enumerate(panels[:-1]):
        if index not in slots:
            continue
        pid = str(panel.get("id"))
        role = str(panel.get("role", ""))
        upcoming = str(panels[index + 1].get("role", ""))
        if upcoming == "climax":
            kind, who, shot, angle, presence = "calm", (hero,), "extreme-close", "eye", "absent"
            action = f"{hero}'s eyes narrow in the half-light, a breath before the storm"
        elif role == "climax" and not treatment.has_villain:
            kind, who, shot, angle, presence = "impact", (), "extreme-long", "overhead", "absent"
            action = "the room goes still after the decision; nothing moves for a moment"
        elif role == "climax":
            kind, who, shot, angle, presence = "impact", (), "extreme-long", "overhead", "defeated"
            action = f"the shockwave of the clash rolls out; fragments of {name} scatter across the ground"
        elif upcoming in {"hidden_problem", "diagnosis", "insert"} and index < len(panels) * 0.7 and not treatment.has_villain:
            kind, who, shot, angle, presence = "establishing", (), "long", "dutch", "absent"
            action = "the workspace in silence, one thing slightly out of place"
        elif upcoming in {"hidden_problem", "diagnosis", "insert"} and index < len(panels) * 0.7:
            kind, who, shot, angle, presence = "lurk", (), "long", "dutch", "lurking"
            action = f"{name} stir in the shadows, unseen, closer than before"
        elif toggle % 2 == 0:
            kind, who, shot, angle, presence = "establishing", (), "extreme-long", "high", "absent"
            action = "the world in silence: vast, glowing, still"
            toggle += 1
        else:
            kind, who, shot, angle, presence = "reaction", (hero,) if hero else (), "close", "low", "absent"
            action = f"{hero} pauses, taking in what just happened" if hero else "a quiet moment"
            toggle += 1
        if role in {"aftermath", "win"} and kind in {"lurk", "calm"}:
            kind, who, shot, angle, presence = "quiet", (), "long", "eye", "absent"
            action = "a quiet beat: the dust settles and the light returns"
        fillers.append(PanelTreatment(
            panel_id=f"{pid}-f{len(fillers) + 1}", role="filler", location=treatment.world,
            action=action, villain_presence=presence, shot=shot, angle=angle, after=pid,
            cast=tuple(n for n in who if n),
        ))
    return fillers


def _model_fillers(raw: Any, treatment: "Treatment", ir: Mapping[str, Any]) -> list[PanelTreatment] | None:
    """The director's own fillers, if they obey the rules; ``None`` means use the rules."""
    if not isinstance(raw, list):
        return None
    ids = [str(p.get("id")) for p in ir.get("panels", []) or []]
    cast = set(_episode_cast(ir))
    target = len(_rule_fillers(treatment, ir))
    fillers: list[PanelTreatment] = []
    for item in raw:
        if not isinstance(item, dict) or str(item.get("after")) not in ids[:-1]:
            continue
        after = str(item.get("after"))
        who = tuple(str(n) for n in item.get("cast", []) or [] if str(n) in cast)[:2]
        fillers.append(PanelTreatment(
            panel_id=f"{after}-f{len(fillers) + 1}", role="filler",
            location=str(item.get("location") or treatment.world),
            action=str(item.get("action") or ""), villain_presence=str(item.get("villain_presence") or "absent"),
            shot=str(item.get("shot") or ""), angle=str(item.get("angle") or ""), after=after, cast=who,
        ))
    if not target or abs(len(fillers) - target) > 1:
        return None
    return fillers


def pick_villain(text: str) -> tuple[str, str]:
    for kind, pattern in _PATTERNS:
        if pattern.search(text or ""):
            return VILLAINS[kind]
    return VILLAINS["default"]


def _abstract_screen(match: re.Match[str]) -> str:
    prefix = "Luminous" if match.start() == 0 else "luminous"
    return f"{prefix} abstract {match.group('dev').lower()}"


def _pulse_replacement(
    text: str,
    match: re.Match[str],
    *,
    plural: bool,
    abstract_only: bool = False,
) -> str:
    art = (match.groupdict().get("art") or "").lower()
    if abstract_only:
        base = "the abstract glowing light" if art == "the" else "abstract glowing light"
    elif plural:
        base = "the bright signal pulses" if art == "the" else "bright signal pulses"
    elif art == "the":
        base = "the bright signal pulse"
    elif not art and _POSSESSIVE_PREFIX_RE.search(text[:match.start()]):
        base = "bright signal pulse"
    else:
        base = "a bright signal pulse"
    return base[0].upper() + base[1:] if match.start() == 0 else base


def no_text(value: str) -> str:
    """Strip anything that would ask the image model to draw words or readable code."""
    cleaned = _TEXT_REQUEST.sub("", value or "")
    cleaned = " ".join(cleaned.split())
    # Quoted text removal can leave an orphan reporting verb right before punctuation
    # (e.g. "displays show 'GREEN'." -> "displays show ."). Drop the orphan verb and space.
    cleaned = re.sub(
        r"\b(?:that\s+|which\s+)?(?:say(?:s|ing)?|read(?:s|ing)?|show(?:s|ing)?|display(?:s|ing)?)\s*(?=[.,;:!?]|$)",
        "",
        cleaned,
        flags=re.I,
    )
    cleaned = _SCREEN_WITH_TEXT_RE.sub(_abstract_screen, cleaned)
    cleaned = _CODE_LINES_RE.sub(
        lambda m: _pulse_replacement(cleaned, m, plural=m.group("unit").lower().endswith("s")),
        cleaned,
    )
    cleaned = _TECH_TEXT_RE.sub(
        lambda m: _pulse_replacement(
            cleaned,
            m,
            plural=m.group("phrase").lower().endswith("s"),
            abstract_only=m.group("phrase").lower() == "source code",
        ),
        cleaned,
    )
    cleaned = _TEXTY_SCREEN_ADJ_RE.sub(_abstract_screen, cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return " ".join(cleaned.split()).strip(" ,;:")


def parse_shot_plan(spec: str) -> dict[str, str] | list[str]:
    """Parse ``--shots`` specs into a panel/role map or positional shot list."""
    raw = (spec or "").strip()
    if not raw:
        return {}
    tokens = [part.strip() for part in raw.split(",")]
    non_empty = [tok for tok in tokens if tok]
    if not non_empty:
        return {}
    if any("=" in tok for tok in non_empty):
        mapping: dict[str, str] = {}
        for tok in non_empty:
            if "=" not in tok:
                raise ValueError(f"mixed key=shot and bare shot token {tok!r}")
            key, raw_shot = (piece.strip() for piece in tok.split("=", 1))
            shot = raw_shot.lower()
            if not key or shot not in SHOTS:
                raise ValueError(f"invalid shot constraint {tok!r}; expected key=<{'|'.join(SHOTS)}>")
            mapping[key] = shot
        return mapping
    shots: list[str] = []
    for tok in tokens:
        shot = tok.lower()
        if shot and shot not in SHOTS:
            raise ValueError(f"unknown shot {tok!r}; choose from {', '.join(SHOTS)}")
        shots.append(shot)
    return shots


def resolve_shot_plan(
    shot_plan: str | Sequence[str] | Mapping[str, str] | None,
    panels: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Resolve an explicit shot plan into ``{panel_id: shot}``."""
    if not shot_plan:
        return {}
    if isinstance(shot_plan, str):
        parsed = parse_shot_plan(shot_plan)
        if not parsed:
            return {}
        shot_plan = parsed

    panel_list = list(panels)
    panel_ids = [str(p.get("id")) for p in panel_list]
    if isinstance(shot_plan, Mapping):
        resolved: dict[str, str] = {}
        for raw_key, raw_shot in shot_plan.items():
            shot = str(raw_shot or "").strip().lower()
            if not shot:
                continue
            if shot not in SHOTS:
                raise ValueError(f"unknown shot {raw_shot!r}; choose from {', '.join(SHOTS)}")
            key = str(raw_key).strip()
            matched = [pid for pid in panel_ids if pid == key]
            if not matched:
                matched = [pid for pid in panel_ids if pid.endswith(f"-{key}")]
            if not matched and key.isdigit():
                idx = int(key) - 1
                if 0 <= idx < len(panel_ids):
                    matched = [panel_ids[idx]]
            if not matched:
                matched = [str(p.get("id")) for p in panel_list if str(p.get("role", "")) == key]
            if not matched:
                available = ", ".join(panel_ids) or "none"
                raise ValueError(f"shot_plan key {raw_key!r} matched no panel (available: {available})")
            for pid in matched:
                resolved[pid] = shot
        return resolved

    if isinstance(shot_plan, Sequence) and not isinstance(shot_plan, (bytes, bytearray)):
        resolved = {}
        for index, raw_shot in enumerate(shot_plan):
            shot = str(raw_shot or "").strip().lower()
            if not shot:
                continue
            if shot not in SHOTS:
                raise ValueError(f"unknown shot {raw_shot!r}; choose from {', '.join(SHOTS)}")
            if index >= len(panel_ids):
                raise ValueError(
                    f"shot_plan has entry at index {index + 1}, but episode has {len(panel_ids)} panels"
                )
            resolved[panel_ids[index]] = shot
        return resolved

    raise ValueError(f"unsupported shot_plan type: {type(shot_plan).__name__}")


def _cues_closeup(panel: Mapping[str, Any]) -> bool:
    if str(panel.get("shot", "")).strip().lower() in CLOSE_SHOTS:
        return True
    chunks = [str(panel.get("caption") or "")]
    chunks.extend(str(line.get("text") or "") for line in (panel.get("lines") or []))
    return bool(_CLOSEUP_CUE.search(" ".join(chunks)))


def _ensure_closeup_coverage(
    treatment: Treatment,
    ir: Mapping[str, Any],
    pinned: Mapping[str, str],
) -> None:
    """Guarantee at least one close-up or extreme close-up on 3+ panel strips."""
    story_panels = list(ir.get("panels", []) or [])
    if len(story_panels) < 3:
        return
    if any(
        treatment.panels.get(str(p.get("id"))) is not None
        and treatment.panels[str(p.get("id"))].shot in CLOSE_SHOTS
        for p in story_panels
    ):
        return

    candidates = [
        p
        for p in story_panels
        if str(p.get("id")) not in pinned and str(p.get("role", "")) not in {"climax", "setup"}
    ]
    if not candidates:
        return

    def _priority(panel: Mapping[str, Any]) -> int:
        role = str(panel.get("role", ""))
        if role == "diagnosis":
            return 0
        if _cues_closeup(panel):
            return 1
        if role == "hidden_problem":
            return 2
        if role in {"aftermath", "insert"}:
            return 3
        return 4

    chosen = min(candidates, key=_priority)
    pid = str(chosen.get("id"))
    treatment.panels[pid].shot = "close"
    treatment.repairs.append(f"{pid}: promoted to close-up for shot diversity")


def _rules(
    ir: Mapping[str, Any], story_text: str, *, frame: Mapping[str, Any] | None = None, arc_type: str = "",
    world: str = "",
) -> Treatment:
    name, look = pick_villain(story_text)
    world = world or WORLD_DEFAULT
    villainless = arc_type in NO_VILLAIN_TYPES
    if frame:
        villain = frame.get("villain") or {}
        if villain.get("look") and not villainless:
            name, look = str(villain.get("label") or name), str(villain["look"])
        world_spec = frame.get("world") or {}
        if world_spec.get("label"):
            world = str(world_spec["label"])
    if villainless:
        name = look = ""
    treatment = Treatment(villain_name=name, villain_look=look, world=world, arc_type=arc_type)
    for panel in ir.get("panels", []) or []:
        role = str(panel.get("role", "aftermath"))
        shot, angle, presence = _role_defaults(arc_type, role)
        action = {
            "setup": "the heroes arrive and survey the scene, calm before the storm",
            "hidden_problem": f"{name} creep in unnoticed in the shadows behind the heroes",
            "diagnosis": f"the heroes spot the first trace of {name}",
            "climax": f"the heroes clash with {name} in a burst of light and motion lines",
            "aftermath": f"the dust settles; scattered remains of {name} fade away",
            "win": "the heroes stand victorious, the world calm again",
        }.get(role, "the heroes press on")
        action = TYPE_ACTION.get(arc_type, {}).get(role, action).format(name=name)
        table = {**ROLE_ACTION, **TYPE_MOTION.get(arc_type, {})}
        motion, impact, reaction, camera = table.get(
            role, ("the scene moves with the heroes", "", "the other side answers", "steady push in")
        )
        if frame and role == "setup" and not villainless:
            shot_spec = frame.get("shot") or {}
            if shot_spec.get("label"):
                action = f"{action}; opening on {shot_spec['label']}"
        treatment.panels[str(panel.get("id"))] = PanelTreatment(
            panel_id=str(panel.get("id")), role=role, location=world,
            action=action, villain_presence=presence, shot=shot, angle=angle,
            motion=motion, impact=impact, reaction=reaction, camera=camera,
        )
    treatment.fillers = _rule_fillers(treatment, ir)
    return treatment


def _repair(
    treatment: Treatment,
    ir: Mapping[str, Any],
    *,
    shot_plan: Mapping[str, str] | None = None,
) -> Treatment:
    """Enforce the craft rules on whatever the model returned."""
    arc_type = treatment.arc_type
    villainless = arc_type in NO_VILLAIN_TYPES
    pinned = dict(shot_plan or {})
    for panel in ir.get("panels", []) or []:
        pid = str(panel.get("id"))
        role = str(panel.get("role", "aftermath"))
        item = treatment.panels.get(pid)
        default_shot, default_angle, default_presence = _role_defaults(arc_type, role)
        if item is None:
            item = PanelTreatment(pid, role, treatment.world, "the heroes press on", default_presence, default_shot, default_angle)
            treatment.panels[pid] = item
            treatment.repairs.append(f"{pid}: missing, rule default used")
        if item.shot not in SHOTS:
            treatment.repairs.append(f"{pid}: unknown shot {item.shot!r}")
            item.shot = default_shot
        if item.angle not in ANGLES:
            treatment.repairs.append(f"{pid}: unknown angle {item.angle!r}")
            item.angle = default_angle
        if item.villain_presence not in {"absent", "lurking", "attacking", "defeated"}:
            item.villain_presence = default_presence
        if villainless:
            item.villain_presence = "absent"
        elif arc_type == "mystery" and role != "climax" and item.villain_presence == "attacking" and role in {"setup", "hidden_problem", "diagnosis"}:
            treatment.repairs.append(f"{pid}: culprit kept out of frame until the reveal")
            item.villain_presence = "absent"
        if pid in pinned:
            if item.shot != pinned[pid]:
                treatment.repairs.append(f"{pid}: shot pinned to {pinned[pid]} by shot_plan")
                item.shot = pinned[pid]
            if role == "climax" and item.angle not in {"low", "dutch"}:
                treatment.repairs.append(f"{pid}: climax made dynamic")
                item.angle = "low"
        elif role == "climax" and (item.angle not in {"low", "dutch"} or item.shot not in {"full", "long"}):
            treatment.repairs.append(f"{pid}: climax made dynamic")
            item.shot, item.angle = "full", "low" if item.angle not in {"low", "dutch"} else item.angle
        item.location = no_text(item.location) or treatment.world
        item.action = no_text(item.action) or "the heroes press on"
        for field_name in ("motion", "impact", "reaction", "camera"):
            value = no_text(getattr(item, field_name) or "").strip()
            setattr(item, field_name, trim_at_boundary(value, 160))
    _ensure_closeup_coverage(treatment, ir, pinned)
    previous: tuple[str, str] | None = None
    for panel in ir.get("panels", []) or []:
        pid = str(panel.get("id"))
        item = treatment.panels[pid]
        if previous == (item.shot, item.angle):
            treatment.repairs.append(f"{pid}: repeated shot varied")
            item.angle = next(a for a in ANGLES if a != item.angle and a != "overhead")
        previous = (item.shot, item.angle)
    treatment.villain_look = no_text(treatment.villain_look)
    treatment.world = no_text(treatment.world) or WORLD_DEFAULT  # callers pass a style world through `direct`
    for filler in treatment.fillers:
        filler.shot = filler.shot if filler.shot in SHOTS else "extreme-long"
        filler.angle = filler.angle if filler.angle in ANGLES else "high"
        if filler.villain_presence not in {"absent", "lurking", "attacking", "defeated"} or villainless:
            filler.villain_presence = "absent"
        filler.location = no_text(filler.location) or treatment.world
        filler.action = no_text(filler.action) or "a quiet beat"
        for field_name in ("motion", "impact", "reaction", "camera"):
            value = no_text(getattr(filler, field_name) or "").strip()
            setattr(filler, field_name, trim_at_boundary(value, 160))
    # No two consecutive panels - story or filler - share both shot and angle.
    previous = None
    for pid, item in treatment.sequence(ir):
        if item is None:
            continue
        if previous == (item.shot, item.angle):
            treatment.repairs.append(f"{pid}: repeated shot varied")
            item.angle = next(a for a in ANGLES if a != item.angle and a != "overhead")
        previous = (item.shot, item.angle)
    return treatment


def direct(
    ir: Mapping[str, Any],
    *,
    story_text: str,
    complete_json: Callable[..., Mapping[str, Any]] | None = None,
    tone: str = "",
    frame: Mapping[str, Any] | None = None,
    arc_type: str = "",
    world: str = "",
    shot_plan: str | Sequence[str] | Mapping[str, str] | None = None,
) -> Treatment:
    """A visual treatment for the episode: model when available, rules otherwise.

    ``world`` is the style's own setting (StyleCard.world); it replaces the plain default
    when the idea deck does not name one.

    ``arc_type`` is the shape of the story. The default shape (or "") keeps the original
    villain treatment; pivot and training-montage have no villain at all.

    ``frame`` is the idea-deck draw for this chapter (world, villain archetype, genre,
    motif, opening shot). It fixes the look a chapter cannot repeat, while the facts of
    the episode still decide what happens.

    ``shot_plan`` pins specific panel or role shots cleanly after directing.
    """
    arc_type = "" if arc_type == DEFAULT_TYPE else arc_type
    villainless = arc_type in NO_VILLAIN_TYPES
    effective_plan = shot_plan if shot_plan is not None else ir.get("shot_plan")
    resolved_shots = resolve_shot_plan(effective_plan, ir.get("panels", []) or [])
    frame_brief = _frame_brief(frame, villainless=villainless)
    if complete_json is None:
        return _with_tone(
            _repair(_rules(ir, story_text, frame=frame, arc_type=arc_type, world=world), ir, shot_plan=resolved_shots),
            tone,
        )
    panels = [
        {"id": p.get("id"), "role": p.get("role"), "caption": p.get("caption", ""),
         "lines": [f"{l.get('speaker')}: {l.get('text')}" for l in p.get("lines", [])],
         "cast": list(p.get("cast", []))}
        for p in ir.get("panels", []) or []
    ]
    prompt = DIRECTOR_PROMPT + ARC_DIRECTION.get(arc_type, "") + (
        f" The art style's own world is: {world}. Stay inside it." if world else ""
    )
    payload: dict[str, Any] = {
        "title": ir.get("logline", ""),
        "story": story_text[:3000],
        "panels": panels,
        "visual_tone": tone,
        "frame": frame_brief,
        **({"story_shape": arc_type} if arc_type else {}),
    }
    if resolved_shots:
        prompt += " Follow shot_plan exactly for every panel id listed there."
        payload["shot_plan"] = resolved_shots
    try:
        answer = complete_json(
            prompt=prompt,
            payload=payload,
            schema_hint={
                "villain": {"name": "str", "look": "str"},
                "world": "str",
                "panels": ["object"],
                "fillers": ["object"],
            },
            stage="episode.director",
        )
    except Exception:  # noqa: BLE001 - a missing director never blocks a render
        fallback = _with_tone(
            _repair(_rules(ir, story_text, frame=frame, arc_type=arc_type, world=world), ir, shot_plan=resolved_shots),
            tone,
        )
        fallback.repairs.append("director model unavailable; rules used")
        return fallback
    raw_villain = answer.get("villain")
    villain: Mapping[str, Any] = raw_villain if isinstance(raw_villain, dict) else {}
    default_name, default_look = pick_villain(story_text)
    treatment = Treatment(
        villain_name="" if villainless else trim_at_boundary(str(villain.get("name") or default_name), 40),
        villain_look="" if villainless else trim_at_boundary(str(villain.get("look") or default_look), 300),
        world=trim_at_boundary(str(answer.get("world") or world or WORLD_DEFAULT), 300),
        source="model",
        arc_type=arc_type,
    )
    roles = {str(p.get("id")): str(p.get("role", "")) for p in ir.get("panels", []) or []}
    for item in answer.get("panels") or []:
        if not isinstance(item, dict) or str(item.get("id")) not in roles:
            continue
        pid = str(item.get("id"))
        treatment.panels[pid] = PanelTreatment(
            panel_id=pid, role=roles[pid],
            location=str(item.get("location") or ""), action=str(item.get("action") or ""),
            villain_presence=str(item.get("villain_presence") or "absent"),
            shot=str(item.get("shot") or ""), angle=str(item.get("angle") or ""),
        )
    fixed_villain = None if villainless else frame_villain(frame)
    if fixed_villain is not None:
        name, look = fixed_villain
        if (name, look) != (treatment.villain_name, treatment.villain_look):
            treatment.repairs.append("villain name taken from the chapter frame (deterministic)")
        treatment.villain_name, treatment.villain_look = name, look
    model_fillers = _model_fillers(answer.get("fillers"), treatment, ir)
    if model_fillers is None:
        treatment.fillers = _rule_fillers(treatment, ir)
        treatment.repairs.append("fillers: model count off target; rule fillers used")
    else:
        treatment.fillers = model_fillers
    return _with_tone(_repair(treatment, ir, shot_plan=resolved_shots), tone)


def _with_tone(treatment: Treatment, tone: str) -> Treatment:
    for item in list(treatment.panels.values()) + list(treatment.fillers):
        item.tone = tone
    return treatment


__all__: Sequence[str] = (
    "Treatment",
    "PanelTreatment",
    "ARC_DIRECTION",
    "CLOSE_SHOTS",
    "NO_VILLAIN_TYPES",
    "direct",
    "no_text",
    "parse_shot_plan",
    "pick_villain",
    "resolve_shot_plan",
    "SHOT_WORDS",
    "ANGLE_WORDS",
)
