"""The M1 pipeline: ingest -> redact -> beats -> arcs -> canon -> episode plan -> IR.

Runs outside the live plugin (two-phase rule, HANDOFF §2): the CLI reads history
read-only, runs the gate and the story engine, and writes artifacts to a data
directory. No image step in M1 - script only.

Design notes that matter for the gate:

* **Deterministic pass on everything.** Every event is gated deterministically,
  no model involved.
* **Model pass where it pays.** The name vocabulary is learned once for the whole
  window (batched, cached) and applied everywhere; span-level detail is only run
  on the story subset that actually feeds the episode. Both are recorded, so a
  reader knows which events were deep-gated.
* **Fail closed.** A model error or a residue hit quarantines the event. Nothing
  that failed the gate reaches beats.
* **Cache.** Every model call goes through ``canon.cache``, so a second run over
  the same window is byte-identical.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Iterable, Sequence

from canon.brief import CHAPTER_MAX_TURNS, MAX_TURNS, build_brief, entail_brief
from canon.story import DEFAULT_VOICES, edit_script, jobs_for, write_synopsis
from canon.selection import select_stories
from canon.threads import beats_by_thread, thread_index
from canon.arcs import ARC_ROLES, build_arc, build_arcs, build_brief_arc, classify_arc_type, turn_texts
from canon.beats import Beat, beats_from_events
from canon.cache import ModelCache
from canon.aliases import AliasBook, apply_to_ir, masks_in
from canon.assets import load_kits
from canon.canon import Canon, build_canon
from canon.caption import compose_caption, trim_caption
from canon.casting import build_cast_map, summarise as summarise_cast
from canon.facts import build_facts, facts_text
from canon.episode import (
    CHAPTER_MAX_STORY,
    EpisodeRefused,
    EpisodeIR,
    LLMDialogueWriter,
    TemplateDialogueWriter,
    plan_episode,
    usable_anchor_count,
    validate_ir,
)
from canon.events import KIND_MESSAGE_ASSISTANT, KIND_MESSAGE_USER, Event
from canon.gate import GatedEvent, recast_gated
from canon.grounding import ClaimGate
from canon.llm import CachedLLM, LLMClient, NullLLM
from canon.panel_prompt import episode_cast
from canon.redact import DeterministicRedactor, ModelEntityGate, RedactionGate
from canon.series import (
    SERIES_FILE,
    load as load_series,
    order_for_series,
    save as save_series,
    series_file_for,
    verify_series_compatibility,
)
from canon.series_cast import (
    CastConfigError,
    SeriesCastConfig,
    build_cast_binding,
    event_agent_role_key,
    event_role_key,
    is_unmapped_label,
    load_cast_config,
    opaque_role_id,
    resolve_voices,
    sanitize_profile_value,
    validate_cast_kits,
)
from canon.source import Window
from canon.threads import build_threads
from canon.titling import write_title

from adapters.hermes import leakscan
from adapters.hermes.evidence import MAX_SCRIPT_CHARS, EvidenceLeak, write_json, write_text
from adapters.hermes.history import HermesHistorySource
from adapters.hermes.llm import FallbackLLM, OpenAICompatLLMClient
from adapters.hermes.redact_source import public_rules, rules_from_snapshot
from adapters.hermes.snapshot import SnapshotStore, build_snapshot

#: People and tools that may appear by name in the comic.
DEFAULT_ALLOWLIST = (
    "Sahil",
    "Kensei",
    "KENSEI",
    "Hermes",
    "Claude",
    "Codex",
    "GitHub",
    "Discord",
    "Telegram",
    "Python",
    "SQLite",
    "Linux",
    "Turbofit",
    "Severian",
    "ComfyUI",
)

#: Tools/platforms safe to allowlist in cast-configured runs without hardcoded legacy characters.
NON_CAST_ALLOWLIST = (
    "Hermes",
    "Claude",
    "Codex",
    "GitHub",
    "Discord",
    "Telegram",
    "Python",
    "SQLite",
    "Linux",
    "ComfyUI",
)

DEFAULT_HERO = "Sahil"
MAX_DEEP_EVENTS = 200
MAX_TEXT_CHARS = 2000
MAX_TEXT_EVENTS = 20000

#: Which events are worth fetching text for, cheapest-story-first. ``tool.call``
#: is absent on purpose: the raw ``tool_calls`` JSON is not story material, and a
#: week contains tens of thousands of them.
TEXT_PRIORITY: dict[str, int] = {
    "message.user": 0,
    "kanban.task.created": 1,
    "kanban.comment": 1,
    "session.started": 2,
    "kanban.task.status": 3,
    "cron.run.finished": 3,
    "kanban.task.finished": 4,
    "kanban.run.finished": 5,
    "memory.fact": 6,
    "message.assistant": 7,
    "message.tool_result": 8,
    "message.meta": 9,
}


@dataclass
class PipelineConfig:
    data_dir: Path
    #: V12: which harnesses to read ("hermes", "claude-code", "codex", "opencode", "pi", "jsonl").
    harnesses: tuple[str, ...] = ("hermes",)
    #: Per-harness options: {"codex": {"root": "...", "agent": "Codex"}, "jsonl": {"mapping": {...}}}.
    harness_options: Mapping[str, Mapping[str, Any]] | None = None
    #: V11: character voices from the user's kits (tone only), merged over the defaults.
    voices: Mapping[str, str] | None = None
    #: V16: story tone - a name, a blend ("comical+serious", "comical:70,serious:30") or "auto".
    tone: str = "witty"
    #: V7: "meaning" judges captions and lines on meaning (plain English allowed);
    #: "strict" keeps the M1 vocabulary floor. The library default stays strict so the
    #: M1 tests and evals keep measuring the accepted gate; the plugin CLI defaults to meaning.
    wording: str = "strict"
    #: Story mode judges narrator captions on the "story" layer (they may interpret: stakes,
    #: feeling, motive; names, numbers and events stay strict). True keeps the literal
    #: caption layer instead (`hermes reel plan --strict-captions`).
    strict_captions: bool = False
    #: Offer each panel's speakers' punchiest real sentences (gated, verbatim) to the script
    #: writer as candidate lines. False (`--no-real-lines`) keeps every line model-written.
    real_lines: bool = True
    days: float = 7.0
    since: float | None = None
    until: float | None = None
    #: Smallest strip/chapter that may export (D7: a strip is 4-6 panels). Tests on the
    #: tiny synthetic home lower it; production keeps the spec.
    min_strip_panels: int = 4
    #: A strip can be capped to fewer story panels than the arc's six roles (a 4-panel strip
    #: is the social/README shape); 0 keeps every role.
    strip_panels: int = 0
    profiles: tuple[str, ...] | None = None
    #: Read every profile under ``<home>/profiles`` instead of only the one this process
    #: runs as. Off by default: profiles are separate by design.
    all_profiles: bool = False
    format: str = "strip"
    season: int = 1
    number: int = 1
    use_model: bool = False
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST
    hero: str = DEFAULT_HERO
    max_deep_events: int = MAX_DEEP_EVENTS
    max_text_chars: int = MAX_TEXT_CHARS
    max_text_events: int = MAX_TEXT_EVENTS
    max_vocabulary: int = 400
    privacy: str = "private"
    episodes: int = 1
    writer_llm: Any = None
    writer: Any = None
    max_writer_calls: int = 150
    snapshot_root: Path | None = None
    #: Where the private memory snapshot lives. Never the evidence folder (M1-F4).
    state_dir: Path | None = None
    llm: LLMClient | None = None
    #: Opt-in versioned series cast configuration (file path or parsed SeriesCastConfig).
    cast_config: SeriesCastConfig | Path | str | None = None
    #: Asset root for kit lookup (overrides cast_config.assets_dir when set).
    assets_dir: Path | None = None
    #: Override built-in kits root (defaults to repository ``kits/``).
    builtin_kits_dir: Path | None = None


@dataclass
class PipelineResult:
    events_ingested: int = 0
    events_with_text: int = 0
    gated: int = 0
    quarantined: int = 0
    beats: list[Beat] = field(default_factory=list)
    arc_count: int = 0
    ir: EpisodeIR | None = None
    canon: Canon | None = None
    gate_report: dict[str, Any] = field(default_factory=dict)
    cache_stats: dict[str, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    grounding: dict[str, Any] = field(default_factory=dict)
    threads: int = 0
    quarantine_rate: float = 0.0
    title: str = ""
    title_source: str = ""
    cast: list[dict[str, Any]] = field(default_factory=list)
    cast_diagnostics: list[str] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    llm_usage: dict[str, Any] = field(default_factory=dict)
    stories: list[dict[str, Any]] = field(default_factory=list)
    skipped_stories: list[dict[str, str]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.ir and not self.problems)


# --------------------------------------------------------------------------- #
# memory snapshots (owner decision 2: snapshot into the data dir each run)
# --------------------------------------------------------------------------- #


#: The memory snapshot never lives here. Audit M1-F4: raw MEMORY/USER text with 53
#: emails and 25 absolute paths was written into the evidence folder. Snapshots are
#: hashes plus gated text, in the plugin state dir, mode 600 - see
#: adapters/hermes/snapshot.py and adapters/hermes/evidence.py.


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def render_script_markdown(ir: EpisodeIR) -> str:
    lines = [
        f"# {ir.logline}",
        "",
        f"- episode: {ir.episode_id}",
        f"- format: {ir.format} ({len(ir.panels)} panels)",
        f"- arcs: {', '.join(ir.arc_ids)}",
        f"- dialogue: {ir.writer}",
        "",
    ]
    for number, panel in enumerate(ir.panels, start=1):
        lines.append(f"## Panel {number} - {panel.role} ({panel.shot})")
        if panel.caption:
            lines.append(f"*{panel.caption}*")
        lines.append(f"setting: {panel.setting} | cast: {', '.join(panel.cast)} | pose: {panel.pose} | expression: {panel.expression}")
        for line in panel.lines:
            lines.append(f"- {line.speaker}: \u201c{line.text}\u201d  [{', '.join(line.source_event_ids)}]")
        lines.append("")
    return "\n".join(lines)


def markdown_of(ir: EpisodeIR) -> str:
    """The rendered script, used for checks that must see exactly what is written."""
    return render_script_markdown(ir)


def evidence_by_event(
    gated: Iterable[GatedEvent],
    *,
    display: Mapping[str, str] | None = None,
    hero: str = "Sahil",
) -> dict[str, str]:
    """The gated text each event can ground a line with.

    Text plus its string fields: a line may only repeat what is in here, so a beat
    built from a tool name can only be talked about in those terms.
    """
    evidence: dict[str, str] = {}
    for event in gated:
        # The event's own identity is part of its evidence: a beat that names the
        # kind or the actor is repeating the record, not inventing.
        # R3-F4: no provenance in the evidence either - a model shown a record dump
        # writes one back. The structured sentence is the event's own fact.
        composed = compose_caption(event, display=display, hero=hero)
        parts: list[str] = [composed] if composed else []
        text = event.payload.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
        for key, value in sorted(event.payload.items()):
            if key in {"text", "gate"}:
                continue
            if isinstance(value, str) and value:
                parts.append(f"{key}: {value}")
            elif isinstance(value, list) and value and all(isinstance(item, str) for item in value):
                parts.append(f"{key}: {', '.join(value)}")
        evidence[event.id] = "\n".join(parts)
    return evidence


def _story_subset(beats: Sequence[Beat], limit: int) -> list[str]:
    """Event ids worth a span-level model pass: the beats the episode will use."""
    ranked = sorted(beats, key=lambda beat: (-beat.intensity, beat.ts, beat.id))[:limit]
    ids: list[str] = []
    for beat in ranked:
        ids.extend(beat.event_ids)
    return ids


def _text_for(source: HermesHistorySource, event: Event, limit: int) -> str | None:
    if event.content_ref is None:
        return None
    try:
        text = source.fetch_content(event.content_ref)
    except Exception:  # noqa: BLE001 - a missing text is not fatal, it is reported
        return None
    if not text:
        return None
    return text[:limit]


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #


#: Per-event evidence kept in a trace file: enough to check a caption, bounded so the
#: evidence folder does not become a copy of the history.
TRACE_EVIDENCE_CHARS = 1200


def episode_trace(
    ir: Any,
    arc: Any,
    story_beats: Sequence[Any],
    facts: Sequence[Mapping[str, Any]],
    evidence_map: Mapping[str, str],
    regate: Any = None,
    clean: Any = None,
    fact_sources: Sequence[str] = (),
    planned_ir: Any = None,
) -> dict[str, Any]:
    """Everything needed to follow a caption or line back to its gated source (QAR10-03).

    panel -> caption_source_event_ids / line source_event_ids -> the gated evidence text
    of those events, plus the arc and beats the episode was planned from and the thread
    fact list the captions were also judged against.
    """
    cited: list[str] = []
    for panel in ir.panels:
        cited.extend(panel.caption_source_event_ids)
        for line in panel.lines:
            cited.extend(line.source_event_ids)
    cited = list(dict.fromkeys(cited))
    # t_abdaa3a9 (Quan): the 1,200-char evidence below is for reading, not replay - a cut
    # changes the word overlap the citations were ranked by, so 4-6 lines did not replay.
    # The pool is archived whole, in the order the ranking used, with its digest. Text the
    # gate would still flag is re-gated and listed, so a replay knows which ids are inexact.
    from canon.episode import pool_digest

    from adapters.hermes.evidence import Gated

    pool = dict(getattr(ir, "citation_pool", {}) or {})
    regated: list[str] = []

    def exact(key: str, text: str) -> Any:
        if clean is None or clean(text):
            return Gated(text)
        regated.append(key)
        return (regate or (lambda value: value))(text)

    archived = {event_id: exact(event_id, text) for event_id, text in pool.items()}
    # A caption can also cite a thread fact's event that is not a story event; its whole
    # text is kept too, so every cited id can be read in full.
    outside = {
        event_id: exact(event_id, str(evidence_map.get(event_id, "")))
        for event_id in cited
        if event_id not in pool and str(evidence_map.get(event_id, "")).strip()
    }
    sources = [exact(f"fact[{index}]", str(text)) for index, text in enumerate(fact_sources or ())]
    return {
        "episode_id": ir.episode_id,
        "arc": arc.as_dict(),
        "beats": [beat.as_dict() for beat in story_beats],
        "panels": [
            {
                "id": panel.id,
                "role": panel.role,
                "own_event_ids": list(panel.event_ids),
                # Pre-alias text: the exact string each citation was ranked on.
                "caption": exact(f"{panel.id}.caption", panel.caption) if panel.caption else "",
                "caption_source_event_ids": list(panel.caption_source_event_ids),
                "lines": [
                    {
                        "speaker": line.speaker,
                        "text": exact(f"{panel.id}.line[{n}]", line.text),
                        "source_event_ids": list(line.source_event_ids),
                    }
                    for n, line in enumerate(panel.lines)
                ],
            }
            for panel in (planned_ir or ir).panels
        ],
        "evidence_pool": {
            "sha256": pool_digest(pool),
            "order": list(pool),
            "events": archived,
            "cited_outside_pool": outside,
            "regated_ids": regated,
            "caption_pool_ids": list(getattr(ir, "caption_pool_ids", ()) or ()),
            "fact_source_texts": sources,
            "how_to_replay": "python -m eval.trace_replay <trace.json>",
        },
        "facts": list(facts),
        "evidence": {
            # A cut can leave a fragment the evidence writer reads as a secret, so the
            # cut text is re-gated (same lesson as the story rows in metrics).
            event_id: (regate or (lambda text: text))(
                str(evidence_map.get(event_id, ""))[:TRACE_EVIDENCE_CHARS]
            )
            for event_id in cited
        },
        "judged_against": "panel evidence (these ids) + the thread facts listed here",
    }


def _thread_for_candidate(candidate: Any, by_event: Mapping[str, Any]) -> Any:
    """The thread holding most of a story candidate's events, or None."""
    counts: dict[str, int] = {}
    holders: dict[str, Any] = {}
    for event_id in getattr(candidate, "event_ids", ()) or ():
        thread = by_event.get(event_id)
        if thread is None:
            continue
        counts[thread.id] = counts.get(thread.id, 0) + 1
        holders[thread.id] = thread
    if not counts:
        return None
    best = max(sorted(counts), key=lambda thread_id: counts[thread_id])
    return holders[best]


#: The roles a short strip keeps, in story order, when it is capped below six panels.
STRIP_ORDER = ("setup", "hidden_problem", "climax", "win", "diagnosis", "aftermath")


def _trim_arc(arc: Any, panels: int) -> Any:
    """Keep the story's spine when a strip is capped below the arc's six roles.

    Panels are dropped from the least structural end of the story, never from the middle: a
    4-panel strip keeps setup, the hidden problem, the climax and the win, so it still has a
    beginning, a fight and a result.
    """
    if int(panels) <= 0:
        return arc
    keep = [role for role in STRIP_ORDER[: max(1, int(panels))] if role in arc.roles]
    if not keep or len(keep) >= len(arc.roles):
        return arc
    trimmed = dict(arc.roles)
    for role in list(trimmed):
        if role not in keep:
            trimmed.pop(role)
    return replace(arc, roles=trimmed)


_SCRIPT_IR_NUM_RE = re.compile(r"^script-ir-(\d+)\.json$")


def _existing_episode_numbers(data_dir: Path) -> set[int]:
    if not data_dir.is_dir():
        return set()
    numbers: set[int] = set()
    for path in data_dir.iterdir():
        match = _SCRIPT_IR_NUM_RE.fullmatch(path.name)
        if match and path.is_file():
            numbers.add(int(match.group(1)))
    return numbers


def _iter_episode_ir_files(data_dir: Path) -> list[Path]:
    """Return sorted ``script-ir.json`` and ``script-ir-<N>.json`` files in ``data_dir``."""
    if not data_dir.is_dir():
        return []
    return sorted(
        path
        for path in data_dir.iterdir()
        if path.is_file()
        and (path.name == "script-ir.json" or _SCRIPT_IR_NUM_RE.fullmatch(path.name))
    )


def _read_json_mapping(path: Path, data_dir: Path, label: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CastConfigError(
            f"existing {label} {path.name} in {data_dir} cannot be verified; use a fresh --data-dir"
        ) from exc
    if not isinstance(raw, Mapping):
        raise CastConfigError(
            f"existing {label} {path.name} in {data_dir} is malformed; use a fresh --data-dir"
        )
    return raw


def _verify_cast_directory_isolation(data_dir: Path, series_id: str) -> None:
    """Refuse mixing distinct series or uncasted chapters in the same ``--data-dir``."""
    if not data_dir.is_dir():
        return
    legacy_ledger = data_dir / SERIES_FILE
    if legacy_ledger.is_file():
        legacy = load_series(legacy_ledger)
        if legacy.chapters:
            raise CastConfigError(
                f"directory {data_dir} already holds uncasted series ledger {SERIES_FILE}; "
                f"use a fresh --data-dir for series {series_id!r}"
            )
    own_name = f"series-{series_id}.json"
    has_episodes = bool(_existing_episode_numbers(data_dir))
    for ledger_path in sorted(data_dir.glob("series-*.json")):
        if ledger_path.name == own_name:
            continue
        other = load_series(ledger_path)
        other_id = other.series_id or ledger_path.stem.removeprefix("series-")
        if other.chapters or other.series_id or has_episodes:
            raise CastConfigError(
                f"directory {data_dir} already belongs to series {other_id!r}; "
                f"use a fresh --data-dir for series {series_id!r} so existing episodes are preserved"
            )
    for path in _iter_episode_ir_files(data_dir):
        raw = _read_json_mapping(path, data_dir, "episode file")
        binding = raw.get("cast_binding")
        if not isinstance(binding, Mapping):
            raise CastConfigError(
                f"directory {data_dir} already holds uncasted episode {path.name}; "
                f"use a fresh --data-dir for series {series_id!r}"
            )
        found_id = str(binding.get("series_id") or "")
        if found_id != series_id:
            raise CastConfigError(
                f"directory {data_dir} already holds {path.name} for series {found_id!r}; "
                f"use a fresh --data-dir for series {series_id!r}"
            )


def _verify_uncasted_directory_isolation(data_dir: Path) -> None:
    """Refuse running without ``--cast-config`` in a directory that already holds a cast series."""
    if not data_dir.is_dir():
        return
    for ledger_path in sorted(data_dir.glob("series-*.json")):
        raw = _read_json_mapping(ledger_path, data_dir, "series ledger")
        other = load_series(ledger_path)
        if other.chapters or other.series_id or other.cast_digest or other.cast_roles or raw.get("cast_digest"):
            other_id = other.series_id or ledger_path.stem.removeprefix("series-")
            raise CastConfigError(
                f"directory {data_dir} already belongs to cast-configured series {other_id!r} "
                f"({ledger_path.name}); pass --cast-config or use a fresh --data-dir so existing episodes are preserved"
            )
    for path in _iter_episode_ir_files(data_dir):
        raw = _read_json_mapping(path, data_dir, "episode file")
        binding = raw.get("cast_binding")
        if binding is not None:
            found_id = str(binding.get("series_id") or "") if isinstance(binding, Mapping) else ""
            detail = f" for series {found_id!r}" if found_id else ""
            raise CastConfigError(
                f"directory {data_dir} already holds cast-configured episode {path.name}{detail}; "
                "pass --cast-config or use a fresh --data-dir so existing episodes are preserved"
            )


def validate_cast_setup(
    cast_config: SeriesCastConfig | Path | str | None,
    *,
    data_dir: Path,
    assets_dir: Path | None = None,
    builtin_kits_dir: Path | None = None,
    redactor: DeterministicRedactor | None = None,
    privacy: str = "private",
) -> tuple[SeriesCastConfig | None, dict[str, Any]]:
    """Validate opt-in cast config, kits, display names, and series ledger before any model call."""
    if cast_config is None:
        _verify_uncasted_directory_isolation(data_dir)
        return None, {}
    if redactor is None:
        rule_set = list(rules_from_snapshot())
        if privacy == "public":
            rule_set.extend(public_rules())
        redactor = DeterministicRedactor(rule_set)
    cast_cfg = (
        cast_config
        if isinstance(cast_config, SeriesCastConfig)
        else load_cast_config(cast_config)
    )
    resolved_assets = assets_dir or cast_cfg.assets_dir or data_dir
    builtin_kits = builtin_kits_dir or (Path(__file__).resolve().parents[2] / "kits")
    user_kits = resolved_assets / "kits"
    kits = load_kits(builtin_kits, user_kits)
    validate_cast_kits(cast_cfg, kits, user_kits_dir=user_kits)
    for name in cast_cfg.all_display_names():
        if redactor.findings(name):
            raise CastConfigError(f"configured character name {name!r} triggers redaction rules")
    _verify_cast_directory_isolation(data_dir, cast_cfg.series_id)
    series_path = series_file_for(data_dir, cast_cfg.series_id)
    verify_series_compatibility(
        load_series(series_path),
        series_id=cast_cfg.series_id,
        cast_digest=cast_cfg.cast_digest,
        cast_roles=cast_cfg.ledger_roles(),
    )
    return cast_cfg, kits


def _resolve_cast_setup(
    config: PipelineConfig,
    redactor: DeterministicRedactor,
) -> tuple[SeriesCastConfig | None, dict[str, Any]]:
    return validate_cast_setup(
        config.cast_config,
        data_dir=config.data_dir,
        assets_dir=config.assets_dir,
        builtin_kits_dir=config.builtin_kits_dir,
        redactor=redactor,
        privacy=getattr(config, "privacy", "private"),
    )


def _sanitize_gated_event(
    sealed: GatedEvent,
    actor_role: str,
    agent_role: str,
    cast_cfg: SeriesCastConfig,
) -> GatedEvent:
    """Replace raw profile/actor metadata on a gated event with safe display names or opaque labels."""
    if agent_role:
        agent_member = cast_cfg.member_for_role(agent_role)
        profile_label = agent_member.name if agent_member is not None else opaque_role_id(agent_role)
    else:
        profile_label = ""

    if actor_role == "owner":
        actor_label = cast_cfg.owner.name
    elif actor_role:
        actor_member = cast_cfg.member_for_role(actor_role)
        actor_label = actor_member.name if actor_member is not None else opaque_role_id(actor_role)
    else:
        actor_label = sealed.actor

    payload = dict(sealed.payload)
    for field_name in ("assignee", "profile", "created_by", "author"):
        raw_val = payload.get(field_name)
        if isinstance(raw_val, str) and raw_val.strip():
            payload[field_name] = sanitize_profile_value(raw_val, cast_cfg)

    ref = dict(sealed.ref)
    store = ref.get("store", "")
    if isinstance(store, str) and store.startswith("profile:"):
        ref["store"] = f"profile:{profile_label or opaque_role_id(agent_role or store)}"

    subject = sealed.subject
    if sealed.kind == "kanban.run.started" and profile_label:
        subject = profile_label
    elif sealed.kind == "kanban.comment" and actor_label:
        subject = actor_label

    return recast_gated(
        sealed,
        profile=profile_label,
        actor=actor_label,
        subject=subject,
        payload=payload,
        ref=ref,
    )


def _sanitize_memory_key(key: str, cast_cfg: SeriesCastConfig) -> str:
    """Replace profile directory prefixes in memory snapshot keys."""
    prof, sep, filename = key.partition("/")
    if not sep:
        return key
    safe_prof = sanitize_profile_value(prof, cast_cfg)
    return f"{safe_prof}{sep}{filename}"


def _sanitize_memory_hashes(
    hashes: Mapping[str, list[str]],
    cast_cfg: SeriesCastConfig,
) -> dict[str, list[str]]:
    """Replace profile directory prefixes in ``memory.json`` hash keys."""
    return {
        _sanitize_memory_key(key, cast_cfg): list(values)
        for key, values in hashes.items()
    }


def _sanitize_memory_counts(
    counts: Mapping[str, Any],
    cast_cfg: SeriesCastConfig,
) -> dict[str, Any]:
    """Replace profile directory prefixes inside ``memory.json`` / ``metrics.json`` counts."""
    out = dict(counts)
    raw_per_file = counts.get("per_file")
    if isinstance(raw_per_file, Mapping):
        per_file: dict[str, int] = {}
        for key, value in raw_per_file.items():
            safe_key = _sanitize_memory_key(str(key), cast_cfg)
            per_file[safe_key] = per_file.get(safe_key, 0) + int(value)
        out["per_file"] = per_file
    return out


def _primary_agent_for_episode(participating: set[str], cast_cfg: SeriesCastConfig) -> str:
    """Choose the primary mapped agent name for story panel jobs."""
    if "hermes:default" in participating and "hermes:default" in cast_cfg.roles:
        return cast_cfg.roles["hermes:default"].name
    for role_key in sorted(participating):
        if role_key != "owner" and role_key in cast_cfg.roles:
            return cast_cfg.roles[role_key].name
    if "hermes:default" in cast_cfg.roles:
        return cast_cfg.roles["hermes:default"].name
    return cast_cfg.owner.name


def _episode_panel_beats(
    arc: Any,
    all_beats: Sequence[Beat],
    brief: Any,
    evidence: Mapping[str, str],
    *,
    fmt: str,
) -> list[Beat]:
    """Return the exact beats that feed story panels for ``arc``."""
    beat_index = {beat.id: beat for beat in all_beats}
    brief_text_ids = {
        turn.event_id
        for turn in (getattr(brief, "turns", ()) or ())
        if getattr(turn, "event_id", "") and str(evidence.get(turn.event_id, "")).strip()
    }
    panel_beats = [
        beat_index[arc.roles[role]]
        for role in ARC_ROLES
        if role in arc.roles
        and arc.roles[role] in beat_index
        and any(eid in brief_text_ids for eid in beat_index[arc.roles[role]].event_ids)
    ]
    if fmt == "chapter":
        used = set(arc.roles.values())
        extra = [
            beat
            for beat in all_beats
            if beat.id not in used and any(eid in brief_text_ids for eid in beat.event_ids)
        ][: max(0, CHAPTER_MAX_STORY - len(panel_beats))]
        panel_beats.extend(extra)
    return panel_beats


def run_pipeline(
    config: PipelineConfig,
    *,
    source: HermesHistorySource | None = None,
) -> PipelineResult:
    result = PipelineResult()
    started = time.time()

    rule_set = list(rules_from_snapshot())
    if getattr(config, "privacy", "private") == "public":
        rule_set.extend(public_rules())
    redactor = DeterministicRedactor(rule_set)
    cast_cfg, loaded_kits = _resolve_cast_setup(config, redactor)
    config.data_dir.mkdir(parents=True, exist_ok=True)
    effective_hero = cast_cfg.owner.name if cast_cfg is not None else config.hero

    own_source = source is None
    snapshot_store = SnapshotStore(config.state_dir)
    # Audit M1-F4 in one line: the private snapshot may not live inside the evidence
    # folder, whoever passes the paths.
    if os.environ.get("REEL_ALLOW_CUSTOM_STATE") != "1":
        state_root = snapshot_store.root.resolve()
        data_root = config.data_dir.resolve()
        for forbidden in (data_root, data_root.parent):
            try:
                state_root.relative_to(forbidden)
            except ValueError:
                continue
            raise ValueError(
                "state_dir must not sit inside the evidence folder "
                f"({state_root} is under {forbidden}): the memory snapshot is private "
                "(mode 600, hashes + gated text). Set REEL_ALLOW_CUSTOM_STATE=1 only "
                "for tests."
            )
    prior_hashes = snapshot_store.hashes_by_file()
    if source is None:
        # V12: other harnesses read alongside (or instead of) Hermes.
        from adapters.harnesses.sessions import CompositeSource, make_source

        parts: list[Any] = []
        for harness in config.harnesses or ("hermes",):
            if harness == "hermes":
                parts.append(HermesHistorySource(
                    profiles=config.profiles,
                    all_profiles=config.all_profiles,
                    memory_snapshot=prior_hashes or None,
                    snapshot_root=config.snapshot_root,
                ))
            else:
                options = dict((config.harness_options or {}).get(harness, {}))
                parts.append(make_source(harness, root=str(options.get("root", "")),
                                         mapping=options.get("mapping"), agent_name=str(options.get("agent", ""))))
        source = parts[0] if len(parts) == 1 else CompositeSource(parts)

    try:
        # A pinned window is what makes "the same input twice" meaningful on a live
        # system: without it, two runs minutes apart look at different weeks.
        now = config.until or datetime.now(tz=timezone.utc).timestamp()
        window = Window(since=config.since if config.since is not None else now - config.days * 86400, until=config.until)

        # ---- ingest ------------------------------------------------------
        events = list(source.events(window))
        result.events_ingested = len(events)
        result.timings["ingest_s"] = round(time.time() - started, 2)

        # Memory facts snapshot for the *next* run's diff (decision 2). Hashes and
        # gated text only, in the private state dir, never in evidence (M1-F4).
        try:
            current_facts = source.current_memory_facts()
        except Exception as exc:  # noqa: BLE001
            current_facts = {}
            result.problems.append(f"memory read failed: {type(exc).__name__}: {exc}")

        # ---- text --------------------------------------------------------
        candidates = [event for event in events if event.kind in TEXT_PRIORITY]
        candidates.sort(
            key=lambda item: (TEXT_PRIORITY[item.kind], item.ts, item.id)
        )
        texts: dict[str, str] = {}
        for event in candidates[: config.max_text_events]:
            text = _text_for(source, event, config.max_text_chars)
            if text:
                texts[event.id] = text
        result.events_with_text = len(texts)
        result.timings["text_s"] = round(time.time() - started, 2)

        # ---- gate: deterministic + vocabulary ----------------------------
        cache_dir = (
            config.data_dir / "cache" / f"cast-{cast_cfg.series_id}-{cast_cfg.cast_digest}"
            if cast_cfg is not None
            else config.data_dir / "cache"
        )
        cache = ModelCache(cache_dir)
        # Never construct a model route on the caller's behalf: a run that was given no
        # client is the deterministic path, so nothing reaches an endpoint the operator
        # did not name.
        client: LLMClient = config.llm or NullLLM()
        cached = CachedLLM(client, cache, stage="m1")
        # R3-F6: the writer, title and entailment calls use the owner's main model. They
        # are given gated text only: the redaction model (when the operator names one) is
        # a separate client, so nothing ungated reaches the writer's endpoint.
        writer_cached = (
            CachedLLM(config.writer_llm, cache, stage="m1")
            if config.writer_llm is not None
            else cached
        )

        # Agent/profile names are the cast, not internal projects: keep them out of the
        # learned-sensitive vocabulary or every character turns into "Agent N".
        # For cast-configured runs, never allowlist raw directory/profile IDs.
        if cast_cfg is not None:
            cast_allowlist = tuple(sorted({*NON_CAST_ALLOWLIST, *cast_cfg.all_display_names()}))
            actor_role_by_id = {event.id: event_role_key(event) for event in events}
            agent_role_by_id = {event.id: event_agent_role_key(event) for event in events}
        else:
            cast_allowlist = tuple(sorted({*config.allowlist, *(p.name for p in source.info().profiles)}))
            actor_role_by_id = {}
            agent_role_by_id = {}
        model_gate = (
            # No model client, no model pass: without a named route redaction is the
            # deterministic regex pass only, rather than a gate that fails closed on
            # every candidate because it has nothing to ask.
            ModelEntityGate(cached, allowlist=cast_allowlist)
            if config.use_model and not isinstance(client, NullLLM)
            else None
        )
        gate = RedactionGate(
            redactor,
            privacy=getattr(config, "privacy", "private"),
            model_gate=model_gate,
            allowlist=cast_allowlist,
            max_chars=config.max_text_chars,
        )

        if cast_cfg is not None:
            mapped_roles = {"", "owner"} | set(cast_cfg.roles)

            def _is_mapped_event(event_id: str) -> bool:
                return (
                    actor_role_by_id.get(event_id, "") in mapped_roles
                    and agent_role_by_id.get(event_id, "") in mapped_roles
                )
        else:

            def _is_mapped_event(event_id: str) -> bool:
                return True

        if model_gate is not None and texts:
            vocab_texts = (
                [text for eid, text in texts.items() if _is_mapped_event(eid)]
                if cast_cfg is not None
                else texts.values()
            )
            if vocab_texts:
                gate.learn_vocabulary(vocab_texts, max_candidates=config.max_vocabulary)
        result.timings["vocabulary_s"] = round(time.time() - started, 2)

        gated: list[GatedEvent] = []
        gated_by_id: dict[str, GatedEvent] = {}
        for event in events:
            sealed = gate.gate_event(event, text=texts.get(event.id))
            if sealed is not None:
                if cast_cfg is not None:
                    sealed = _sanitize_gated_event(
                        sealed,
                        actor_role_by_id.get(sealed.id, ""),
                        agent_role_by_id.get(sealed.id, ""),
                        cast_cfg,
                    )
                gated.append(sealed)
                gated_by_id[sealed.id] = sealed
        result.gated = gate.report.gated
        result.quarantined = len(gate.report.quarantined)
        result.timings["gate_s"] = round(time.time() - started, 2)

        # ---- beats -------------------------------------------------------
        beats = beats_from_events(gated)
        result.beats = beats

        # ---- deep model pass on the story subset (decision 3, D5) --------
        if model_gate is not None:
            deep_ids = set(_story_subset(beats, config.max_deep_events))
            upgraded: list[GatedEvent] = []
            for event in events:
                existing = gated_by_id.get(event.id)
                if existing is None:
                    continue
                if event.id not in deep_ids:
                    upgraded.append(existing)
                    continue
                if cast_cfg is not None and not _is_mapped_event(event.id):
                    upgraded.append(existing)
                    continue
                sealed = gate.gate_event(event, text=texts.get(event.id), deep=True)
                if sealed is None:
                    # Quarantined at the deeper level: drop it from the story.
                    continue
                if cast_cfg is not None:
                    sealed = _sanitize_gated_event(
                        sealed,
                        actor_role_by_id.get(sealed.id, ""),
                        agent_role_by_id.get(sealed.id, ""),
                        cast_cfg,
                    )
                upgraded.append(sealed)
            gated = [item for item in upgraded if item is not None]
            gated_by_id = {item.id: item for item in gated}
            result.gated = len(gated)
            result.quarantined = len(gate.report.quarantined)
            beats = beats_from_events(gated)
            result.beats = beats
            result.timings["deep_gate_s"] = round(time.time() - started, 2)

        result.gate_report = gate.report.as_dict()
        # Audit M1-F5: a run that quarantines more than 5% of what it gated is not
        # stable enough to build a story on.
        processed = result.gated + result.quarantined
        result.quarantine_rate = (result.quarantined / processed) if processed else 0.0
        if result.quarantine_rate > 0.05:
            result.problems.append(
                f"quarantine rate {result.quarantine_rate:.1%} exceeds 5%"
            )
        result.cache_stats = cache.stats()

        # ---- arcs, canon, episode ---------------------------------------
        # Audit M1-F2: an arc comes from ONE linked thread, never from stitched events.
        threads = build_threads(gated)
        arcs = build_arcs(
            threads,
            beats,
            events_by_id=gated_by_id,
            window_start=window.since,
        )
        result.threads = len(threads)
        result.arc_count = len(arcs)
        canon = build_canon(gated, beats, hero_names=(effective_hero,), window_start=window.since)
        result.canon = canon

        if not arcs:
            result.problems.append("no arc could be built from this window")
            return result

        # Round 3 item 3: a script never shows a redaction marker. Every masked value
        # gets a stable fictional alias from the private alias book.
        alias_book = AliasBook.load(snapshot_store.root / "aliases.json")
        if cast_cfg is not None:
            cast_map = {"owner": effective_hero}
            for name in cast_cfg.all_display_names():
                cast_map[name] = name
            cast = list(dict.fromkeys([effective_hero, *cast_cfg.all_display_names()]))
            result.cast = cast_cfg.summary_entries()
            extra_allow = tuple(
                word for name in cast_cfg.all_display_names() for word in name.split() if word
            )
            claim_allow = tuple(dict.fromkeys(tuple(cast_allowlist) + extra_allow + (effective_hero,)))
            claim_cast = tuple(dict.fromkeys(cast_cfg.all_display_names() + (effective_hero,)))
        else:
            cast = [effective_hero] + sorted(
                {
                    beat.profile
                    for beat in beats
                    if beat.profile and beat.profile != effective_hero
                }
            )
            # Audit M1-F6: profiles are characters, not database ids.
            cast_map = build_cast_map(
                [beat.profile for beat in beats if beat.profile],
                hero=effective_hero,
            )
            cast_map["default"] = cast_map.get("default", "Kensei")
            # R4-F1: the owner is a character in his own story, and no one is called "Assistant".
            cast_map["owner"] = effective_hero
            cast_map = {
                profile: character
                for profile, character in cast_map.items()
                if character.strip().lower() not in {"assistant", "default", "agent", ""}
            }
            # A character name is released in the evidence, so it must survive the gate:
            # a profile whose name matches a learned sensitive term gets a neutral alias.
            safe_cast: dict[str, str] = {}
            for index, (profile, character) in enumerate(sorted(cast_map.items()), start=1):
                redacted, findings = gate.redact_text(character)
                safe_cast[profile] = character if not findings else f"Agent {index}"
            cast_map = safe_cast
            result.cast = summarise_cast(cast_map)
            claim_allow = tuple(cast_map.values()) + tuple(config.allowlist) + (effective_hero,)
            claim_cast = tuple(dict.fromkeys(tuple(cast_map.values()) + (effective_hero,)))

        writer: Any
        if config.writer is not None:
            writer = config.writer
        elif config.use_model:
            writer = LLMDialogueWriter(writer_cached, cast)
        else:
            # Round 8 / R7-F1: captions are written prose. With no model there is no
            # writer, so the run produces no strip and reports why.
            writer = TemplateDialogueWriter(cast, hero=effective_hero)

        # Audit M1-F1 + round 3: claims in both layers must be supported by the gated
        # content of the events the line cites.
        grounder = ClaimGate(
            writer_cached if config.use_model else None,
            allow=claim_allow,
            # r10b / QAR10-01: a caption naming a character must find that name in its
            # cited evidence; the allowlist only makes the name legal vocabulary.
            cast=claim_cast,
            meaning=config.wording == "meaning",
        )
        # Who did what: only an actor of a cited event may speak its panel (R2-F2).
        actors_by_event: dict[str, str] = {}
        for event in gated:
            if cast_cfg is not None:
                a_role = actor_role_by_id.get(event.id, "")
                g_role = agent_role_by_id.get(event.id, "")
                if is_unmapped_label(event.actor) or is_unmapped_label(event.profile):
                    actors_by_event[event.id] = ""
                elif a_role == "owner":
                    actors_by_event[event.id] = effective_hero
                elif a_role and a_role in cast_cfg.roles:
                    actors_by_event[event.id] = cast_cfg.roles[a_role].name
                elif g_role and g_role in cast_cfg.roles:
                    actors_by_event[event.id] = cast_cfg.roles[g_role].name
                else:
                    actors_by_event[event.id] = ""
                continue
            # Prefer the profile: on message events the actor is a tool name, and a tool
            # cannot speak. Fall back to the actor only when it is a known character.
            # QAR-M2-01: the owner's own messages belong to the owner, whichever profile ran.
            who = effective_hero if event.actor == "owner" else (cast_map.get(event.profile) or cast_map.get(event.actor) or "")
            actors_by_event[event.id] = who

        evidence_map = evidence_by_event(gated, display=cast_map, hero=effective_hero)
        # The gated text of each message: the pool the script's real lines are drawn from.
        # Gated means redacted; nothing here comes from raw history.
        quotes_by_event: dict[str, str] = {}
        if config.real_lines:
            for event in gated:
                text = (event.payload or {}).get("text")
                if (
                    event.kind in {KIND_MESSAGE_USER, KIND_MESSAGE_ASSISTANT}
                    and isinstance(text, str)
                    and text.strip()
                ):
                    quotes_by_event[event.id] = text
        cast_names = tuple(dict.fromkeys(tuple(cast_map.values()) + (effective_hero,)))
        structured_captions = {
            event.id: compose_caption(event, display=cast_map, hero=effective_hero)
            for event in gated
        }

        # R3-F1: pick stories by meaning (sessions with a goal and an outcome, or task
        # families whose titles describe real work). Bookkeeping is never a story.
        # Keep the full ranked list here; the loop stops once it reaches `episodes`,
        # so skipped or unmapped top-ranked stories never crowd out eligible ones.
        selected, all_candidates = select_stories(gated, limit=None)
        # Round 10: story rows carry text cut at 160 characters, and a cut can leave a
        # fragment the evidence writer reads as a secret (it refused the whole metrics file
        # on the September window). Re-gate the cut text so evidence holds the gated form.
        result.stories = []
        for candidate in all_candidates[:12]:
            row = candidate.as_dict()
            for field_name in ("title", "goal", "outcome"):
                if isinstance(row.get(field_name), str) and row[field_name]:
                    row[field_name], _ = gate.redact_text(row[field_name])
            result.stories.append(row)
        threads_index = {thread.key: thread for thread in threads}
        # Round 10: a selected session can be a child in a lineage, whose thread key is the
        # lineage root, so lookup by key missed it ("no thread for this story" for 5 of 11
        # r9 candidates). Resolve through the candidate's own events instead.
        thread_by_event = thread_index(threads)
        grouped_beats = beats_by_thread(beats, threads)
        thread_events: dict[str, list[GatedEvent]] = {}
        for event in gated:
            for thread in threads:
                if event.id in thread.event_ids:
                    thread_events.setdefault(thread.id, []).append(event)
                    break

        episodes: list[dict[str, Any]] = []
        ir: EpisodeIR | None = None
        skipped: list[dict[str, str]] = []
        used_anchors: set[str] = set()
        used_arcs: set[str] = set()
        from canon.ideas import draw as draw_frame

        series_path = series_file_for(config.data_dir, cast_cfg.series_id if cast_cfg is not None else "")
        series = load_series(series_path)
        existing_ep_nums = (
            _existing_episode_numbers(config.data_dir)
            if (cast_cfg is not None or bool(series.chapters))
            else set()
        )
        allocated_ep_nums: set[int] = set(existing_ep_nums)
        selected, series_skipped = order_for_series(selected, series)
        skipped.extend(series_skipped)
        for candidate in selected:
            if len(episodes) >= max(1, config.episodes):
                break
            thread = threads_index.get(candidate.key) or _thread_for_candidate(
                candidate, thread_by_event
            )
            if thread is None:
                skipped.append({"key": candidate.key, "reason": "no thread for this story"})
                continue
            t_events = thread_events.get(thread.id, [])
            # R3-F2: build deterministic brief, arc, panel beats, and facts first (0 LLM calls)
            # so participation gating reflects what this episode actually relies on.
            brief = build_brief(
                thread.key,
                thread.kind,
                t_events,
                gate=None,
                max_turns=CHAPTER_MAX_TURNS if config.format == "chapter" else MAX_TURNS,
            )
            if not brief.ok:
                missing = [
                    name for name, value in (("goal", brief.goal), ("outcome", brief.outcome)) if not value
                ]
                skipped.append(
                    {
                        "key": candidate.key,
                        "reason": f"brief not usable ({len(brief.turns)} turns"
                        + (f"; no {' or '.join(missing)}" if missing else "")
                        + ")",
                    }
                )
                continue
            thread_beats = grouped_beats.get(thread.id, [])
            thread_event_map = {event.id: event for event in t_events}
            # R3-F3: panels are anchored on beats that carry gated text. A kanban thread
            # has lifecycle beats; a session thread usually has too few, so its beats
            # are built from the brief itself (round 4).
            if usable_anchor_count(thread_beats, brief=brief, evidence=evidence_map) >= 3:
                arc = build_arc(
                    thread_beats, thread=thread, events_by_id=thread_event_map
                )
                story_beats = thread_beats
            else:
                arc, story_beats = build_brief_arc(brief, thread_event_map)
            if arc is None:
                skipped.append({"key": candidate.key, "reason": "no arc"})
                continue
            target_arc = (
                _trim_arc(arc, config.strip_panels)
                if (config.format == "strip" and config.strip_panels)
                else arc
            )
            all_arc_beats = story_beats or thread_beats
            panel_beats = _episode_panel_beats(
                target_arc, all_arc_beats, brief, evidence_map, fmt=config.format
            )
            thread_facts = build_facts(
                brief, thread_event_map, display=cast_map, hero=effective_hero
            )

            participating: set[str] = set()
            if cast_cfg is not None:
                relied_event_ids = (
                    ({brief.goal_event_id} if brief.goal_event_id else set())
                    | ({brief.outcome_event_id} if brief.outcome_event_id else set())
                    | {turn.event_id for turn in brief.turns if turn.event_id}
                    | {eid for beat in panel_beats for eid in beat.event_ids}
                    | {fact.event_id for fact in thread_facts if fact.event_id}
                )
                participating = (
                    {actor_role_by_id.get(eid, "") for eid in relied_event_ids}
                    | {agent_role_by_id.get(eid, "") for eid in relied_event_ids}
                ) - {""}
                unmapped = sorted(r for r in participating if r != "owner" and r not in cast_cfg.roles)
                if unmapped:
                    opaque = ", ".join(opaque_role_id(r) for r in unmapped)
                    skipped.append({"key": candidate.key, "reason": f"unmapped participating role ({opaque})"})
                    result.cast_diagnostics.append(
                        f"candidate {candidate.key}: unmapped participating role(s): {', '.join(unmapped)}"
                    )
                    continue
                story_t_events = [ev for ev in t_events if _is_mapped_event(ev.id)]
                episode_beats: Sequence[Beat] = [
                    beat for beat in all_arc_beats if all(_is_mapped_event(eid) for eid in beat.event_ids)
                ]
                # Rebuild arc from mapped beats/events if unused unmapped beats were stripped.
                if len(episode_beats) != len(all_arc_beats):
                    story_event_map_mapped = {event.id: event for event in story_t_events}
                    if usable_anchor_count(episode_beats, brief=brief, evidence=evidence_map) >= 3:
                        arc = build_arc(
                            episode_beats, thread=thread, events_by_id=story_event_map_mapped
                        )
                    else:
                        arc, episode_beats = build_brief_arc(brief, story_event_map_mapped)
                    if arc is None:
                        skipped.append({"key": candidate.key, "reason": "no arc"})
                        continue
            else:
                story_t_events = t_events
                episode_beats = all_arc_beats

            if config.use_model:
                brief = entail_brief(brief, story_t_events, grounder)
                if not brief.ok:
                    missing = [
                        name for name, value in (("goal", brief.goal), ("outcome", brief.outcome)) if not value
                    ]
                    skipped.append(
                        {
                            "key": candidate.key,
                            "reason": f"brief not usable ({len(brief.turns)} turns"
                            + (f"; no {' or '.join(missing)}" if missing else "")
                            + ")",
                        }
                    )
                    continue
                story_event_map = {event.id: event for event in story_t_events}
                thread_facts = build_facts(
                    brief, story_event_map, display=cast_map, hero=effective_hero
                )

            # The shape of the story is read from what actually happened, not assumed.
            arc = replace(arc, type=classify_arc_type(goal=brief.goal, turns=turn_texts(brief), outcome=brief.outcome))
            # R4-F5: two episodes in one run must not stand on the same events.
            beat_index = {beat.id: beat for beat in episode_beats}
            anchor_events = {
                event_id
                for beat_id in arc.roles.values()
                for event_id in (beat_index.get(beat_id).event_ids if beat_index.get(beat_id) else ())
            }
            if anchor_events & used_anchors:
                skipped.append(
                    {"key": candidate.key, "reason": "shares anchor events with an earlier episode"}
                )
                continue
            owner_texts = tuple(
                str((event.payload or {}).get("text", ""))
                for event in story_t_events
                if str((event.payload or {}).get("role", "")) == "user"
                and str((event.payload or {}).get("text", "")).strip()
            )
            outcome_caption = ""
            if brief.outcome:
                for event in story_t_events:
                    if str((event.payload or {}).get("text", "")).strip() == brief.outcome.strip():
                        outcome_caption = compose_caption(event, display=cast_map, hero=effective_hero)
                        break
                if not outcome_caption:
                    outcome_caption = trim_caption(brief.outcome)
            thread_facts_text = facts_text(thread_facts)
            thread_evidence_text = "\n".join(
                [brief.text()]
                + [
                    evidence_map.get(event_id, "")
                    for event_id in candidate.event_ids
                    if cast_cfg is None or _is_mapped_event(event_id)
                ]
            )
            # R5-F4: if the model title fails grounding, fall back to the thread title
            # shortened on a word boundary - never to the owner's own message.
            raw_title = str(candidate.title or arc.title or "").strip()
            # R5-F4: the title is never the owner's own message. When the thread's only
            # title is that message, name the kind of work instead.
            owner_haystack = " ".join(owner_texts).lower()
            if raw_title and raw_title.lower()[:24] in owner_haystack:
                raw_title = "A session" if candidate.kind == "session" else "A task"
            derived_title = trim_caption(raw_title, max_words=6)
            title, title_source = write_title(
                writer_cached if config.use_model else None,
                beats=episode_beats,
                derived=derived_title,
                hero=effective_hero,
                gate=grounder,
                evidence=brief.text(),
                alias=alias_book.resolve,
            )
            if arc.id in used_arcs:
                # Two candidates can resolve to the same thread; a strip is told once.
                skipped.append({"key": candidate.key, "reason": "thread already told"})
                continue
            index = len(episodes) + 1
            if config.format == "strip" and config.strip_panels:
                arc = _trim_arc(arc, config.strip_panels)
            # M3: one frame per chapter - a different world/villain/genre/motif/shot, drawn
            # deterministically from the hero and the chapter number, avoiding recent looks.
            # Advance saved episode numbers monotonically with the series ledger so
            # follow-on runs (including adding co-stars) never overwrite prior plans.
            chapter_number = series.next_number
            min_free = max([chapter_number, *(num + 1 for num in allocated_ep_nums)])
            episode_number = max(config.number + index - 1, min_free)
            frame = draw_frame(effective_hero, chapter_number, avoid=series.avoid())

            def _plan(episode_writer: Any, layer: str = "caption") -> EpisodeIR:
                return plan_episode(
                    arc,
                    episode_beats,
                    canon,
                    format=config.format,
                    season=config.season,
                    number=episode_number,
                    writer=episode_writer,
                    hero=effective_hero,
                    grounder=grounder,
                    evidence=evidence_map,
                    brief=brief,
                    structured_captions=structured_captions,
                    actors_by_event=actors_by_event,
                    cast_map=cast_map,
                    thread_evidence=thread_evidence_text,
                    owner_texts=owner_texts,
                    outcome_caption=outcome_caption,
                    facts=thread_facts,
                    facts_evidence=thread_facts_text,
                    title=title,
                    frame=frame,
                    caption_layer=layer,
                    quotes_by_event=quotes_by_event or None,
                )

            # V6-V8: story mode - a plain-English synopsis first, the script written from
            # it in character voices, then an editor read and at most one rewrite pass.
            story_mode = (
                config.wording == "meaning" and config.use_model
                and config.writer is None and isinstance(writer, LLMDialogueWriter)
            )
            story_info: dict[str, Any] | None = None
            try:
                if story_mode:
                    synopsis = write_synopsis(
                        writer_cached.complete_json, title=title, facts_text=thread_facts_text,
                        brief_text=brief.text(), gate=grounder,
                    )
                    if synopsis.fields.get("title"):
                        # V6: a story title from the checked synopsis beats "A session".
                        title, title_source = alias_book.resolve(synopsis.fields["title"]), "synopsis"
                    from canon.tones import auto_tone, parse_tone

                    if str(config.tone).strip().lower() == "auto":
                        tone = auto_tone(drama=int(getattr(candidate, "drama", 0) or 0), finished=arc.status == "resolved")
                    else:
                        tone = parse_tone(config.tone)
                    merged_defaults = {**DEFAULT_VOICES, **dict(config.voices or {})}
                    story_voices = (
                        resolve_voices(cast_cfg, loaded_kits, merged_defaults)
                        if cast_cfg is not None
                        else merged_defaults
                    )
                    primary_agent = (
                        _primary_agent_for_episode(participating, cast_cfg)
                        if cast_cfg is not None
                        else (cast_map.get("default") or "Kensei")
                    )
                    story_ctx = {
                        "tone": tone.writer_text(),
                        "voices": story_voices,
                        "synopsis": synopsis.text(),
                        "hook": synopsis.fields.get("hook", ""),
                        "jobs": jobs_for(arc.type, hero=effective_hero, agent=primary_agent),
                        "notes": {},
                    }
                    first_writer = LLMDialogueWriter(writer_cached, cast, story=story_ctx)
                    # Story captions may interpret (stakes, feeling) but add no event, name
                    # or number; `--strict-captions` keeps them literal.
                    narration = "caption" if config.strict_captions else "story"
                    episode_ir = _plan(first_writer, narration)
                    review = edit_script(writer_cached.complete_json,
                                         [p.as_dict() for p in episode_ir.panels], synopsis=synopsis.text(), tone=tone.label,
                                         cast=cast_names)
                    reviews = [review]
                    jargon = first_writer.jargon_dropped
                    if 0 < review["score"] < 4 and review["problems"]:
                        notes = {str(item["panel"]).rsplit("-", 1)[-1]: item["issue"] for item in review["problems"]}
                        second_writer = LLMDialogueWriter(writer_cached, cast, story=dict(story_ctx, notes=notes))
                        try:
                            second = _plan(second_writer, narration)
                        except EpisodeRefused:
                            second = None
                        if second is not None:
                            review2 = edit_script(writer_cached.complete_json,
                                                  [p.as_dict() for p in second.panels], synopsis=synopsis.text(), tone=tone.label,
                                                  cast=cast_names)
                            reviews.append(review2)
                            if len(second.panels) >= len(episode_ir.panels) - 1 and review2["score"] >= review["score"]:
                                episode_ir = second
                                jargon = second_writer.jargon_dropped
                    story_info = {"arc_type": arc.type, "synopsis": synopsis.as_dict(), "editor": reviews, "jargon_dropped": jargon,
                                  "tone": tone.as_dict()}
                else:
                    episode_ir = _plan(writer)
            except EpisodeRefused as refused:
                skipped.append({"key": candidate.key, "reason": str(refused)})
                continue
            # Round 10: D7 says a strip is 4-6 panels; three was below the spec.
            # V4: a chapter needs 7 true story panels; fillers bring it to 16+ panels.
            min_panels = config.min_strip_panels if config.format == "strip" else 7
            if len(episode_ir.panels) < min_panels:
                # Round 9: a refusal must say why in terms a human can agree with.
                why = "; ".join(episode_ir.grounding.get("reasons", [])[-3:])
                skipped.append(
                    {
                        "key": candidate.key,
                        "reason": (
                            f"fewer than {min_panels} panels ({len(episode_ir.panels)} kept"
                            + (f"; {why}" if why else "")
                            + ")"
                        ),
                    }
                )
                continue
            planned_ir = episode_ir  # pre-alias: the text the citations were ranked on
            episode_ir = apply_to_ir(episode_ir, alias_book)
            if cast_cfg is not None:
                ep_names = episode_cast(episode_ir.as_dict())
                episode_ir = replace(episode_ir, cast_binding=build_cast_binding(cast_cfg, ep_names))
            stray = masks_in(markdown_of(episode_ir))
            if stray:
                result.problems.append(
                    f"episode {episode_number} still shows {len(stray)} redaction marker(s)"
                )
            episodes.append(
                {
                    "number": episode_number,
                    "story_key": candidate.key,
                    "story_kind": candidate.kind,
                    "thread_kind": arc.thread_kind,
                    "thread_label": arc.thread_label,
                    "title": title,
                    "title_words": len(title.split()),
                    "header_title": episode_ir.logline,
                    "header_title_words": len(episode_ir.logline.split()),
                    "title_source": title_source,
                    "chapter": chapter_number,
                    "frame": frame,
                    "panels": len(episode_ir.panels),
                    "cliffhanger": episode_ir.cliffhanger,
                    "grounding": dict(episode_ir.grounding),
                    "brief": brief.as_dict(),
                    "facts": [fact.as_dict() for fact in thread_facts],
                    # t_abdaa3a9: captions also cite facts by their source text.
                    "fact_sources": [str(getattr(fact, "source_text", "") or "") for fact in thread_facts],
                    "ir": episode_ir,
                    "planned_ir": planned_ir,
                    "arc": arc,
                    # The beats the episode was actually planned from (t_abdaa3a9).
                    "story_beats": tuple(episode_beats),
                    "story": story_info,
                }
            )
            allocated_ep_nums.add(episode_number)
            used_anchors |= anchor_events
            used_arcs.add(arc.id)
            # M3: remember the chapter so the next run follows on and never retells it.
            series.record(
                number=chapter_number,
                title=title,
                thread_key=str(getattr(candidate, "key", "")),
                arc_id=arc.id,
                frame=frame,
                cliffhanger=bool(episode_ir.cliffhanger),
            )
            if index == 1:
                ir = episode_ir
                result.title = title
                result.title_source = title_source
        result.skipped_stories = skipped
        alias_book.save()
        if ir is None:
            # Honest empty run: no story could be told, but the run still reports its
            # gate, its stories and its skips so the reason is visible.
            if result.cast_diagnostics:
                result.problems.append(
                    "unmapped participating role stopped candidate episode(s); see local cast diagnostics"
                )
            result.problems.append("no episode could be planned")
            result.episodes = []
            result.skipped_stories = skipped
            result.timings["episode_s"] = round(time.time() - started, 2)
            result.timings["writer"] = "none"
        if cast_cfg is not None:
            series.series_id = cast_cfg.series_id
            series.cast_digest = cast_cfg.cast_digest
            series.cast_roles = cast_cfg.ledger_roles()
        save_series(series_path, series)
        result.episodes = [
            {key: value for key, value in row.items() if key not in {"ir", "planned_ir", "arc", "story_beats", "story", "fact_sources"}}
            for row in episodes
        ]
        if ir is not None:
            result.grounding = dict(ir.grounding)
            if cast_cfg is None:
                result.cast = summarise_cast(cast_map)
            result.ir = ir
        result.timings["episode_s"] = round(time.time() - started, 2)
        result.llm_usage = {
            "redaction": getattr(client, "usage", lambda: {})(),
            "writer": getattr(config.writer_llm, "usage", lambda: {})(),
            "writer_endpoint": getattr(config.writer_llm, "name", "local"),
        }
        if isinstance(result.llm_usage["writer"], dict) and result.llm_usage["writer"].get("budget"):
            if result.llm_usage["writer"]["calls"] > result.llm_usage["writer"]["budget"]:
                result.problems.append("writer budget exceeded")

        # ---- gates on the output ----------------------------------------
        known = set(gated_by_id)
        if ir is None:
            # Nothing was written, so there is nothing to validate or scan. The run has
            # already reported why in problems.
            result.problems.extend([])
            markdown = ""
        else:
            # r10b / QAR10-03: every episode is validated, not only the first, and a
            # caption must carry resolvable source ids like a line does.
            for row in episodes:
                for problem in validate_ir(row["ir"], known_event_ids=known):
                    result.problems.append(f"episode {row['number']}: {problem}")
            markdown = render_script_markdown(ir)
        hits = leakscan.scan_text(markdown)
        if hits:
            result.problems.append(
                "leak scan found "
                + ", ".join(sorted({hit.pattern for hit in hits}))
                + " in the rendered script"
            )
        for panel in (ir.panels if ir is not None else ()):
            for line in panel.lines:
                name_hits = leakscan.scan_text(f"{line.speaker}: {line.text}")
                if any(hit.pattern == "high_entropy" for hit in name_hits):
                    result.problems.append(f"high-entropy token in {panel.id}")

        # ---- private memory snapshot (never evidence) --------------------
        memory_counts: dict[str, Any] = {"files": 0, "facts": 0}
        memory_hashes: dict[str, list[str]] = {}
        try:
            gated_by_file: dict[str, list[str]] = {}
            for key, facts in current_facts.items():
                values: list[str] = []
                for fact in facts:
                    try:
                        redacted, _ = gate.redact_text(fact)
                    except Exception:  # noqa: BLE001 - quarantine, do not store raw
                        redacted = "[REDACTED:health]"
                    values.append(redacted)
                gated_by_file[key] = values
            snapshot = build_snapshot(current_facts, captured_at=now, gated_text=gated_by_file)
            snapshot_store.save(snapshot)
            memory_counts = snapshot.counts()
            memory_hashes = snapshot.hashes_by_file()
            if cast_cfg is not None:
                memory_counts = _sanitize_memory_counts(memory_counts, cast_cfg)
                memory_hashes = _sanitize_memory_hashes(memory_hashes, cast_cfg)
        except Exception as exc:  # noqa: BLE001
            result.problems.append(f"memory snapshot failed: {type(exc).__name__}: {exc}")

        # ---- artifacts (every value re-checked by the evidence writer) ----
        try:
            write_json(
                config.data_dir / "beats.json",
                {"beats": [beat.as_dict() for beat in beats]},
                gate,
            )
            write_json(config.data_dir / "arcs.json", {"arcs": [arc.as_dict() for arc in arcs]}, gate)
            write_json(config.data_dir / "canon.json", canon.as_dict(), gate)
            if ir is not None:
                write_json(config.data_dir / "script-ir.json", ir.as_dict(), gate)
            for row in episodes:
                episode_ir = row["ir"]
                ir_out = config.data_dir / f"script-ir-{row['number']}.json"
                if (cast_cfg is not None or existing_ep_nums) and ir_out.exists():
                    raise CastConfigError(
                        f"refusing to overwrite existing planned episode {ir_out.name} in {config.data_dir}"
                    )
                write_json(
                    ir_out,
                    episode_ir.as_dict(),
                    gate,
                )
                write_text(
                    config.data_dir / f"episode-{row['number']}-script.md",
                    render_script_markdown(episode_ir),
                    gate,
                    max_chars=MAX_SCRIPT_CHARS,
                )
                if row.get("story"):
                    write_json(config.data_dir / f"episode-{row['number']}-story.json", row["story"], gate)
                    facts_by_panel = {
                        panel.id: " / ".join(
                            " ".join(str(evidence_map.get(event_id, "")).split())[:240]
                            for event_id in panel.caption_source_event_ids[:2]
                            if evidence_map.get(event_id)
                        )
                        for panel in episode_ir.panels
                    }
                    # Evidence files carry short strings only, so the notes go out per panel.
                    write_json(
                        config.data_dir / f"episode-{row['number']}-behind-the-panels.json",
                        {"panels": [
                            {"panel": number, "id": panel.id, "role": panel.role,
                             "details": facts_by_panel.get(panel.id, ""),
                             "sources": list(panel.caption_source_event_ids[:3])}
                            for number, panel in enumerate(episode_ir.panels, start=1)
                            if facts_by_panel.get(panel.id)
                        ]},
                        gate,
                    )
                write_json(
                    config.data_dir / f"episode-{row['number']}-trace.json",
                    episode_trace(
                        episode_ir,
                        row["arc"],
                        row["story_beats"],
                        row["facts"],
                        evidence_map,
                        regate=lambda text: gate.redact_text(text)[0],
                        clean=lambda text: not gate.scan(text),
                        fact_sources=row.get("fact_sources") or (),
                        planned_ir=row.get("planned_ir"),
                    ),
                    gate,
                )
            write_json(
                config.data_dir / "memory.json",
                {"counts": memory_counts, "hashes": memory_hashes},
                gate,
            )
            script_path = config.data_dir / "episode-script.md"
            write_text(script_path, markdown, gate, max_chars=MAX_SCRIPT_CHARS)
            write_json(
                config.data_dir / "metrics.json",
                {
                    "generated_at": _now_iso(),
                    "window": {"since": window.since, "until": window.until, "days": config.days},
                    "events_ingested": result.events_ingested,
                    "events_with_text": result.events_with_text,
                    "gated": result.gated,
                    "quarantined": result.quarantined,
                    "beats": len(beats),
                    "arcs": len(arcs),
                    "panels": len(ir.panels) if ir is not None else 0,
                    "writer": ir.writer if ir is not None else "",
                    "grounding": result.grounding,
                    "episodes": result.episodes,
                    "llm_usage": result.llm_usage,
                    "stories": result.stories,
                    "skipped_stories": result.skipped_stories,
                    "threads": result.threads,
                    "title": result.title,
                    "header_title": ir.logline if ir is not None else "",
                    "header_title_words": len(ir.logline.split()) if ir is not None else 0,
                    "title_source": result.title_source,
                    "cast": result.cast,
                    "quarantine_rate": round(result.quarantine_rate, 4),
                    "primary_thread": (arcs[0].as_dict()["thread_kind"] if arcs else ""),
                    "causation_verified": (arcs[0].causation.get("verified") if arcs else None),
                    "gate": result.gate_report,
                    "cache": result.cache_stats,
                    "timings": result.timings,
                    "problems": result.problems,
                    "memory": memory_counts,
                },
                gate,
            )
        except EvidenceLeak as exc:
            result.problems.append(f"evidence refused: {exc}")

        result.artifacts = {
            name: name
            for name in (
                "canon.json",
                "beats.json",
                "arcs.json",
                "script-ir.json",
                "episode-script.md",
                "memory.json",
                "metrics.json",
            )
        }
        result.timings["total_s"] = round(time.time() - started, 2)
        return result
    finally:
        if own_source and source is not None:
            source.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _install_signal_cleanup() -> None:
    """Turn SIGTERM/SIGINT into a normal exit so snapshot copies are removed.

    ``weakref.finalize`` covers garbage collection and a normal interpreter exit,
    but a process killed by a signal never unwinds - which is how private copies of
    the user's databases end up lingering in the snapshot root.
    """

    def _handler(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    for name in ("SIGTERM", "SIGINT"):
        signal_number = getattr(signal, name, None)
        if signal_number is None:
            continue
        try:
            signal.signal(signal_number, _handler)
        except (ValueError, OSError):  # not the main thread, or unsupported
            continue


def main(argv: Sequence[str] | None = None) -> int:
    _install_signal_cleanup()
    parser = argparse.ArgumentParser(
        prog="python -m adapters.hermes.pipeline",
        description="Agent Reel M1 pipeline: ingest, gate, beats, arcs, canon, episode.",
    )
    parser.add_argument("--data-dir", required=True, help="where artifacts and the cache live")
    parser.add_argument("--days", type=float, default=7.0)
    parser.add_argument(
        "--since",
        type=float,
        default=None,
        help="pin the window start (epoch seconds); also freezes determinism",
    )
    parser.add_argument(
        "--until", type=float, default=None, help="pin the window end (epoch seconds)"
    )
    parser.add_argument("--profile", action="append", default=None)
    parser.add_argument(
        "--all-profiles",
        action="store_true",
        help=(
            "read every profile under <home>/profiles. Default reads only the profile this "
            "process runs as, because profiles are kept separate by design"
        ),
    )
    parser.add_argument("--format", choices=("strip", "chapter"), default="strip")
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--season", type=int, default=1)
    parser.add_argument(
        "--llm",
        choices=("none", "local"),
        default="none",
        help="none = deterministic only; local = the rig's serving route (no keys)",
    )
    parser.add_argument("--llm-url", default=os.environ.get("REEL_LLM_BASE_URL"))
    parser.add_argument("--llm-model", default=os.environ.get("REEL_LLM_MODEL", "auto"))
    parser.add_argument("--max-deep-events", type=int, default=MAX_DEEP_EVENTS)
    parser.add_argument(
        "--max-vocabulary",
        type=int,
        default=400,
        help="cap on distinct name candidates sent to the model (most frequent first)",
    )
    parser.add_argument(
        "--max-text-events",
        type=int,
        default=MAX_TEXT_EVENTS,
        help="cap on events whose text is fetched and gated (priority ordered)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        help=(
            "private state dir for the memory snapshot (mode 600, outside evidence); "
            "defaults to $REEL_STATE_DIR or ~/.local/state/agent-reel"
        ),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="how many threads to turn into scripts (each from a different thread)",
    )
    parser.add_argument(
        "--privacy",
        choices=("private", "public"),
        default="private",
        help="public masks money and health (D13); private is the owner's own view",
    )
    parser.add_argument("--json", action="store_true", help="print the metrics JSON")
    args = parser.parse_args(list(argv) if argv is not None else None)

    client: LLMClient | None = None
    if args.llm == "local":
        if not args.llm_url:
            parser.error("--llm local needs --llm-url or REEL_LLM_BASE_URL")
        # Reasoning models spend the whole budget on reasoning_content and return
        # content: null, which looks like an empty answer. Off for JSON tasks.
        client = OpenAICompatLLMClient(
            base_url=args.llm_url,
            model=args.llm_model,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    # The writer/title/entailment route is the plugin's ctx.llm (see
    # adapters/hermes/plugin_llm.py); this CLI only runs the pipeline itself, so it
    # either uses the local model for everything or nothing at all.
    writer_llm: Any = None
    writer_note = ""

    if client is not None:
        fallback_url = os.environ.get("REEL_LLM_FALLBACK_URL")
        fallback = (
            OpenAICompatLLMClient(
                base_url=fallback_url,
                model=os.environ.get("REEL_LLM_FALLBACK_MODEL", args.llm_model),
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            if fallback_url
            else None
        )
        client = FallbackLLM(client, fallback)

    config = PipelineConfig(
        data_dir=Path(args.data_dir).expanduser(),
        days=args.days,
        since=args.since,
        until=args.until,
        profiles=tuple(args.profile) if args.profile else None,
        all_profiles=args.all_profiles,
        format=args.format,
        season=args.season,
        number=args.number,
        use_model=args.llm != "none",
        max_deep_events=args.max_deep_events,
        max_text_events=args.max_text_events,
        max_vocabulary=args.max_vocabulary,
        privacy=args.privacy,
        episodes=max(1, args.episodes),
        writer_llm=writer_llm,
        state_dir=Path(args.state_dir).expanduser() if args.state_dir else None,
        llm=client,
    )
    result = run_pipeline(config)
    if writer_note == "api-server" and writer_llm is not None:
        result.problems = [p for p in result.problems]
    elif writer_note:
        result.timings["writer_note"] = writer_note
    payload = {
        "ok": result.ok,
        "events_ingested": result.events_ingested,
        "events_with_text": result.events_with_text,
        "gated": result.gated,
        "quarantined": result.quarantined,
        "beats": len(result.beats),
        "arcs": result.arc_count,
        "panels": len(result.ir.panels) if result.ir else 0,
        "writer": result.ir.writer if result.ir else "",
        "grounding": result.grounding,
        "threads": result.threads,
        "title": result.title,
        "title_source": result.title_source,
        "quarantine_rate": round(result.quarantine_rate, 4),
        "gate": result.gate_report,
        "cache": result.cache_stats,
        "timings": result.timings,
        "problems": result.problems,
        "artifacts": result.artifacts,
    }
    if args.json:
        print(json.dumps(payload, indent=1, sort_keys=True))
    else:
        print(f"events: {result.events_ingested} (with text: {result.events_with_text})")
        print(f"gated: {result.gated}, quarantined: {result.quarantined}")
        print(f"beats: {len(result.beats)}, arcs: {result.arc_count}")
        if result.ir:
            print(f"episode: {len(result.ir.panels)} panels, dialogue from {result.ir.writer}")
            print(f"script: {result.artifacts.get('script_md')}")
        if result.problems:
            print("problems:")
            for problem in result.problems[:10]:
                print(f"  - {problem}")
    return 0 if result.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
