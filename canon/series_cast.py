"""Opt-in series cast configuration and episode cast-binding receipts.

Private role keys (``owner``, ``hermes:<profile>``, ``harness:<name>``) live only in
the operator's series configuration and local diagnostics. Downstream episode artifacts
carry only safe display names and exact kit slugs (``cast_binding``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from canon.assets import CharacterKit

CAST_CONFIG_SCHEMA_VERSION = 1
CAST_BINDING_SCHEMA_VERSION = 1
UNMAPPED_ROLE_PREFIX = "unmapped-role-"

SERIES_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
ROLE_KEY_RE = re.compile(r"^(hermes|harness):[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
RESERVED_DISPLAY_NAMES = frozenset(
    {"assistant", "default", "agent", "the fleet", "user", "owner", "narrator", ""}
)

_TOP_LEVEL_KEYS = frozenset({"schema_version", "series_id", "assets_dir", "owner", "roles"})
_MEMBER_KEYS = frozenset({"name", "kit", "voice"})
_BINDING_KEYS = frozenset({"schema_version", "series_id", "cast_digest", "characters"})
_BINDING_CHAR_KEYS = frozenset({"kit", "owner"})
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


class CastConfigError(ValueError):
    """A series cast configuration or saved cast binding is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class CastMember:
    """One resolved series character binding."""

    name: str
    kit: str
    voice: str = ""
    owner: bool = False


@dataclass(frozen=True, slots=True)
class SeriesCastConfig:
    """Validated opt-in series cast definition."""

    schema_version: int
    series_id: str
    owner: CastMember
    roles: Mapping[str, CastMember]
    assets_dir: Path | None
    cast_digest: str

    def all_display_names(self) -> tuple[str, ...]:
        names = [self.owner.name]
        for role_key in sorted(self.roles):
            names.append(self.roles[role_key].name)
        return tuple(names)

    def member_for_role(self, role_key: str) -> CastMember | None:
        if role_key == "owner":
            return self.owner
        return self.roles.get(role_key)

    def member_by_display_name(self, name: str) -> CastMember | None:
        target = normalize_display_name(name)
        if not target:
            return None
        if normalize_display_name(self.owner.name) == target:
            return self.owner
        for member in self.roles.values():
            if normalize_display_name(member.name) == target:
                return member
        return None

    def summary_entries(self) -> list[dict[str, Any]]:
        """Shareable cast summary: display names and kit slugs only, never raw role keys."""
        entries = [{"character": self.owner.name, "kit": self.owner.kit, "owner": True}]
        for role_key in sorted(self.roles):
            member = self.roles[role_key]
            entries.append({"character": member.name, "kit": member.kit, "owner": False})
        return entries


def normalize_display_name(name: str) -> str:
    """Collapse whitespace and case-fold so near-identical display names collide cleanly."""
    return " ".join(str(name or "").split()).casefold()


def _validate_display_name(raw: Any, label: str) -> str:
    if not isinstance(raw, str):
        raise CastConfigError(f"{label} name must be a string")
    cleaned = " ".join(raw.split())
    if not cleaned:
        raise CastConfigError(f"{label} name must not be empty")
    if _CONTROL_CHARS.search(cleaned):
        raise CastConfigError(f"{label} name contains invalid control characters")
    if "[REDACTED" in cleaned:
        raise CastConfigError(f"{label} name must not contain a redaction marker")
    if normalize_display_name(cleaned) in RESERVED_DISPLAY_NAMES:
        raise CastConfigError(f"{label} name {cleaned!r} is reserved and cannot be used as a character name")
    return cleaned


def _validate_kit_slug(raw: Any, label: str) -> str:
    if not isinstance(raw, str):
        raise CastConfigError(f"{label} kit must be a string")
    slug = raw.strip()
    if not slug or "/" in slug or "\\" in slug or ".." in slug:
        raise CastConfigError(f"{label} kit slug {raw!r} is invalid")
    return slug


def _parse_member(raw: Any, label: str, *, owner: bool) -> CastMember:
    if not isinstance(raw, Mapping):
        raise CastConfigError(f"{label} must be a JSON object")
    unknown = sorted(set(raw.keys()) - _MEMBER_KEYS)
    if unknown:
        raise CastConfigError(f"{label} has unknown key(s): {', '.join(unknown)}")
    name = _validate_display_name(raw.get("name"), label)
    kit = _validate_kit_slug(raw.get("kit"), label)
    voice_raw = raw.get("voice", "")
    if not isinstance(voice_raw, str):
        raise CastConfigError(f"{label} voice must be a string")
    if _CONTROL_CHARS.search(voice_raw):
        raise CastConfigError(f"{label} voice contains invalid control characters")
    return CastMember(name=name, kit=kit, voice=" ".join(voice_raw.split()), owner=owner)


def _compute_cast_digest(series_id: str, owner: CastMember, roles: Mapping[str, CastMember]) -> str:
    canonical = {
        "schema_version": CAST_CONFIG_SCHEMA_VERSION,
        "series_id": series_id,
        "owner": {"name": owner.name, "kit": owner.kit, "voice": owner.voice},
        "roles": {
            role_key: {"name": member.name, "kit": member.kit, "voice": member.voice}
            for role_key, member in sorted(roles.items())
        },
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def parse_cast_config(data: Mapping[str, Any], *, base_dir: Path | None = None) -> SeriesCastConfig:
    """Parse and structurally validate a ``cast.json`` mapping."""
    if not isinstance(data, Mapping):
        raise CastConfigError("cast configuration must be a JSON object")
    unknown = sorted(set(data.keys()) - _TOP_LEVEL_KEYS)
    if unknown:
        raise CastConfigError(f"cast configuration has unknown key(s): {', '.join(unknown)}")

    version = data.get("schema_version")
    if type(version) is not int or version != CAST_CONFIG_SCHEMA_VERSION:
        raise CastConfigError(
            f"unsupported cast schema_version {version!r} (expected {CAST_CONFIG_SCHEMA_VERSION})"
        )

    series_id = data.get("series_id")
    if not isinstance(series_id, str) or not SERIES_ID_RE.fullmatch(series_id):
        raise CastConfigError(
            f"invalid series_id {series_id!r} (expected 1-64 chars of [a-z0-9._-], starting with alphanumeric)"
        )

    assets_dir: Path | None = None
    if "assets_dir" in data and data["assets_dir"] not in (None, ""):
        raw_assets = data["assets_dir"]
        if not isinstance(raw_assets, str):
            raise CastConfigError("assets_dir must be a string path")
        resolved = Path(raw_assets).expanduser()
        if not resolved.is_absolute() and base_dir is not None:
            resolved = (base_dir / resolved).resolve()
        assets_dir = resolved

    if "owner" not in data:
        raise CastConfigError("cast configuration requires an 'owner' assignment")
    owner = _parse_member(data["owner"], "owner", owner=True)

    raw_roles = data.get("roles")
    if not isinstance(raw_roles, Mapping):
        raise CastConfigError("cast configuration requires a 'roles' object")

    roles: dict[str, CastMember] = {}
    seen_names: dict[str, str] = {normalize_display_name(owner.name): "owner"}
    for role_key, raw_member in sorted(raw_roles.items()):
        if not isinstance(role_key, str):
            raise CastConfigError("role keys must be strings")
        if role_key == "owner":
            raise CastConfigError("'owner' belongs in top-level 'owner', not under 'roles'")
        if not ROLE_KEY_RE.fullmatch(role_key):
            raise CastConfigError(
                f"invalid role key {role_key!r} (expected 'hermes:<profile>' or 'harness:<name>')"
            )
        member = _parse_member(raw_member, f"roles[{role_key!r}]", owner=False)
        norm = normalize_display_name(member.name)
        if norm in seen_names:
            raise CastConfigError(
                f"duplicate character display name {member.name!r} across {seen_names[norm]!r} and {role_key!r}"
            )
        seen_names[norm] = role_key
        roles[role_key] = member

    digest = _compute_cast_digest(series_id, owner, roles)
    return SeriesCastConfig(
        schema_version=CAST_CONFIG_SCHEMA_VERSION,
        series_id=series_id,
        owner=owner,
        roles=roles,
        assets_dir=assets_dir,
        cast_digest=digest,
    )


def load_cast_config(path: Path | str) -> SeriesCastConfig:
    """Load and validate ``cast.json`` from disk."""
    target = Path(path).expanduser()
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CastConfigError(f"cannot read cast config {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise CastConfigError(f"invalid JSON in cast config {target}: {exc}") from exc
    return parse_cast_config(raw, base_dir=target.resolve().parent)


def validate_cast_kits(
    config: SeriesCastConfig,
    kits: Mapping[str, CharacterKit],
    *,
    user_kits_dir: Path | None = None,
) -> None:
    """Ensure every kit slug named in ``config`` exists as a usable (non-draft) kit."""
    members: list[tuple[str, CastMember]] = [("owner", config.owner)] + [
        (role_key, member) for role_key, member in sorted(config.roles.items())
    ]
    missing: list[str] = []
    for role_label, member in members:
        if member.kit in kits:
            continue
        draft_manifest = (user_kits_dir / member.kit / "kit.json") if user_kits_dir is not None else None
        if draft_manifest is not None and draft_manifest.is_file():
            missing.append(
                f"{role_label} ({member.name}) references draft kit {member.kit!r} with no sheet images yet"
            )
        else:
            missing.append(f"{role_label} ({member.name}) references unknown kit {member.kit!r}")
    if missing:
        raise CastConfigError("; ".join(missing))


def event_agent_role_key(event: Any) -> str:
    """Canonical internal role key for the non-owner agent associated with an event."""
    payload = getattr(event, "payload", None) or {}
    harness = str(payload.get("harness") or "").strip()
    if harness and harness != "hermes":
        return f"harness:{harness}"
    profile = str(getattr(event, "profile", "") or "").strip()
    if profile:
        return f"hermes:{profile}"
    return ""


def event_role_key(event: Any) -> str:
    """Canonical internal role key for the primary actor of an event."""
    payload = getattr(event, "payload", None) or {}
    role = str(payload.get("role") or "").strip().lower()
    actor = str(getattr(event, "actor", "") or "").strip()
    if role == "user" or actor.lower() in {"owner", "user"}:
        return "owner"
    if str(getattr(event, "source", "") or "").strip() == "kanban" and actor:
        if actor.lower() in {"system", "cron", "kanban"}:
            return event_agent_role_key(event)
        return actor if ":" in actor and ROLE_KEY_RE.fullmatch(actor) else f"hermes:{actor}"
    return event_agent_role_key(event)


def opaque_role_id(role_key: str) -> str:
    """Privacy-preserving placeholder label for an unmapped role."""
    digest = hashlib.sha256(role_key.encode("utf-8")).hexdigest()[:8]
    return f"{UNMAPPED_ROLE_PREFIX}{digest}"


def is_unmapped_label(label: str) -> bool:
    return str(label or "").startswith(UNMAPPED_ROLE_PREFIX)


def sanitize_profile_value(
    raw_value: str,
    config: SeriesCastConfig,
    *,
    default_namespace: str = "hermes",
) -> str:
    """Convert a raw profile or actor identifier into a safe display name or opaque label."""
    value = str(raw_value or "").strip()
    if not value:
        return ""
    if value == "owner":
        return config.owner.name
    role_key = value if ":" in value and ROLE_KEY_RE.fullmatch(value) else f"{default_namespace}:{value}"
    member = config.member_for_role(role_key)
    if member is not None:
        return member.name
    return opaque_role_id(role_key)


def build_cast_binding(config: SeriesCastConfig, display_names: Iterable[str]) -> dict[str, Any]:
    """Build the shareable episode ``cast_binding`` receipt with display names -> kit slugs only."""
    characters: dict[str, dict[str, Any]] = {}
    for name in display_names:
        member = config.member_by_display_name(name)
        if member is None:
            raise CastConfigError(f"episode character {name!r} has no mapping in series cast {config.series_id!r}")
        characters[member.name] = {"kit": member.kit, "owner": member.owner}
    return {
        "schema_version": CAST_BINDING_SCHEMA_VERSION,
        "series_id": config.series_id,
        "cast_digest": config.cast_digest,
        "characters": characters,
    }


def parse_cast_binding(raw: Any) -> dict[str, Any]:
    """Validate an episode ``cast_binding`` block loaded from a saved Script IR."""
    if not isinstance(raw, Mapping):
        raise CastConfigError("saved cast_binding must be a JSON object")
    unknown = sorted(set(raw.keys()) - _BINDING_KEYS)
    if unknown:
        raise CastConfigError(f"saved cast_binding has unknown key(s): {', '.join(unknown)}")

    version = raw.get("schema_version")
    if type(version) is not int or version != CAST_BINDING_SCHEMA_VERSION:
        raise CastConfigError(
            f"unsupported cast_binding schema_version {version!r} (expected {CAST_BINDING_SCHEMA_VERSION})"
        )

    series_id = raw.get("series_id")
    if not isinstance(series_id, str) or not SERIES_ID_RE.fullmatch(series_id):
        raise CastConfigError(f"invalid cast_binding series_id {series_id!r}")

    cast_digest = raw.get("cast_digest")
    if not isinstance(cast_digest, str) or not cast_digest.strip():
        raise CastConfigError("saved cast_binding is missing cast_digest")

    raw_chars = raw.get("characters")
    if not isinstance(raw_chars, Mapping) or not raw_chars:
        raise CastConfigError("saved cast_binding requires a non-empty 'characters' mapping")

    characters: dict[str, dict[str, Any]] = {}
    for name, entry in raw_chars.items():
        if not isinstance(name, str) or ROLE_KEY_RE.fullmatch(name):
            raise CastConfigError(f"invalid character key in cast_binding: {name!r}")
        valid_name = _validate_display_name(name, "cast_binding character")
        if not isinstance(entry, Mapping):
            raise CastConfigError(f"cast_binding entry for {valid_name!r} must be a JSON object")
        extra = sorted(set(entry.keys()) - _BINDING_CHAR_KEYS)
        if extra:
            raise CastConfigError(f"cast_binding entry for {valid_name!r} has unknown key(s): {', '.join(extra)}")
        kit_slug = _validate_kit_slug(entry.get("kit"), f"cast_binding[{valid_name!r}]")
        owner_flag = entry.get("owner")
        if type(owner_flag) is not bool:
            raise CastConfigError(f"cast_binding entry for {valid_name!r} requires boolean 'owner'")
        characters[valid_name] = {"kit": kit_slug, "owner": owner_flag}

    return {
        "schema_version": CAST_BINDING_SCHEMA_VERSION,
        "series_id": series_id,
        "cast_digest": cast_digest,
        "characters": characters,
    }


def resolve_voices(
    config: SeriesCastConfig,
    kits: Mapping[str, CharacterKit],
    default_voices: Mapping[str, str],
) -> dict[str, str]:
    """Resolve character voices keyed by display name (config > kit > defaults)."""
    voices = dict(default_voices)
    members = [config.owner] + [config.roles[key] for key in sorted(config.roles)]
    for member in members:
        kit = kits.get(member.kit)
        kit_voice = kit.voice if kit is not None else ""
        resolved = member.voice or kit_voice or voices.get(member.name, "")
        if resolved:
            voices[member.name] = resolved
    return voices


def resolve_bound_kits(
    raw_binding: Any,
    cast_names: Iterable[str],
    kits: Mapping[str, CharacterKit],
) -> tuple[dict[str, CharacterKit], list[str]]:
    """Resolve saved episode characters strictly to exact kit slugs (never by display name)."""
    try:
        binding = parse_cast_binding(raw_binding)
    except CastConfigError as exc:
        return {}, [f"invalid cast_binding: {exc}"]

    characters = binding["characters"]
    ordered_names = list(dict.fromkeys([*cast_names, *characters.keys()]))
    bound: dict[str, CharacterKit] = {}
    problems: list[str] = []
    for name in ordered_names:
        entry = characters.get(name)
        if entry is None:
            problems.append(f"character {name!r} has no entry in saved cast_binding")
            continue
        slug = entry["kit"]
        kit = kits.get(slug)
        if kit is None:
            problems.append(
                f"no character kit {slug!r} for {name!r} (add one with `hermes reel kit add`)"
            )
            continue
        bound[name] = kit
    return bound, problems
