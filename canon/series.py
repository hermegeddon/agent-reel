"""The series ledger: chapter numbering, continuity, and never telling the same story twice.

One file per user, next to the plan data. It records what has been told (thread, title,
arc), which frame each chapter used (world/villain/genre...), and whether the last
chapter left a thread open. The next run reads it to:

* put the open thread first, so the series follows on;
* refuse a story the series has already told, or a near-duplicate of one;
* give the idea deck a list of looks to avoid.

Nothing here invents facts: it only decides which true story is told next and how it is
framed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

SERIES_FILE = "series.json"
#: Two titles sharing this much vocabulary are treated as the same story retold.
DUPLICATE_TITLE_OVERLAP = 0.6
#: How many past chapters the deck avoids repeating a key from.
AVOID_WINDOW = 3

#: Placeholder titles carry no identity: keying on them would make every story look
#: like the first one, and the ledger would refuse perfectly good chapters.
GENERIC_TITLES = {"", "a session", "a task", "untitled story", "a story"}

_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "from", "that",
    "this", "its", "is", "was", "were", "be", "as", "at", "by", "it", "not", "no",
}


def _words(text: str) -> set[str]:
    return {
        word[:8]
        for word in re.findall(r"[a-z0-9]{3,}", (text or "").lower())
        if word not in _STOP
    }


def title_overlap(a: str, b: str) -> float:
    """Jaccard overlap of two titles' content words (0..1)."""
    left, right = _words(a), _words(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def is_generic_title(title: str) -> bool:
    return " ".join((title or "").lower().split()) in GENERIC_TITLES


def story_key(title: str, thread_key: str = "") -> str:
    """Stable fingerprint of a story, for the ledger.

    A generic placeholder title ("A session") identifies nothing, so it contributes no
    key: the thread key alone decides repeats. Otherwise the first placeholder chapter
    would make every later chapter look like a repeat of it.
    """
    if is_generic_title(title):
        return ""
    slug = "-".join(sorted(_words(title)))
    if not slug:
        return ""
    return f"{thread_key}:{slug}" if thread_key else slug


@dataclass
class Series:
    """Everything the tool remembers about one user's run of chapters."""

    chapters: list[dict[str, Any]] = field(default_factory=list)
    series_id: str = ""
    cast_digest: str = ""
    cast_roles: dict[str, dict[str, Any]] = field(default_factory=dict)

    # ---- reading -----------------------------------------------------------
    @property
    def next_number(self) -> int:
        return max((int(ch.get("number", 0)) for ch in self.chapters), default=0) + 1

    @property
    def open_thread(self) -> str:
        """The thread the last chapter left hanging, if any."""
        for chapter in reversed(self.chapters):
            if chapter.get("open_thread"):
                return str(chapter["open_thread"])
            if chapter.get("closed"):
                return ""
        return ""

    @property
    def open_title(self) -> str:
        for chapter in reversed(self.chapters):
            if chapter.get("open_thread"):
                return str(chapter.get("title", ""))
        return ""

    def told_threads(self) -> set[str]:
        return {str(ch.get("thread_key", "")) for ch in self.chapters if ch.get("thread_key")}

    def told_keys(self) -> set[str]:
        return {str(ch.get("story_key", "")) for ch in self.chapters if ch.get("story_key")}

    def told_titles(self) -> list[str]:
        return [str(ch.get("title", "")) for ch in self.chapters if ch.get("title")]

    def already_told(self, title: str, thread_key: str = "") -> str:
        """Why this story must not be told again, or '' when it is fresh."""
        key = story_key(title, thread_key)
        if key and key in self.told_keys():
            return "the series has already told this story"
        if thread_key and thread_key in self.told_threads():
            return "the series has already told this thread"
        if not is_generic_title(title):
            for told in self.told_titles():
                if title_overlap(title, told) >= DUPLICATE_TITLE_OVERLAP:
                    return f"too close to an earlier chapter ({told!r})"
        return ""

    def avoid(self, window: int = AVOID_WINDOW) -> dict[str, list[str]]:
        """Deck keys used recently, per section, so the next draw differs."""
        out: dict[str, list[str]] = {}
        for chapter in self.chapters[-window:]:
            frame = chapter.get("frame") or {}
            for section, value in frame.items():
                key = value.get("key") if isinstance(value, Mapping) else None
                if not key:
                    continue
                plural = f"{section}s"
                out.setdefault(plural, []).append(str(key))
        return out

    # ---- writing -----------------------------------------------------------
    def record(
        self,
        *,
        number: int,
        title: str,
        thread_key: str,
        arc_id: str,
        frame: Mapping[str, Any] | None = None,
        cliffhanger: bool = False,
        closed: bool | None = None,
        episodes_in_run: int = 1,
    ) -> dict[str, Any]:
        open_thread = str(thread_key) if cliffhanger else ""
        chapter = {
            "number": int(number),
            "title": str(title),
            "thread_key": str(thread_key),
            "story_key": story_key(title, thread_key),
            "arc_id": str(arc_id),
            "frame": {k: v for k, v in (frame or {}).items() if isinstance(v, Mapping)},
            "open_thread": open_thread,
            "closed": (not cliffhanger) if closed is None else bool(closed),
        }
        self.chapters.append(chapter)
        return chapter

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"version": 1, "chapters": self.chapters}
        if self.series_id:
            out["series_id"] = self.series_id
        if self.cast_digest:
            out["cast_digest"] = self.cast_digest
        if self.cast_roles:
            out["cast_roles"] = {
                str(k): dict(v) for k, v in sorted(self.cast_roles.items())
            }
        return out


def load(path: Path) -> Series:
    if not path.exists():
        return Series()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Series()
    if not isinstance(raw, Mapping):
        return Series()
    chapters = raw.get("chapters")
    raw_roles = raw.get("cast_roles")
    cast_roles = (
        {
            str(k): dict(v)
            for k, v in raw_roles.items()
            if isinstance(k, str) and isinstance(v, Mapping)
        }
        if isinstance(raw_roles, Mapping)
        else {}
    )
    return Series(
        [dict(ch) for ch in chapters or [] if isinstance(ch, Mapping)],
        series_id=str(raw.get("series_id", "")),
        cast_digest=str(raw.get("cast_digest", "")),
        cast_roles=cast_roles,
    )


def series_file_for(data_dir: Path, series_id: str = "") -> Path:
    """Keep cast-configured ledgers isolated from legacy ``series.json``."""
    return data_dir / (f"series-{series_id}.json" if series_id else SERIES_FILE)


def verify_series_compatibility(
    series: Series,
    *,
    series_id: str,
    cast_digest: str,
    cast_roles: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Reject silent recasting or removal of existing roles; allow adding new co-stars."""
    from canon.series_cast import CastConfigError

    if series.series_id and series.series_id != series_id:
        raise CastConfigError(
            f"series ledger belongs to {series.series_id!r}, cannot mix with {series_id!r}"
        )
    if not series.cast_digest and series.chapters:
        raise CastConfigError(
            f"series ledger already holds {len(series.chapters)} uncasted chapter(s); "
            "use a fresh --data-dir (and new series_id)"
        )
    if not series.cast_digest or series.cast_digest == cast_digest:
        return
    if series.cast_roles and cast_roles is not None:
        for slot, prev in series.cast_roles.items():
            cur = cast_roles.get(slot)
            if cur is None:
                raise CastConfigError(
                    f"existing role binding ({prev.get('name', slot)!r}) was removed for series {series_id!r}; "
                    "use a fresh --data-dir (and new series_id) instead of dropping existing cast members"
                )
            prev_sig = (prev.get("name"), prev.get("kit"), prev.get("lead"))
            cur_sig = (cur.get("name"), cur.get("kit"), cur.get("lead"))
            if cur_sig != prev_sig:
                raise CastConfigError(
                    f"cast binding changed for series {series_id!r} ({prev_sig!r} -> {cur_sig!r}); "
                    "use a fresh --data-dir (and new series_id) instead of silently recasting existing chapters"
                )
        return
    raise CastConfigError(
        f"cast revision changed for series {series_id!r} ({series.cast_digest} -> {cast_digest}); "
        "use a fresh --data-dir (and new series_id) instead of silently recasting existing chapters"
    )


def save(path: Path, series: Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(series.as_dict(), indent=1) + "\n", encoding="utf-8")


def order_for_series(
    candidates: Sequence[Any],
    series: Series,
    *,
    title_of: Callable[[Any], str] = lambda c: str(getattr(c, "title", "") or ""),
) -> tuple[list[Any], list[dict[str, str]]]:
    """Continuity first, then fresh stories only.

    The open thread leads so the next chapter follows on. Any candidate whose story the
    series already told - or a near-duplicate of one - is dropped, with the reason kept so
    a run can say why it skipped a story rather than silently reusing it.
    """
    ranked = list(candidates)
    open_thread = series.open_thread
    if open_thread:
        ranked.sort(key=lambda c: (str(getattr(c, "key", "")) != open_thread,))
    kept: list[Any] = []
    dropped: list[dict[str, str]] = []
    for candidate in ranked:
        key = str(getattr(candidate, "key", ""))
        reason = series.already_told(title_of(candidate), key)
        # The open thread is allowed one more visit even though its thread was told.
        if reason and key != open_thread:
            dropped.append({"key": key, "reason": reason})
            continue
        kept.append(candidate)
    return kept, dropped


def frame_summary(frame: Mapping[str, Any]) -> str:
    """A short human line for a run report."""
    return ", ".join(
        f"{name}={value.get('key')}" for name, value in frame.items() if isinstance(value, Mapping)
    )
