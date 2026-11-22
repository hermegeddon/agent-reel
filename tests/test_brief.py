"""R3-F2: a thread brief built from gated text, entailment-checked."""

from __future__ import annotations

from canon.brief import (
    CHAPTER_MAX_TURNS,
    MAX_TURNS,
    MIN_TURNS,
    _spread,
    build_brief,
    event_text,
    is_incidental_acknowledgement,
    is_non_factual_chatter,
    is_story_event,
    materially_changes_outcome,
)
from canon.events import (
    KIND_MESSAGE_ASSISTANT,
    KIND_MESSAGE_USER,
    KIND_TASK_COMMENT,
    KIND_TASK_CREATED,
    KIND_TASK_FINISHED,
    KIND_TASK_STATUS,
    build_event,
)
from canon.gate import seal
from canon.grounding import ClaimGate


class _NullCache:
    def get(self, _key):
        return None

    def set(self, _key, _value):
        return None


def message(session, kind, text, ts, actor="Sahil"):
    event = build_event(
        source="session", kind=kind, ts=ts, profile=actor, actor=actor, subject=session,
        payload={"text": text, "session_id": session},
        store="main", table="messages", pk=f"{session}-{ts}",
    )
    return seal(event, event.payload)


def session_events():
    return [
        message("s1", KIND_MESSAGE_USER, "Please build the read-only SQLite adapter for the reel", 1.0),
        message("s1", KIND_MESSAGE_ASSISTANT, "Starting on the adapter and the snapshot path.", 2.0, "default"),
        message("s1", KIND_MESSAGE_USER, "The sidecar files must not be created", 3.0),
        message("s1", KIND_MESSAGE_ASSISTANT, "Added the header check so WAL databases are copied first.", 4.0, "default"),
        message("s1", KIND_MESSAGE_USER, "Good, now prove it with strace", 5.0),
        message("s1", KIND_MESSAGE_ASSISTANT, "Built and tested the adapter; 42 tests pass and no sidecars appear.", 6.0, "default"),
    ]


def test_the_goal_is_the_first_thing_the_user_asked_for():
    brief = build_brief("s1", "session", session_events())
    assert brief.goal.startswith("Please build the read-only SQLite adapter")


def test_the_outcome_is_the_final_assistant_message():
    brief = build_brief("s1", "session", session_events())
    assert "42 tests pass" in brief.outcome


def test_the_brief_has_between_three_and_six_turns():
    brief = build_brief("s1", "session", session_events())
    assert MIN_TURNS <= len(brief.turns) <= MAX_TURNS


def test_turns_are_ordered_and_carry_an_actor():
    brief = build_brief("s1", "session", session_events())
    assert [turn.text for turn in brief.turns]
    assert all(turn.actor for turn in brief.turns)


def test_a_task_family_brief_uses_the_created_title_and_the_completion():
    created = build_event(
        source="kanban", kind=KIND_TASK_CREATED, ts=1.0, profile="kensei", actor="kensei",
        subject="t1", payload={"task_id": "t1", "title": "Build the SQLite read-only adapter"},
        store="main", table="tasks", pk="t1-c",
    )
    rows = [seal(created, created.payload)]
    for index, (kind_field, text) in enumerate(
        (("blocked", "the lockfile drifted on the adapter"), ("blocked", "the fixture was stale again"),
         ("completed", "octacon completed the adapter and the snapshot path")),
        start=2,
    ):
        row = build_event(
            source="kanban", kind=KIND_TASK_STATUS, ts=float(index), profile="octacon",
            actor="octacon", subject="t1", payload={"task_id": "t1", "kind": kind_field, "text": text},
            store="main", table="task_events", pk=f"t1-{kind_field}{index}",
        )
        rows.append(seal(row, row.payload))
    brief = build_brief("t1", "task", rows)
    assert brief.goal.startswith("Build the SQLite")
    assert "completed the adapter" in brief.outcome


def test_a_verbatim_brief_line_is_grounded_without_asking_the_model():
    """A brief line is quoted from the thread, so it cannot be unsupported."""

    calls = {"n": 0}

    class Counting:
        name = "counting"

        def complete_json(self, **_kwargs):
            calls["n"] += 1
            return {"supported": False}

    from canon.cache import ModelCache
    from canon.llm import CachedLLM

    gate = ClaimGate(CachedLLM(Counting(), ModelCache.__new__(ModelCache), stage="t"))
    gate.llm.cache = _NullCache()
    brief = build_brief("s1", "session", session_events(), gate=gate)
    assert brief.ok
    assert not brief.dropped
    assert calls["n"] == 0, "a quote must not cost a model call"


def test_event_text_prefers_the_text_field():
    rows = session_events()
    assert event_text(rows[0]).startswith("Please build")


def test_brief_is_not_ok_without_a_goal():
    rows = [message("s2", KIND_MESSAGE_ASSISTANT, "An answer with no question behind it.", 1.0, "default")]
    brief = build_brief("s2", "session", rows)
    assert not brief.ok


def test_the_outcome_is_the_message_that_reports_a_result():
    rows = [
        message("s1", KIND_MESSAGE_USER, "please benchmark the memory system properly", 1.0),
        message("s1", KIND_MESSAGE_ASSISTANT, "Starting on the harness and the scoring path", 2.0, "denji"),
        message("s1", KIND_MESSAGE_ASSISTANT, "Benchmark complete: 87.3 against 79.5 on the long set", 3.0, "denji"),
        message("s1", KIND_MESSAGE_ASSISTANT, "Go received.", 4.0, "denji"),
    ]
    brief = build_brief("s1", "session", rows)
    assert "87.3" in brief.outcome
    assert brief.outcome != "Go received."


def test_incidental_acknowledgement_classifier():
    for text in (
        "ack",
        "Thanks for the update.",
        "Thanks for the update! ❤️",
        "Thanks for keeping me posted.",
        "Got it, thanks!",
        "Got it, thanks for the update!",
        "Following along here.",
        "LGTM, thank you.",
        "Hi team, looks good!",
    ):
        assert is_incidental_acknowledgement(text), f"expected incidental: {text!r}"

    for substantive in (
        "Hold release until retry check finishes.",
        "Good, now prove it with strace",
        "Thanks for fixing the lock timeout.",
        "Verified queue tests stay green after restart.",
    ):
        assert not is_incidental_acknowledgement(substantive), f"expected substantive: {substantive!r}"


def test_incidental_task_comment_after_completion_does_not_become_outcome_or_turn():
    base_rows = [
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_CREATED, ts=1.0, profile="default", actor="owner",
                subject="t2", payload={"task_id": "t2", "title": "Fix queue lock guard cleanly"},
                store="kanban", table="tasks", pk="t2-c",
            ),
            {"task_id": "t2", "title": "Fix queue lock guard cleanly"},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=2.0, profile="default", actor="default",
                subject="t2", payload={"task_id": "t2", "text": "Checked lock guard timeout across workers."},
                store="kanban", table="task_comments", pk="t2-k1",
            ),
            {"task_id": "t2", "text": "Checked lock guard timeout across workers."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_FINISHED, ts=3.0, profile="default", actor="default",
                subject="t2", payload={"task_id": "t2", "text": "Fixed the queue lock cleanly."},
                store="kanban", table="tasks", pk="t2-f",
            ),
            {"task_id": "t2", "text": "Fixed the queue lock cleanly."},
        ),
    ]
    for trailing in (
        "Thanks for the update.",
        "Thanks for keeping me posted.",
        "Thanks for the update! ❤️",
        "Enjoy the rest of your evening!",
        "Glad the bug is behind us. Thanks, everyone!",
        "Great work, everyone! Have a great weekend!",
        "Thanks again, team!",
    ):
        assert is_non_factual_chatter(trailing), trailing
        assert not materially_changes_outcome(trailing), trailing
        comment = seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=4.0, profile="default", actor="bystander",
                subject="t2", payload={"task_id": "t2", "text": trailing},
                store="kanban", table="task_comments", pk="t2-k2",
            ),
            {"task_id": "t2", "text": trailing},
        )
        assert not is_story_event(comment), trailing
        brief = build_brief("t2", "task", base_rows + [comment])
        assert brief.outcome == "Fixed the queue lock cleanly."
        assert brief.outcome_event_id == base_rows[2].id
        assert all(turn.event_id != comment.id for turn in brief.turns)

    for material_comment in (
        "Hold release until retry check finishes.",
        "The queue is frozen again.",
        "Users see blank pages on checkout.",
        "Glad the bug is behind us, but the queue is frozen.",
        "Glad the bug is behind us \u2014 did staging pass?",
        "The bug is back.",
    ):
        assert not is_non_factual_chatter(material_comment), material_comment
        assert materially_changes_outcome(material_comment), material_comment
        blocker_ev = seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=4.0, profile="default", actor="reviewer",
                subject="t2", payload={"task_id": "t2", "text": material_comment},
                store="kanban", table="task_comments", pk=f"t2-k-{abs(hash(material_comment))}",
            ),
            {"task_id": "t2", "text": material_comment},
        )
        blocked_brief = build_brief("t2", "task", base_rows + [blocker_ev])
        assert blocked_brief.outcome == material_comment
        assert blocked_brief.outcome_event_id == blocker_ev.id
        assert blocker_ev.id in blocked_brief.post_completion_event_ids
        assert any(turn.event_id == blocker_ev.id for turn in blocked_brief.turns)


def test_later_meeting_notes_comment_leaves_post_completion_report_unresolved():
    # Six prior events fill MAX_TURNS=6 before any post-completion comments arrive.
    base_rows = [
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_CREATED, ts=1.0, profile="default", actor="owner",
                subject="t3", payload={"task_id": "t3", "title": "Fix queue lock guard cleanly"},
                store="kanban", table="tasks", pk="t3-c",
            ),
            {"task_id": "t3", "title": "Fix queue lock guard cleanly"},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=2.0, profile="default", actor="Kensei",
                subject="t3", payload={"task_id": "t3", "text": "Audited lock timeout across worker pool."},
                store="kanban", table="task_comments", pk="t3-k1",
            ),
            {"task_id": "t3", "text": "Audited lock timeout across worker pool."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=3.0, profile="default", actor="Maren",
                subject="t3", payload={"task_id": "t3", "text": "Traced stale bypass on queue handoff."},
                store="kanban", table="task_comments", pk="t3-k2",
            ),
            {"task_id": "t3", "text": "Traced stale bypass on queue handoff."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=4.0, profile="default", actor="Kensei",
                subject="t3", payload={"task_id": "t3", "text": "Patched queue guard threshold limit cleanly."},
                store="kanban", table="task_comments", pk="t3-k3",
            ),
            {"task_id": "t3", "text": "Patched queue guard threshold limit cleanly."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=5.0, profile="default", actor="Maren",
                subject="t3", payload={"task_id": "t3", "text": "Confirmed worker suite passed after restart."},
                store="kanban", table="task_comments", pk="t3-k4",
            ),
            {"task_id": "t3", "text": "Confirmed worker suite passed after restart."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_FINISHED, ts=6.0, profile="default", actor="Kensei",
                subject="t3", payload={"task_id": "t3", "text": "Fixed the queue lock cleanly."},
                store="kanban", table="tasks", pk="t3-f",
            ),
            {"task_id": "t3", "text": "Fixed the queue lock cleanly."},
        ),
    ]
    frozen_ev = seal(
        build_event(
            source="kanban", kind=KIND_TASK_COMMENT, ts=7.0,
            profile="unmapped-role-deadbeef", actor="unmapped-role-deadbeef",
            subject="t3", payload={"task_id": "t3", "text": "The queue is frozen again."},
            store="kanban", table="task_comments", pk="t3-k-unmapped",
        ),
        {"task_id": "t3", "text": "The queue is frozen again."},
    )
    blank_ev = seal(
        build_event(
            source="kanban", kind=KIND_TASK_COMMENT, ts=7.0,
            profile="unmapped-role-deadbeef", actor="unmapped-role-deadbeef",
            subject="t3", payload={"task_id": "t3", "text": "Users see blank pages on checkout."},
            store="kanban", table="task_comments", pk="t3-k-blank",
        ),
        {"task_id": "t3", "text": "Users see blank pages on checkout."},
    )
    relief_ev = seal(
        build_event(
            source="kanban", kind=KIND_TASK_COMMENT, ts=7.0,
            profile="unmapped-role-deadbeef", actor="unmapped-role-deadbeef",
            subject="t3", payload={"task_id": "t3", "text": "Glad the bug is behind us. Thanks, everyone!"},
            store="kanban", table="task_comments", pk="t3-k-relief",
        ),
        {"task_id": "t3", "text": "Glad the bug is behind us. Thanks, everyone!"},
    )
    meeting_notes = "Added the queue lock summary to Friday's meeting notes."
    retry_notes = "Meeting notes updated: review retry checklist next Monday."
    for mapped_actor in ("Kensei", "Maren"):
        notes_ev = seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=8.0,
                profile=mapped_actor, actor=mapped_actor,
                subject="t3", payload={"task_id": "t3", "text": meeting_notes},
                store="kanban", table="task_comments", pk=f"t3-notes-{mapped_actor}",
            ),
            {"task_id": "t3", "text": meeting_notes},
        )
        brief_a = build_brief("t3", "task", base_rows + [frozen_ev, notes_ev])
        assert brief_a.outcome == "The queue is frozen again."
        assert brief_a.outcome_event_id == frozen_ev.id
        assert frozen_ev.id in brief_a.post_completion_event_ids
        assert any(turn.event_id == frozen_ev.id for turn in brief_a.turns)

        brief_c = build_brief("t3", "task", base_rows + [blank_ev, notes_ev])
        assert brief_c.outcome == "Users see blank pages on checkout."
        assert brief_c.outcome_event_id == blank_ev.id
        assert blank_ev.id in brief_c.post_completion_event_ids
        assert any(turn.event_id == blank_ev.id for turn in brief_c.turns)

        brief_d = build_brief("t3", "task", base_rows + [relief_ev, notes_ev])
        assert brief_d.outcome == "Fixed the queue lock cleanly."
        assert relief_ev.id not in brief_d.post_completion_event_ids
        assert all(turn.event_id != relief_ev.id for turn in brief_d.turns)

    retry_notes_ev = seal(
        build_event(
            source="kanban", kind=KIND_TASK_COMMENT, ts=8.0,
            profile="Kensei", actor="Kensei",
            subject="t3", payload={"task_id": "t3", "text": retry_notes},
            store="kanban", table="task_comments", pk="t3-retry-notes",
        ),
        {"task_id": "t3", "text": retry_notes},
    )
    brief_b = build_brief("t3", "task", base_rows + [frozen_ev, retry_notes_ev])
    assert frozen_ev.id in brief_b.post_completion_event_ids
    assert any(turn.event_id == frozen_ev.id for turn in brief_b.turns)


def test_chapter_turn_selection_one_slot_remaining_retains_required_turns():
    ordinary_pairs = [
        (
            seal(
                build_event(
                    source="kanban", kind=KIND_TASK_COMMENT, ts=float(i + 1), profile="default", actor="Kensei",
                    subject="t-spread", payload={"task_id": "t-spread", "text": f"Ordinary step {i} completed cleanly."},
                    store="kanban", table="task_comments", pk=f"sp-{i}",
                ),
                {"task_id": "t-spread", "text": f"Ordinary step {i} completed cleanly."},
            ),
            f"Ordinary step {i} completed cleanly.",
        )
        for i in range(4)
    ]
    assert [text for _, text in _spread(ordinary_pairs, 1)] == ["Ordinary step 0 completed cleanly."]
    assert _spread(ordinary_pairs, 0) == []

    rows = [
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_CREATED, ts=1.0, profile="default", actor="owner",
                subject="t-ch", payload={"task_id": "t-ch", "title": "Harden chapter lock handling"},
                store="kanban", table="tasks", pk="ch-c",
            ),
            {"task_id": "t-ch", "title": "Harden chapter lock handling"},
        ),
    ]
    ordinary_texts = [
        "Audited initial lock timeout configuration.",
        "Traced stale queue bypass under load.",
        "Patched worker retry backoff limit.",
        "Confirmed staging queue stayed balanced.",
    ]
    for idx, msg in enumerate(ordinary_texts, start=2):
        rows.append(
            seal(
                build_event(
                    source="kanban", kind=KIND_TASK_COMMENT, ts=float(idx), profile="default", actor="Kensei",
                    subject="t-ch", payload={"task_id": "t-ch", "text": msg},
                    store="kanban", table="task_comments", pk=f"ch-ord-{idx}",
                ),
                {"task_id": "t-ch", "text": msg},
            )
        )
    rows.append(
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_FINISHED, ts=6.0, profile="default", actor="Kensei",
                subject="t-ch", payload={"task_id": "t-ch", "text": "Fixed the queue lock cleanly."},
                store="kanban", table="tasks", pk="ch-fin",
            ),
            {"task_id": "t-ch", "text": "Fixed the queue lock cleanly."},
        )
    )
    required_evs = []
    for idx in range(1, 14):
        msg = f"Follow-up verification #{idx:02d}: retry check reported pending detail."
        ev = seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=float(6 + idx), profile="default", actor="Kensei",
                subject="t-ch", payload={"task_id": "t-ch", "text": msg},
                store="kanban", table="task_comments", pk=f"ch-req-{idx}",
            ),
            {"task_id": "t-ch", "text": msg},
        )
        required_evs.append(ev)
        rows.append(ev)

    brief = build_brief("t-ch", "task", rows, max_turns=CHAPTER_MAX_TURNS)
    assert len(brief.post_completion_event_ids) == 13
    assert len(brief.turns) == CHAPTER_MAX_TURNS == 14
    selected_ids = {turn.event_id for turn in brief.turns}
    assert {ev.id for ev in required_evs} <= selected_ids
    assert len(selected_ids - {ev.id for ev in required_evs}) == 1


def test_clean_task_meeting_notes_non_material_for_mapped_and_unmapped():
    for note in (
        "Added the queue lock summary to Friday's meeting notes.",
        "Filed meeting notes for the queue lock rollout.",
        "The meeting notes were updated.",
    ):
        assert not materially_changes_outcome(note), note

    assert materially_changes_outcome("Added a new bug to Friday's meeting notes.")
    assert materially_changes_outcome("Noted blank pages on checkout in meeting notes.")

    base_rows = [
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_CREATED, ts=1.0, profile="default", actor="owner",
                subject="t4", payload={"task_id": "t4", "title": "Fix queue lock guard cleanly"},
                store="kanban", table="tasks", pk="t4-c",
            ),
            {"task_id": "t4", "title": "Fix queue lock guard cleanly"},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=2.0, profile="default", actor="Kensei",
                subject="t4", payload={"task_id": "t4", "text": "Audited lock timeout across worker pool."},
                store="kanban", table="task_comments", pk="t4-k1",
            ),
            {"task_id": "t4", "text": "Audited lock timeout across worker pool."},
        ),
        seal(
            build_event(
                source="kanban", kind=KIND_TASK_FINISHED, ts=3.0, profile="default", actor="Kensei",
                subject="t4", payload={"task_id": "t4", "text": "Fixed the queue lock cleanly."},
                store="kanban", table="tasks", pk="t4-f",
            ),
            {"task_id": "t4", "text": "Fixed the queue lock cleanly."},
        ),
    ]
    note_text = "Added the queue lock summary to Friday's meeting notes."
    for author in ("Kensei", "unmapped-role-deadbeef"):
        note_ev = seal(
            build_event(
                source="kanban", kind=KIND_TASK_COMMENT, ts=4.0, profile=author, actor=author,
                subject="t4", payload={"task_id": "t4", "text": note_text},
                store="kanban", table="task_comments", pk=f"t4-note-{author}",
            ),
            {"task_id": "t4", "text": note_text},
        )
        brief = build_brief("t4", "task", base_rows + [note_ev])
        assert brief.outcome == "Fixed the queue lock cleanly."
        assert brief.outcome_event_id == base_rows[2].id
        assert brief.post_completion_event_ids == ()
        assert all(turn.event_id != note_ev.id for turn in brief.turns)
