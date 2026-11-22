# Opt-In Series Casting (`--cast-config`)

Series casting lets you explicitly bind your owner role and each participating agent profile or harness to a chosen character display name and exact character kit slug, while preserving legacy behavior whenever `--cast-config` is omitted.

## Quick start (two-character pilot)

1. **Prerequisite — prepare and approve your lead kit first.** `kensei` is bundled in the repository (`kits/kensei`), whereas `maren-lead` does **not** exist on disk until you add or generate it. Copying the example `cast.json` before `maren-lead` has an image sheet on disk fails preflight intentionally with `unknown kit 'maren-lead'` (zero LLM calls). Note also that `hermes reel kit create --dry-run` saves a sheetless draft kit, which still fails preflight until an actual sheet file is added (`hermes reel kit add`) or generated:

```sh
hermes reel kit add maren-lead \
  --name Maren \
  --style anime \
  --sheet /path/to/maren-anime.png \
  --description "Compact signal cartographer in a cropped deep petrol-teal high-collar jacket with an angled matte-charcoal map tube on a signal-vermilion cross-body strap." \
  --licence "Original fictional character sheet" \
  --assets-dir /path/to/assets

hermes reel kit approve maren-lead --style anime --yes --assets-dir /path/to/assets
hermes reel kit approve kensei --style anime --yes --assets-dir /path/to/assets
```

*(Note: `hermes reel kit add` records `role: "cast"` in `kit.json` and writes no kit `voice`; `cast.json`'s `"owner"` entry is what casts Maren as narrative lead (`"lead": true`) and supplies her dialogue `"voice"`.)*

2. Create a working two-character `cast.json` (`schema_version: 1`) mapping `owner` and `hermes:default`. Only list roles whose non-draft kits already exist on disk (you can add more co-stars later once their kits are ready):

```json
{
  "schema_version": 1,
  "series_id": "main-series",
  "assets_dir": "../assets",
  "owner": {
    "name": "Maren",
    "kit": "maren-lead",
    "voice": "fast, inquisitive, tracks where reports and reality parted ways"
  },
  "roles": {
    "hermes:default": {
      "name": "Kensei",
      "kit": "kensei"
    }
  }
}
```

3. Run `plan --dry-run` with `--cast-config`:

```sh
hermes reel plan --dry-run \
  --cast-config /path/to/cast.json \
  --assets-dir /path/to/assets \
  --data-dir /path/to/plan-out
```

4. Render an episode against its saved kit bindings:

```sh
hermes reel render --plan-dir /path/to/plan-out --episode 1 --style anime --assets-dir /path/to/assets
```

## Schema (`cast.json`, `schema_version: 1`)

- `schema_version`: exact integer `1`.
- `series_id`: `[a-z0-9][a-z0-9._-]{0,63}`; isolates both series continuity (`series-<series_id>.json`) and model cache directories (`cache/cast-<series_id>-<cast_digest>/`).
- `assets_dir` *(optional)*: root containing `kits/` and `approvals.json`. Relative paths resolve from the `cast.json` directory. **Precedence:** CLI `--assets-dir` > `cast.json` `assets_dir` > plugin default data dir.
- `owner`: `{ "name", "kit", "voice"? }`. Sets the episode's narrative lead character name, kit slug, and optional dialogue voice (config `voice` overrides kit `voice`).
- `roles`: mapping of `"hermes:<profile>"` or `"harness:<name>"` to `{ "name", "kit", "voice"? }`. `"owner"` cannot appear under `roles`.
- Display names across `owner` and `roles` must be unique (whitespace-collapsed and case-folded), cannot use reserved words (`assistant`, `default`, `agent`, `the fleet`, `user`, `owner`, `narrator`), and cannot trip deterministic redaction rules.
- Every referenced `kit` in `cast.json` must exist on disk as a non-draft kit (containing at least one sheet image). Sheetless draft kits fail preflight before any model call.

## Adding a co-star later

Once you prepare a kit for another role (for example `harness:claude-code`), you can **add** that new role entry directly to your existing `cast.json` under the same `series_id`:

- Keeping all previously recorded character bindings unchanged and adding new co-stars is allowed seamlessly in `series-<series_id>.json`, and model outputs run under a fresh `cache/cast-<series_id>-<cast_digest>/` directory.
- Follow-on runs in the same `--data-dir` allocate episode numbers strictly past both `series.next_number` and every existing `script-ir-<N>.json` file on disk (`script-ir-2.json`, `episode-2-*`, rendered via `--episode 2`), even if an explicit episode number is repeated, so Chapter 1 (`script-ir-1.json`, rendered via `--episode 1`) stays intact with its original `cast_binding`.
- Recasting an existing character (`name`, `kit`, or lead assignment), removing an already-recorded role, mixing a different `series_id`, or running `plan` without `--cast-config` inside the same `--data-dir` fails preflight cleanly before any LLM call (`exit code 2`) and leaves existing episodes and ledgers intact; always use a **fresh `--data-dir`** (together with a new `series_id`) when recasting, switching series, or running uncasted.

## Participation, validation timing, and privacy

- **Preflight (`0` LLM calls):** Malformed `cast.json`, duplicate/unsafe display names, missing or draft kits referenced in `cast.json`, symmetric `--data-dir` collisions (casting into an uncasted directory, mixing series IDs, or omitting `--cast-config` inside an already cast-configured directory), and incompatible recasts/removals in `series-<series_id>.json` fail immediately with exit code `2` before any model call.
- **Unmapped roles elsewhere or unused in the same thread do not block:** When `--all-profiles` or `kanban.db` encounters unmapped bystander roles—even in the same thread as an eligible story—incidental courtesy replies, acknowledgements (including emoji variants such as `"Thanks for keeping me posted."` or `"Thanks for the update! ❤️"`), and pure non-factual celebratory closing chatter (such as `"Glad the bug is behind us. Thanks, everyone!"`) preserve a task's recorded result (`tasks.result` / `completed`) and stay out of story turns. Any factual post-completion report, question, caveat, or contradiction (such as `"The queue is frozen again."` or `"Hold release until retry check finishes."`) **fails closed**: it overrides the outcome for participation checks and remains tracked across any later mapped comments (such as meeting-notes follow-ups) so an unresolved unmapped report still stops that episode cleanly before writing. Non-relied-on unmapped events are isolated at the gate (`unmapped-role-<sha256[:8]>`), excluded from episode briefs/facts/beats and LLM payloads, and do not block planning.
- **Relied-on unmapped participants skip only that episode before writing (without crowding out eligible stories):** Before any writing-model call (`write_title`, `write_synopsis`, `plan_episode`), the pipeline determines which events the episode's brief (`goal`, `outcome`, `turns`), panel beats, and atomic facts rely on. If any relied-on event belongs to an unmapped role, only that episode is skipped (never silently recast), planning continues walking the full ranked candidate list until `--episodes` eligible episodes are planned or candidates are exhausted, `metrics.json` records an opaque `unmapped-role-...` skip reason, and `hermes reel plan` prints the actionable missing role key locally.
- **Shareable `cast_binding` & consent separation:** Episode IR files (`script-ir-*.json`) and render outputs (`render-manifest.json`, `sidecar.json`) store `cast_binding` with display names, exact `kit` slugs, and `"lead": true|false` (narrative framing only)—never raw `hermes:<profile>` or `harness:<name>` role keys. Narrative `"lead": true` never grants permission to depict a real person; likeness and public-render consent always come strictly from the bound kit's own `real_person`, `owner`, and `public_consent` flags plus approved sheet digests in `approvals.json`.

## Rendering exact bound kits

- When `script-ir-*.json` contains `cast_binding`, `hermes reel render` resolves every character strictly by its saved `kit` slug (ignoring later edits to `cast.json` and refusing display-name fallback matching).
- Missing, malformed, unapproved, or byte-changed sheets (SHA-256 mismatch in `approvals.json`), as well as unconsented `real_person` kits under `--public`, fail preflight before any image call.
- `PanelRequest.bindings` is hashed into `canon.render.request_key()` and saved in `render-manifest.json` (`bindings`).
- Episodes planned without `--cast-config` contain no `cast_binding` and continue using legacy name lookup unchanged.
