# Opt-In Series Casting (`--cast-config`)

Series casting lets you explicitly bind your owner role and each participating agent profile or harness to a chosen character display name and exact character kit slug, while preserving legacy behavior whenever `--cast-config` is omitted.

## Quick start

1. Create a `cast.json` file (`schema_version: 1`) mapping `owner` and every participating `hermes:<profile>` or `harness:<name>` role key:

```json
{
  "schema_version": 1,
  "series_id": "main-series",
  "assets_dir": "../assets",
  "owner": {
    "name": "Rowan",
    "kit": "rowan-lead",
    "voice": "dry, practical, treats bugs like stuck clockwork"
  },
  "roles": {
    "hermes:default": {
      "name": "Kensei",
      "kit": "kensei"
    },
    "harness:claude-code": {
      "name": "Tally",
      "kit": "tally-standin"
    }
  }
}
```

2. Run `plan --dry-run` with `--cast-config`:

```sh
hermes reel plan --dry-run \
  --cast-config /path/to/cast.json \
  --assets-dir /path/to/assets \
  --data-dir /path/to/plan-out
```

3. Render an episode cleanly against its saved kit bindings:

```sh
hermes reel render --plan-dir /path/to/plan-out --episode 1 --style anime --assets-dir /path/to/assets
```

## Schema (`cast.json`, `schema_version: 1`)

- `schema_version`: exact integer `1`.
- `series_id`: `[a-z0-9][a-z0-9._-]{0,63}`; isolates both series continuity (`series-<series_id>.json`) and model cache directories (`cache/cast-<series_id>-<cast_digest>/`).
- `assets_dir` *(optional)*: root containing `kits/` and `approvals.json`. Relative paths resolve from the `cast.json` directory. **Precedence:** CLI `--assets-dir` > `cast.json` `assets_dir` > plugin default data dir.
- `owner`: `{ "name", "kit", "voice"? }`. Sets the owner character name, kit slug, and optional dialogue voice (config `voice` overrides kit `voice`).
- `roles`: mapping of `"hermes:<profile>"` or `"harness:<name>"` to `{ "name", "kit", "voice"? }`. `"owner"` cannot appear under `roles`.
- Display names across `owner` and `roles` must be unique (whitespace-collapsed and case-folded), cannot use reserved words (`assistant`, `default`, `agent`, `the fleet`, `user`, `owner`, `narrator`), and cannot trip deterministic redaction rules.
- Every referenced `kit` must exist on disk as a non-draft kit (containing at least one sheet image). Sheetless draft kits fail preflight before any model call.

## Participation, validation timing, and privacy

- **Preflight (`0` LLM calls):** Malformed `cast.json`, duplicate/unsafe display names, missing or draft kits, and series/cast digest mismatches in `series-<series_id>.json` fail immediately with exit code `2`.
- **Unmapped bystanders do not abort runs:** When `--all-profiles` or `kanban.db` encounters an unmapped bystander profile, the redaction gate replaces raw profile/actor/subject/memory-key identifiers with opaque labels (`unmapped-role-<sha256[:8]>`). Unmapped events are never sent to model redaction or story-writing prompts and never enter `canon.json` agent entities.
- **Unmapped participants stop before writing:** Before calling the story writer on a candidate episode, `reel` checks every participating role in that candidate thread. If any participant is missing from `roles`, that candidate episode is skipped before writing (never silently recast), recording an opaque skip reason in `metrics.json` and printing the actionable role key locally.
- **Shareable `cast_binding`:** Episode IR files (`script-ir-*.json`) and render outputs (`render-manifest.json`, `sidecar.json`) store only safe display names and exact kit slugs (`cast_binding`), never raw `hermes:<profile>` or `harness:<name>` role keys.

## Rendering exact bound kits

- When `script-ir-*.json` contains `cast_binding`, `hermes reel render` resolves every character strictly by its saved `kit` slug (ignoring later edits to `cast.json` and refusing display-name fallback matching).
- Missing, malformed, unapproved, or byte-changed sheets (SHA-256 mismatch in `approvals.json`), as well as unconsented `real_person` kits under `--public`, fail preflight before any image call.
- `PanelRequest.bindings` is hashed into `canon.render.request_key()` and saved in `render-manifest.json` (`bindings`).
- Episodes planned without `--cast-config` contain no `cast_binding` and continue using legacy name lookup unchanged.
