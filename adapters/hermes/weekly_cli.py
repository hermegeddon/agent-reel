"""Native weekly launcher. Source only: no install, schedule or credential discovery."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def register_weekly_cli(subs: Any) -> None:
    weekly = subs.add_parser('weekly', help='Run one explicitly approved weekly occurrence locally')
    for name in ('root', 'cast-config', 'assets-dir', 'home', 'occurrence', 'since', 'until',
                 'model', 'writer-provider', 'writer-model'):
        weekly.add_argument('--' + name, required=True)
    # Host ignores --model; ComfyUI resolves workflow/URL from saved settings.
    # Do not offer either until their effective routing can be pinned explicitly.
    weekly.add_argument('--source', required=True, choices=('openai', 'gemini', 'fal'))


class InertStrictWriter:
    """Explicit OFFLINE test interface, NOT an installed Hermes capability.

    A test supplies a single-route callable. Nothing discovers this on ctx.llm;
    there is no CLI escape hatch, credential route, retry or fallback here.
    """
    def __init__(self, complete_strict: Any, provider: str, model: str):
        self.complete_strict, self.provider, self.model = complete_strict, provider, model

    def complete(self, **kwargs: Any) -> Any:
        return self.complete_strict(**kwargs, provider=self.provider, model=self.model)


def weekly_command(args: argparse.Namespace, *, ctx: Any,
                   _inert_writer: InertStrictWriter | None = None) -> int:
    from adapters.hermes.reel_cli import register_cli, reel_command
    from scripts.weekly_rehearsal import JOB, explicit_route, run_weekly
    from datetime import datetime, timezone

    # No transcript, cast names, provider response bodies, paths or exception strings
    # enter the one-line machine result. Detailed local plan output stays in the run.
    try:
        home = Path(args.home).expanduser().resolve()
        if not os.environ.get('HERMES_HOME') or Path(os.environ['HERMES_HOME']).resolve() != home:
            raise ValueError('home mismatch')
        if ctx is None or getattr(ctx, 'llm', None) is None or not callable(getattr(ctx, 'dispatch_tool', None)):
            raise ValueError('host context missing')
        if not all(explicit_route(x) for x in (args.writer_provider, args.writer_model, args.model)):
            raise ValueError('explicit route names required')
        # Installed PluginLlm overrides enter auxiliary fallback. Until a strict
        # host contract is separately implemented AND verified through dispatch,
        # production always refuses, before private inputs, claims or planning.
        if _inert_writer is None:
            print(json.dumps(dict(job=JOB, phase='refused', reason='strict_writer_route_unavailable'), sort_keys=True))
            return 2
        if (type(_inert_writer) is not InertStrictWriter or
                (_inert_writer.provider, _inert_writer.model) != (args.writer_provider, args.writer_model)):
            raise ValueError('inert route binding mismatch')
        occurrence = datetime.fromisoformat(args.occurrence.replace('Z', '+00:00'))
        if occurrence.utcoffset() is None:
            raise ValueError('occurrence requires timezone')
        occurrence = occurrence.astimezone(timezone.utc)
        run_key = occurrence.strftime('%G-W%V')
        bound = SimpleNamespace(llm=_inert_writer,
                                dispatch_tool=ctx.dispatch_tool)
        parser = argparse.ArgumentParser()
        register_cli(parser)

        def invoke(argv: list[str]) -> int:
            return reel_command(parser.parse_args(argv), ctx=bound)

        # Suppress model/prompt-bearing handler diagnostics on either stream. The
        # controller separately captures plan stdout for its exact no-story gate.
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            state = run_weekly(root=Path(args.root).expanduser(), run_key=run_key,
                               cast=Path(args.cast_config).expanduser(), assets=Path(args.assets_dir).expanduser(),
                               home=home, occurrence=occurrence.isoformat(), since=args.since, until=args.until,
                               source=args.source, model=args.model,
                               writer_provider=args.writer_provider, writer_model=args.writer_model, invoke=invoke)
        result = dict(job=JOB, run_key=run_key, phase=state['phase'])
        code = {'complete': 0, 'no_story': 0, 'busy': 3, 'approval_refused': 4}.get(state['phase'], 5)
    except Exception:
        result, code = dict(job=JOB, phase='refused'), 2
    print(json.dumps(result, sort_keys=True))
    return code
