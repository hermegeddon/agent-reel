"""Synthetic native command seam; no installed plugin discovery or provider calls."""
import argparse
import json
from types import SimpleNamespace

import pytest

from adapters.hermes import reel_cli


def test_registered_weekly_entry_refuses_even_empty_history(weekly, monkeypatch, capsys):
    import sqlite3
    from scripts.weekly_rehearsal import JOB

    args, harness, _calls = weekly
    with sqlite3.connect(args['home'] / 'state.db') as conn:
        conn.execute('DELETE FROM messages')
        conn.execute('DELETE FROM sessions')
    # A genuine parser + handler path, not a controller invocation callback.
    reel_cli.bind(SimpleNamespace(llm=object(), dispatch_tool=harness.dispatch_tool))
    parser = argparse.ArgumentParser()
    reel_cli.register_cli(parser)
    parsed = parser.parse_args([
        'weekly', '--root', str(args['root']), '--cast-config', str(args['cast']),
        '--assets-dir', str(args['assets']), '--home', str(args['home']),
        '--occurrence', '2026-11-27T23:00:00Z',
        '--since', '2026-11-20', '--until', '2026-11-27',
        '--source', 'openai', '--model', 'synthetic-image',
        '--writer-provider', 'synthetic', '--writer-model', 'synthetic-writer',
    ])
    try:
        assert parsed.func(parsed) == 2
        result = json.loads(capsys.readouterr().out)
        assert result == {'job': JOB, 'phase': 'refused', 'reason': 'strict_writer_route_unavailable'}
        assert harness.llm_calls == harness.image_calls == []
        assert not args['root'].exists()
    finally:
        reel_cli.bind(None)


def _entry_args(args):
    parser = argparse.ArgumentParser()
    reel_cli.register_cli(parser)
    argv = ['weekly']
    for key, flag in [('root', 'root'), ('cast', 'cast-config'), ('assets', 'assets-dir'),
                      ('home', 'home'), ('occurrence', 'occurrence'), ('since', 'since'),
                      ('until', 'until'), ('source', 'source'), ('model', 'model'),
                      ('writer_provider', 'writer-provider'), ('writer_model', 'writer-model')]:
        argv += ['--' + flag, str(args[key])]
    return parser.parse_args(argv)


def _inert_entry(args, ctx):
    """Not a host capability: explicitly inject the offline single-route seam."""
    from adapters.hermes.weekly_cli import InertStrictWriter, weekly_command
    return weekly_command(args, ctx=ctx, _inert_writer=InertStrictWriter(
        getattr(ctx.llm, 'complete', lambda **kw: None), args.writer_provider, args.writer_model))


def test_native_entry_full_real_handlers_and_writer_adapter(weekly, monkeypatch, capsys):
    from canon import providers
    args, harness, _calls = weekly
    routes = []

    def complete(**kwargs):
        routes.append((kwargs['provider'], kwargs['model']))
        text = kwargs['messages'][1]['content']
        prompt, tail = text.split('\n\nInput JSON:\n', 1)
        payload, _, hint = tail.partition('\n\nResponse shape:\n')
        answer = harness.llm_complete_json(prompt=prompt, payload=json.loads(payload),
                                           schema_hint=json.loads(hint) if hint else None)
        return SimpleNamespace(text=json.dumps(answer), usage=None)

    original = providers.make_generator
    def generate(source, **kwargs):
        return original(source, **kwargs, transport=harness.transport, env={'OPENAI_API_KEY': 'synthetic'})
    monkeypatch.setattr(providers, 'make_generator', generate)
    reel_cli.bind(SimpleNamespace(llm=SimpleNamespace(complete=complete), dispatch_tool=harness.dispatch_tool))
    try:
        parsed = _entry_args(args)
        assert _inert_entry(parsed, reel_cli._CTX) == 0
        assert json.loads(capsys.readouterr().out)['phase'] == 'complete'
        assert 0 < len(harness.image_calls) <= 6
        assert routes and set(routes) == {('synthetic', 'synthetic-writer')}
        before = len(routes), len(harness.image_calls), len(harness.vision_calls)
        assert _inert_entry(parsed, reel_cli._CTX) == 0
        assert json.loads(capsys.readouterr().out)['phase'] == 'complete'
        assert before == (len(routes), len(harness.image_calls), len(harness.vision_calls))
    finally:
        reel_cli.bind(None)


@pytest.mark.parametrize('case,code,phase', [('busy', 3, 'busy'), ('home', 2, 'refused'),
                                          ('interruption', 2, 'refused')])
def test_entry_exit_and_diagnostic_redaction(weekly, monkeypatch, capsys, case, code, phase):
    args, harness, _calls = weekly
    if case == 'busy':
        (args['root'] / 'weekly-reel/2026-W48/claim').mkdir(parents=True)
    elif case == 'home':
        monkeypatch.setenv('HERMES_HOME', str(args['home'] / 'wrong'))
    else:
        def fail(*_args, **_kwargs):
            print('PRIVATE PROMPT DATA')
            raise RuntimeError('PRIVATE PROVIDER RESPONSE')
        monkeypatch.setattr(reel_cli, 'reel_command', fail)
    rc = _inert_entry(_entry_args(args), SimpleNamespace(llm=object(), dispatch_tool=harness.dispatch_tool))
    assert rc == code
    output = capsys.readouterr()
    assert json.loads(output.out)['phase'] == phase
    assert 'PRIVATE' not in output.out + output.err
    assert harness.image_calls == []


@pytest.mark.parametrize('changes', [dict(since='2026-11-21'), dict(until='2026-11-28'),
                                     dict(occurrence='2026-11-27T23:00:00'), dict(model=''),
                                     dict(source='host'), dict(since='20261120')])
def test_invalid_window_or_binding_never_invokes(weekly, changes):
    from scripts.weekly_rehearsal import run_weekly
    args, harness, calls = weekly
    with pytest.raises(ValueError):
        run_weekly(**dict(args, **changes))
    assert calls == []


# Share the synthetic fixture without broadening history or loading host tools.
@pytest.mark.parametrize('change', ['asset', 'model', 'writer_model', 'occurrence'])
def test_pending_receipt_binds_material_inputs(weekly, change):
    from scripts.weekly_rehearsal import run_weekly
    args, harness, calls = weekly
    (args['assets'] / 'approvals.json').unlink()
    assert run_weekly(**args)['phase'] == 'approval_refused'
    if change == 'asset':
        path = args['assets'] / 'kits/weekly-lead/kit.json'
        path.write_bytes(path.read_bytes() + b' ')
    elif change == 'occurrence':
        args[change] = '2026-11-27T22:00:00+00:00'
    else:
        args[change] += '-changed'
    with pytest.raises(ValueError, match='run inputs changed'):
        run_weekly(**args)
    assert len(calls) == 1
    assert harness.image_calls == []


@pytest.mark.parametrize('collision', ['home', 'assets', 'foreign-root'])
def test_weekly_state_cannot_overlap_other_state(weekly, collision):
    from scripts.weekly_rehearsal import run_weekly
    args, harness, calls = weekly
    if collision == 'foreign-root':
        args['root'].mkdir()
        (args['root'] / 'daily').mkdir()
    else:
        args['root'] = args[collision] / 'weekly'
    with pytest.raises(ValueError, match='weekly-only'):
        run_weekly(**args)
    assert calls == []


@pytest.mark.parametrize('surface', ['complete', 'complete_strict'])
def test_unverified_host_refuses_before_private_work(weekly, monkeypatch, capsys, surface):
    from scripts import weekly_rehearsal
    from adapters.hermes.weekly_cli import weekly_command
    args, harness, _calls = weekly
    entered = []
    def private_work(**kwargs):
        entered.append('private-work')
        return {'phase': 'no_story'}
    def failing_primary(**kwargs):
        entered.append('primary')
        raise TimeoutError('synthetic primary failure')
    monkeypatch.setattr(weekly_rehearsal, 'run_weekly', private_work)
    llm = SimpleNamespace(**{surface: failing_primary})
    rc = weekly_command(_entry_args(args), ctx=SimpleNamespace(llm=llm, dispatch_tool=harness.dispatch_tool))
    assert entered == []  # No private history/planning, primary or fallback route.
    assert not args['root'].exists()
    assert rc == 2
    assert json.loads(capsys.readouterr().out)['reason'] == 'strict_writer_route_unavailable'


@pytest.mark.parametrize('field', ['writer_provider', 'writer_model', 'model'])
@pytest.mark.parametrize('sentinel', ['auto', 'AUTO', 'default', 'inherit', 'inherited', 'main', 'none', ' '])
def test_route_sentinels_refuse_without_private_work(weekly, field, sentinel):
    from scripts.weekly_rehearsal import run_weekly
    args, harness, calls = weekly
    with pytest.raises(ValueError, match='explicit supported'):
        run_weekly(**dict(args, **{field: sentinel}))
    assert calls == []
    assert not args['root'].exists()


def test_inert_strict_primary_failure_does_not_route_again(weekly, capsys):
    from adapters.hermes.weekly_cli import InertStrictWriter, weekly_command
    args, harness, _calls = weekly
    routes = []
    def primary(**kwargs):
        routes.append(('primary', kwargs['provider'], kwargs['model']))
        raise TimeoutError('PRIVATE synthetic primary timeout')
    def fallback(**kwargs):
        routes.append(('forbidden-fallback',))
        raise AssertionError('must not route through ctx.llm')
    ctx = SimpleNamespace(llm=SimpleNamespace(complete=fallback), dispatch_tool=harness.dispatch_tool)
    rc = weekly_command(_entry_args(args), ctx=ctx, _inert_writer=InertStrictWriter(
        primary, args['writer_provider'], args['writer_model']))
    assert routes == [('primary', 'synthetic', 'synthetic-writer')]
    assert rc == 2
    assert 'PRIVATE' not in capsys.readouterr().out
    assert harness.image_calls == []


from test_weekly_rehearsal import weekly  # noqa: E402,F401
