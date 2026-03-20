import importlib.util

import pytest

from secretagent.dataset import Dataset, Case
from secretagent.learn.baselines import RoteLearner


def _make_dataset(cases_data):
    """Create a Dataset from a list of (input_args, input_kw, expected_output) tuples."""
    cases = [
        Case(name=f'c{i}', input_args=args, input_kw=kw, expected_output=out)
        for i, (args, kw, out) in enumerate(cases_data)
    ]
    return Dataset(name='test', cases=cases)


def _load_learned(path):
    """Import the generated learned.py module."""
    spec = importlib.util.spec_from_file_location('learned', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- fit() tests ---


def test_fit_creates_file(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'my_func')
    assert outpath.exists()
    assert outpath.name == 'learned.py'


def test_fit_returns_most_common(tmp_path):
    ds = _make_dataset([
        (['hello'], None, 'world'),
        (['hello'], None, 'world'),
        (['hello'], None, 'earth'),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'my_func')
    mod = _load_learned(outpath)
    assert mod.my_func('hello') == 'world'


def test_fit_returns_none_for_unseen(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'my_func')
    mod = _load_learned(outpath)
    assert mod.my_func('unseen') is None


def test_fit_multiple_inputs(tmp_path):
    ds = _make_dataset([
        (['a'], None, '1'),
        (['b'], None, '2'),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'f')
    mod = _load_learned(outpath)
    assert mod.f('a') == '1'
    assert mod.f('b') == '2'


def test_fit_with_kwargs(tmp_path):
    ds = _make_dataset([
        ([], {'x': 1, 'y': 2}, 'ok'),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'f')
    mod = _load_learned(outpath)
    assert mod.f(x=1, y=2) == 'ok'
    assert mod.f(x=1, y=99) is None


def test_fit_with_mixed_args_and_kwargs(tmp_path):
    ds = _make_dataset([
        (['pos'], {'key': 'val'}, 'result'),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'f')
    mod = _load_learned(outpath)
    assert mod.f('pos', key='val') == 'result'


def test_fit_function_named_after_interface(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'classify')
    mod = _load_learned(outpath)
    assert hasattr(mod, 'classify')
    assert mod.classify('a') == 'x'


def test_fit_preserves_list_output(tmp_path):
    ds = _make_dataset([
        (['a'], None, ['x', 'y']),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'f')
    mod = _load_learned(outpath)
    assert mod.f('a') == ['x', 'y']


def test_fit_preserves_dict_output(tmp_path):
    ds = _make_dataset([
        (['a'], None, {'k': 'v'}),
    ])
    outdir = tmp_path / 'out'
    outdir.mkdir()
    outpath = RoteLearner(ds).fit(outdir, 'f')
    mod = _load_learned(outpath)
    assert mod.f('a') == {'k': 'v'}


# --- report() tests ---


def test_report_returns_string():
    ds = _make_dataset([
        (['a'], None, 'x'),
        (['a'], None, 'x'),
        (['b'], None, 'y'),
    ])
    result = RoteLearner(ds).report()
    assert isinstance(result, str)
    assert 'inputs' in result
    assert 'coverage' in result
