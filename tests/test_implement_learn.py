import pytest
from omegaconf import OmegaConf

from secretagent import config
from secretagent.core import interface, _FACTORIES
from secretagent.implement.learnedcode import (
    _find_learned_path, _build_backoff_impl, LearnedCodeFactory,
)


@pytest.fixture(autouse=True)
def clean_config():
    """Reset config before and after each test."""
    saved = config.GLOBAL_CONFIG.copy()
    yield
    config.GLOBAL_CONFIG = saved


def _write_learned_py(workdir, interface_name, body):
    """Write a learned.py with a simple function definition."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / 'learned.py').write_text(
        f'def {interface_name}(*args, **kw):\n'
        f'    {body}\n'
    )


def _write_source_configs(workdir, interface_name, ptools_cfg, n=1):
    """Write n identical source config yamls."""
    cfg_dir = workdir / 'source_configs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cfg = OmegaConf.create({'ptools': {interface_name: ptools_cfg}})
        (cfg_dir / f'source_{i}.yaml').write_text(OmegaConf.to_yaml(cfg))


def _make_interface(name):
    """Create a minimal interface for testing."""
    def stub(x: str) -> str:
        ...
    stub.__name__ = name
    stub.__qualname__ = name
    return interface(stub)


# --- _find_learned_path tests ---


def test_find_learned_path(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    _write_learned_py(tmp_path / '20260101.120000.my_func__rote', 'my_func', 'return "a"')
    path = _find_learned_path('my_func', 'rote')
    assert path.name == 'learned.py'
    assert 'my_func__rote' in str(path.parent.name)


def test_find_learned_path_picks_most_recent(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    _write_learned_py(tmp_path / '20260101.120000.my_func__rote', 'my_func', 'return "old"')
    _write_learned_py(tmp_path / '20260201.120000.my_func__rote', 'my_func', 'return "new"')
    path = _find_learned_path('my_func', 'rote')
    assert '20260201' in str(path)


def test_find_learned_path_no_match(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    with pytest.raises(FileNotFoundError):
        _find_learned_path('no_such_func', 'rote')


# --- build_fn tests (no backoff) ---


def test_build_fn_loads_function(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    _write_learned_py(tmp_path / '20260101.120000.my_func__rote', 'my_func', 'return "hello"')
    iface = _make_interface('my_func')
    factory = LearnedCodeFactory()
    fn = factory.build_fn(iface, learner='rote')
    assert fn('anything') == 'hello'


def test_build_fn_missing_function_name(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    _write_learned_py(tmp_path / '20260101.120000.my_func__rote', 'wrong_name', 'return "hello"')
    iface = _make_interface('my_func')
    factory = LearnedCodeFactory()
    with pytest.raises(AttributeError, match='my_func'):
        factory.build_fn(iface, learner='rote')


def test_implement_via_learned(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    _write_learned_py(tmp_path / '20260101.120000.my_func2__rote', 'my_func2', 'return "impl"')
    iface = _make_interface('my_func2')
    iface.implement_via('learned_code', learner='rote')
    assert iface('x') == 'impl'


# --- backoff tests ---


def test_backoff_returns_learned_when_not_none(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    workdir = tmp_path / '20260101.120000.my_func3__rote'
    _write_learned_py(workdir, 'my_func3', 'return "learned"')
    _write_source_configs(workdir, 'my_func3', {'method': 'direct'})
    iface = _make_interface('my_func3')
    factory = LearnedCodeFactory()
    fn = factory.build_fn(iface, learner='rote', backoff=True)
    assert fn('x') == 'learned'


def test_backoff_falls_back_when_none(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    workdir = tmp_path / '20260101.120000.my_func4__rote'
    _write_learned_py(workdir, 'my_func4', 'return None')
    _write_source_configs(workdir, 'my_func4', {'method': 'direct'})
    iface = _make_interface('my_func4')
    factory = LearnedCodeFactory()
    fn = factory.build_fn(iface, learner='rote', backoff=True)
    # direct factory uses the stub body, which returns None (via ...)
    # so the backoff also returns None — but it exercises the path
    result = fn('x')
    # The stub returns None too, so result is None — the key check is no error
    assert result is None


def test_backoff_uses_direct_fn(tmp_path):
    """Backoff with direct factory pointing to a real function."""
    config.configure(learn=dict(train_dir=str(tmp_path)))
    workdir = tmp_path / '20260101.120000.my_func5__rote'
    _write_learned_py(workdir, 'my_func5', 'return None')
    _write_source_configs(workdir, 'my_func5',
                          {'method': 'direct', 'fn': 'json.loads'})
    iface = _make_interface('my_func5')
    factory = LearnedCodeFactory()
    fn = factory.build_fn(iface, learner='rote', backoff=True)
    assert fn('{"a": 1}') == {'a': 1}


def test_backoff_no_fallback_when_learned_has_answer(tmp_path):
    """When learned returns a value, backoff fn is not called."""
    config.configure(learn=dict(train_dir=str(tmp_path)))
    workdir = tmp_path / '20260101.120000.my_func6__rote'
    _write_learned_py(workdir, 'my_func6', 'return "got_it"')
    _write_source_configs(workdir, 'my_func6',
                          {'method': 'direct', 'fn': 'json.loads'})
    iface = _make_interface('my_func6')
    factory = LearnedCodeFactory()
    fn = factory.build_fn(iface, learner='rote', backoff=True)
    # Should return learned result, not json.loads result
    assert fn('hello') == 'got_it'


# --- _build_backoff_impl error cases ---


def test_backoff_no_source_configs(tmp_path):
    config.configure(learn=dict(train_dir=str(tmp_path)))
    workdir = tmp_path / '20260101.120000.my_func7__rote'
    _write_learned_py(workdir, 'my_func7', 'return None')
    iface = _make_interface('my_func7')
    with pytest.raises(FileNotFoundError, match='source config'):
        _build_backoff_impl(iface, workdir)


def test_backoff_missing_interface_in_yaml(tmp_path):
    workdir = tmp_path / 'workdir'
    workdir.mkdir()
    cfg_dir = workdir / 'source_configs'
    cfg_dir.mkdir()
    cfg = OmegaConf.create({'ptools': {'other_func': {'method': 'direct'}}})
    (cfg_dir / 'source_0.yaml').write_text(OmegaConf.to_yaml(cfg))
    iface = _make_interface('my_func8')
    with pytest.raises(ValueError, match='no ptools.my_func8'):
        _build_backoff_impl(iface, workdir)


def test_backoff_conflicting_configs(tmp_path):
    workdir = tmp_path / 'workdir'
    workdir.mkdir()
    cfg_dir = workdir / 'source_configs'
    cfg_dir.mkdir()
    cfg1 = OmegaConf.create({'ptools': {'my_func9': {'method': 'direct'}}})
    cfg2 = OmegaConf.create({'ptools': {'my_func9': {'method': 'simulate'}}})
    (cfg_dir / 'source_0.yaml').write_text(OmegaConf.to_yaml(cfg1))
    (cfg_dir / 'source_1.yaml').write_text(OmegaConf.to_yaml(cfg2))
    iface = _make_interface('my_func9')
    with pytest.raises(ValueError, match='conflicting'):
        _build_backoff_impl(iface, workdir)


def test_backoff_consistent_configs_ok(tmp_path):
    """Multiple source configs that agree should not raise."""
    workdir = tmp_path / 'workdir'
    workdir.mkdir()
    _write_source_configs(workdir, 'my_func10', {'method': 'direct'}, n=3)
    iface = _make_interface('my_func10')
    impl = _build_backoff_impl(iface, workdir)
    assert impl.implementing_fn is not None
