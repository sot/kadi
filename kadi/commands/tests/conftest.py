import pytest
import ska_sun

import kadi.commands as kc


@pytest.fixture()
def fast_sun_position_method(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ska_sun.conf, "sun_position_method_default", "fast")


@pytest.fixture()
def disable_hrc_scs107_commanding(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kc.conf, "disable_hrc_scs107_commanding", True)


@pytest.fixture(scope="module", autouse=True)
def cmds_dir(tmp_path_factory):
    with kc.conf.set_temp("cache_loads_in_astropy_cache", True):
        with kc.conf.set_temp("clean_loads_dir", False):
            cmds_dir = tmp_path_factory.mktemp("cmds_dir")
            with kc.conf.set_temp("commands_dir", str(cmds_dir)):
                yield
