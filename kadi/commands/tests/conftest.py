import pytest
import ska_sun

import kadi.commands


@pytest.fixture()
def fast_sun_position_method(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ska_sun.conf, "sun_position_method_default", "fast")


@pytest.fixture()
def disable_hrc_scs107_commanding(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(kadi.commands.conf, "disable_hrc_scs107_commanding", True)
