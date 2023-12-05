import pytest
import ska_sun


@pytest.fixture()
def fast_sun_position_method(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ska_sun.conf, "sun_position_method_default", "fast")
