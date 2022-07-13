# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Configuration for kadi.

See https://docs.astropy.org/en/stable/config/index.html#customizing-config-location-in-affiliated-packages
and https://github.com/astropy/astropy/issues/12960.
"""  # noqa

from astropy import config


class ConfigItem(config.ConfigItem):
    rootname = "kadi"
