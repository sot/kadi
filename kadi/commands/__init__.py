# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging

logger = logging.getLogger(__name__)

from astropy.config import ConfigNamespace
from kadi.config import ConfigItem


class Conf(ConfigNamespace):
    """
    Configuration parameters for my subpackage.
    """
    default_lookback = ConfigItem(
        30,
        'Default lookback for previous approved loads (days).'
    )
    cache_loads_in_astropy_cache = ConfigItem(
        False,
        'Cache backstop downloads in the astropy cache. Should typically be False, '
        'but useful during development to avoid re-downloading backstops.'
    )
    cache_starcats = ConfigItem(
        True,
        'Cache star catalogs that are retrieved to a file to avoid repeating the '
        'slow process of identifying fid and stars in catalogs. The cache file is '
        'conf.commands_dir/starcats.db.'
    )
    clean_loads_dir = ConfigItem(
        True,
        'Clean backstop loads (like APR1421B.pkl.gz) in the loads directory that are '
        'older than the default lookback. Most users will want this to be True, but '
        'for development or if you always want a copy of the loads set to False.'
    )
    commands_dir = ConfigItem(
        '~/.kadi',
        'Directory where command loads and command events are stored after '
        'downloading from Google Sheets and OCCweb.'
    )
    commands_version = ConfigItem(
        '1',
        'Default version of kadi commands ("1" or "2").  Overridden by '
        'KADI_COMMANDS_VERSION environment variable.'
    )

    cmd_events_flight_id = ConfigItem(
        '19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ',
        'Google Sheet ID for command events (flight scenario).'
    )

    star_id_match_halfwidth = ConfigItem(
        1.5,
        'Half-width box size of star ID match for get_starcats() (arcsec).'
    )

    fid_id_match_halfwidth = ConfigItem(
        40,
        'Half-width box size of fid ID match for get_starcats() (arcsec).'
    )


# Create a configuration instance for the user
conf = Conf()


from .core import *  # noqa
from .commands import *  # noqa
