# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging

logger = logging.getLogger(__name__)

from kadi import ConfigNamespace, ConfigItem


class Conf(ConfigNamespace):
    """
    Configuration parameters for my subpackage.
    """
    default_lookback = ConfigItem(
        30,
        'Default lookback for previous approved loads (days).'
    )
    update_from_network = ConfigItem(
        True,
        'Use Google sheets and OCCweb to get the latest information about approved '
        'loads and command events like SCS-107. If set to False then the existing '
        'local files will be used.'
    )
    cache_loads_in_astropy_cache = ConfigItem(
        False,
        'Cache backstop downloads in the astropy cache. Should typically be False, '
        'but useful during development to avoid re-downloading backstops.'
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


# Create a configuration instance for the user
conf = Conf()


from .core import *  # noqa
from .commands import *  # noqa
