# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Configuration for kadi.

See https://docs.astropy.org/en/stable/config/index.html#customizing-config-location-in-affiliated-packages
and https://github.com/astropy/astropy/issues/12960.
"""

from astropy import config
from astropy.config import ConfigNamespace


class ConfigItem(config.ConfigItem):
    rootname = "kadi"


class Conf(ConfigNamespace):
    """
    Configuration parameters for kadi.
    """

    default_lookback = ConfigItem(
        30.0, "Default lookback for previous approved loads (days)."
    )
    cache_loads_in_astropy_cache = ConfigItem(
        False,
        "Cache backstop downloads in the astropy cache. Should typically be False, "
        "but useful during development to avoid re-downloading backstops.",
    )
    cache_starcats = ConfigItem(
        True,
        "Cache star catalogs that are retrieved to a file to avoid repeating the "
        "slow process of identifying fid and stars in catalogs. The cache file is "
        "conf.commands_dir/starcats.db.",
    )
    clean_loads_dir = ConfigItem(
        True,
        "Clean backstop loads (like APR1421B.pkl.gz) in the loads directory that are "
        "older than the default lookback. Most users will want this to be True, but "
        "for development or if you always want a copy of the loads set to False.",
    )
    commands_dir = ConfigItem(
        "~/.kadi",
        "Directory where command loads and command events are stored after "
        "downloading from Google Sheets and OCCweb.",
    )

    cmd_events_flight_id = ConfigItem(
        "19d6XqBhWoFjC-z1lS1nM6wLE_zjr4GYB1lOvrEGCbKQ",
        "Google Sheet ID for command events ('flight' or default scenario).",
    )

    cmd_events_custom_id = ConfigItem(
        "11p7_WRfOzuOMwASRGTdv1gjF-Kc-vm6zN59ZcYC5Lzo",
        "Google Sheet ID of command events also include for 'custom' scenario).",
    )

    cmd_events_exclude_intervals_gid = ConfigItem(
        "1681877928",
        "Google Sheet gid for validation exclude intervals in command events",
    )

    star_id_match_halfwidth = ConfigItem(
        1.5, "Half-width box size of star ID match for get_starcats() (arcsec)."
    )

    fid_id_match_halfwidth = ConfigItem(
        40, "Half-width box size of fid ID match for get_starcats() (arcsec)."
    )

    include_in_work_command_events = ConfigItem(
        False, "Include In-work command events that are not yet approved."
    )

    date_start_agasc1p8 = ConfigItem(
        "2024:217:21:59:00",  # 2024-08-04, start of AUG0524B loads
        "Start date for using AGASC 1.8 catalog.",
    )

    disable_hrc_scs107_commanding = ConfigItem(
        False,
        "Disable HRC SCS-107 commanding from #344, strictly for regression testing "
        "of command generation prior to that patch.",
    )

    maneuver_nman_step_time = ConfigItem(
        300.0,
        (
            "Time step (sec) for NMAN maneuver generation in kadi states. "
            "Applied as the step_max kwarg of CxoTime.linspace so effective "
            "step size may be smaller."
        ),
    )

    maneuver_rasl_step_time = ConfigItem(
        120.0,
        (
            "Time step (sec) for RASL maneuver generation in kadi states. "
            "Applied as the step_max kwarg of CxoTime.linspace so effective "
            "step size may be smaller."
        ),
    )
    matching_block_size = ConfigItem(500, "Matching block size for command blocks.")


# Create a configuration instance for the user
conf = Conf()
