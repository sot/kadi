# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Update the events database"""

import argparse
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Type

import maude
import numpy as np
import ska_tdb
from chandra_time import DateTime
from cheta import fetch
from ska_helpers.run_info import log_run_info

from kadi import __version__  # noqa: F401

if TYPE_CHECKING:
    import kadi.events.models as kem

# Use the common kadi.events logger instead of __name__ because this is in kadi.scripts.
logger = logging.getLogger("kadi.events")


def get_opt_parser():
    parser = argparse.ArgumentParser(description="Update the events database")
    parser.add_argument(
        "--stop", default=DateTime().date, help="Processing stop date (default=NOW)"
    )
    parser.add_argument(
        "--start",
        default=None,
        help="Processing start date (loops by --loop-days until --stop date if set)",
    )
    parser.add_argument(
        "--delete-from-start",
        action="store_true",
        help="Delete events after --start and reset update time to --start",
    )
    parser.add_argument(
        "--loop-days",
        default=100,
        type=int,
        help="Number of days in interval for looping (default=100)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default=INFO)",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model class name (e.g. DsnComm) to process [match regex] (default = all)",
    )
    parser.add_argument(
        "--data-root", default=".", help="Root data directory (default='.')"
    )
    parser.add_argument(
        "--maude", action="store_true", help="Use MAUDE data source for telemetry"
    )
    parser.add_argument(
        "--maude-tlm-lookback",
        default=7,
        type=int,
        help=(
            "MAUDE telemetry lookback (days) (default=7). This overrides the base "
            "TlmEvent.lookback=21. Use a longer value for a very extended Safe Mode."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)

    return parser


def try4times(func, *arg, **kwarg):
    """
    Work around problems with sqlite3 database getting locked out from writing

    This is presumably due to read activity.  Not completely understood.

    This function will try to run ``func(*arg, **kwarg)`` a total of 4 times with an
    increasing sequence of wait times between tries.  It catches only a database
    locked error.
    """
    from django.db.utils import OperationalError

    for delay in 0, 5, 10, 60:
        if delay > 0:
            time.sleep(delay)

        try:
            func(*arg, **kwarg)
        except OperationalError as err:
            if "database is locked" in str(err):
                # Locked DB, issue informational warning
                logger.info(
                    "Warning: locked database, waiting {} seconds".format(delay)
                )
            else:
                # Something else so just re-raise
                raise
        else:
            # Success, jump out of loop
            break

    else:
        # After 4 tries bail out with an exception
        raise OperationalError("database is locked")


def delete_from_date(EventModel, start, set_update_date=True):
    from kadi.events import models

    date_start = DateTime(start).date
    cls_name = EventModel.__name__

    if set_update_date:
        update = models.Update.objects.get(name=cls_name)
        logger.info(
            "Updating {} date from {} to {}".format(cls_name, update.date, date_start)
        )
        update.date = date_start
        try4times(update.save)

    events = EventModel.objects.filter(start__gte=date_start)
    logger.info(
        "Deleting {} {} events after {}".format(events.count(), cls_name, date_start)
    )
    try4times(events.delete)


def update_event_model(EventModel, date_stop: str, maude: bool) -> None:
    """Update the event model in the database with events up through date_stop.

    For telemetry events, date_stop is the upper limit on possible events, but most
    commonly it is limited by available telemetry.

    For non-telemetry events, date_stop is a bit fuzzier since some (like DSNComm) can
    go into the future.

    Parameters
    ----------
    EventModel : class
        Event model class to update (e.g. models.DsnComm)
    date_stop : str
        Stop date for event update
    maude : bool
        Use MAUDE data source for telemetry
    """
    import django.db
    from django.core.exceptions import ObjectDoesNotExist

    from kadi.events import models

    date_stop = DateTime(date_stop)
    cls_name = EventModel.__name__

    # The Update model is used to keep track of last date_stop when the events were
    # successfully updated.
    try:
        # Last update for this event model
        update = models.Update.objects.get(name=cls_name)
    except ObjectDoesNotExist:
        logger.info("No previous update for {} found".format(cls_name))
        duration = EventModel.lookback
        update = models.Update(name=cls_name, date=date_stop.date)
        date_start = date_stop - EventModel.lookback
    else:
        # Duration between last update and current date_stop (usually now)
        duration = date_stop - DateTime(update.date)
        date_start = DateTime(update.date) - EventModel.lookback
        update.date = date_stop.date

        # Some events like LoadSegment or DsnComm might change in the database after
        # having been ingested.  Use lookback_delete (less than lookback) to
        # always remove events in that range and re-ingest.
        if duration > 0 and hasattr(EventModel, "lookback_delete"):
            delete_date = DateTime(update.date) - EventModel.lookback_delete
            delete_from_date(EventModel, delete_date, set_update_date=False)

    if duration <= 0:
        logger.info(
            f"Skipping {cls_name} events because update "
            f"duration={duration:.1f} days <= 0"
        )
        return

    # Some events like LoadSegment, DsnComm are defined into the future, so
    # modify date_stop accordingly.  Note that update.date is set to the
    # nominal date_stop (typically NOW), and this refers more to the last date
    # of processing rather than the actual last date in the archive.
    if hasattr(EventModel, "lookforward"):
        date_stop = date_stop + EventModel.lookforward

    logger.info(
        "Updating {} events from {} to {}".format(
            cls_name, date_start.date[:-4], date_stop.date[:-4]
        )
    )

    # Special handling of date_start and date_stop for MAUDE telemetry and Telemetry
    # events. For CXC telemetry use the legacy behavior of fetching all telemetry
    # through lookback, since it is fast and reliable.
    if maude and issubclass(EventModel, models.TlmEvent):
        date_start, date_stop = get_date_start_stop_for_maude(
            EventModel, date_start, date_stop
        )

    # Get events for this model from appropriate resources (telemetry, iFOT, web).  This
    # is returned as a list of dicts with key/val pairs corresponding to model fields.
    events_in_dates = EventModel.get_events(date_start, date_stop)

    # Determine which of the events is not already in the database and
    # put them in a list for saving.
    event_models, events = filter_events_to_add_to_database(
        EventModel, cls_name, events_in_dates
    )

    # Save the new events in an atomic fashion
    with django.db.transaction.atomic():
        for event, event_model in zip(events, event_models):
            try:
                # In order to catch an IntegrityError here and press on, need to
                # wrap this in atomic().  This was driven by bad data in iFOT, namely
                # duplicate PassPlans that point to the same DsnComm, which gives an
                # IntegrityError because those are related as one-to-one.
                with django.db.transaction.atomic():
                    save_event_to_database(cls_name, event, event_model, models)
            except django.db.utils.IntegrityError:
                import traceback

                logger.warning(f"WARNING: IntegrityError skipping {event_model}")
                logger.warning(f"Event dict:\n{event}")
                logger.warning(f"Traceback:\n{traceback.format_exc()}")
                continue

        # If processing got here with no exceptions then save the event update
        # information to database
        update.save()


def get_date_start_stop_for_maude(
    EventModel: Type["kem.TlmEvent"],
    date_start: DateTime,
    date_stop: DateTime,
) -> tuple[DateTime, DateTime]:
    """Get updated date_start and date_stop for MAUDE telemetry events.

    Fetching data from the MAUDE server is relatively slow, so we need to take care to
    not fetch more data than necessary. This function adjusts the start and stop dates
    for telemetry events accordingly. In addition, there can be a mix of definitive
    backorbit data and realtime data with a gap between them. We need to avoid that gap
    and the realtime data.

    Parameters
    ----------
    EventModel : class
        Telemetry Event model class to update (e.g. models.DsnComm)
    date_start : DateTime
        Start date for event update
    date_stop : DateTime
        Stop date for event update

    Returns
    -------
    date_start : DateTime
        Updated start date
    date_stop : DateTime
        Updated stop date
    """
    from django.core.exceptions import ObjectDoesNotExist

    name = EventModel.__name__

    # For getting telemetry to define the states, use 4 hours before the last event time
    # as the start date (as long as that is within the original date_start to
    # date_stop). This will get that last event and so avoid fetching unnecessary
    # telemetry.
    try:
        last_event = EventModel.objects.last()
    except ObjectDoesNotExist:
        pass
    else:
        if date_start.date < last_event.start < date_stop.date:
            date_start = DateTime(last_event.tstart - 4 * 3600)
            logger.info(
                f"Using {date_start.date} as start for {name} (4 hours before last event)"
            )

    # Get the last available backorbit telemetry date for MSIDs required for this
    # event model. If this is before stop_date then set stop_date accordingingly.
    # This handles the situation for MAUDE that the server may provide realtime data
    # during (or just after) a comm with a gap since the backorbit data from the
    # last comm. This would cause problems. Note that the COBSRQID obsid is
    # hardwired into every TlmEvent model.
    msids = ["cobsrqid"] + EventModel.event_msids + (EventModel.aux_msids or [])
    date_last_telemetry = maude.get_last_backorbit_date(msids)
    if date_last_telemetry < date_stop.date:
        date_stop = DateTime(date_last_telemetry)
        logger.info(f"Using last backorbit date {date_stop.date} as stop for {name}")
    return date_start, date_stop


def save_event_to_database(cls_name, event, event_model, models):
    try4times(event_model.save)
    logger.info("Added {} {}".format(cls_name, event_model))
    if "dur" in event and event["dur"] < 0:
        logger.info(
            "WARNING: negative event duration for {} {}".format(cls_name, event_model)
        )
    # Add any foreign rows (many to one)
    for foreign_cls_name, rows in event.get("foreign", {}).items():
        ForeignModel = getattr(models, foreign_cls_name)
        if isinstance(rows, np.ndarray):
            rows = [{key: row[key].tolist() for key in row.dtype.names} for row in rows]  # noqa: PLW2901
        for row in rows:
            # Convert to a plain dict if row is structured array
            foreign_model = ForeignModel.from_dict(row, logger)
            setattr(foreign_model, event_model.model_name, event_model)
            logger.debug("Adding {}".format(foreign_model))
            try4times(foreign_model.save)


def filter_events_to_add_to_database(
    EventModel: Type["kem.BaseEvent"],
    cls_name: str,
    events_in_dates: list[dict],
):
    """Filter events to add to the database.

    This takes a list of events (dicts) and returns those that are not already in the
    database.  This is done by checking the primary key of the event model or else
    (for telemetry events) by checking if the start time of the new event is close to
    that of an existing event.

    Parameters
    ----------
    EventModel : class
        Event model class to update (e.g. models.DsnComm)
    cls_name : str
        Event model class name
    events_in_dates : list of dict
        List of event dictionaries

    Returns
    -------
    event_models : list of EventModel
        List of event model instances to save
    events : list of dict
        List of event dictionaries (filtered from original)
    """
    from kadi.events import models

    # Determine which of the events is not already in the database and
    # put them in a list for saving.
    events = []
    event_models = []

    # Sampling period (sec) for the primary MSID for a telemetry event. Add 0.5 sec
    # margin for good measure. This is used below for fuzzy matching of new and existing
    # events.
    sample_period_max = (
        get_sample_period_max(EventModel.event_msids[0]) + 0.5
        if issubclass(EventModel, models.TlmEvent)
        else None
    )
    for event in events_in_dates:
        event_model = EventModel.from_dict(event, logger)

        # Check if event is already in database by the primary key. This works for
        # all non-telemetry events, and for telemetry events where the current data
        # source (CXC, MAUDE) matches the data source in the database.
        try:
            EventModel.objects.get(pk=event_model.pk)
        except EventModel.DoesNotExist:
            in_database = False
        else:
            in_database = True

        # Because of time stamp differences between MAUDE and CXC, we need to use a
        # time-based check for matches. See the TlmEvent.get_sample_period_max() method
        # for more details. The objects.filter() method gives True/False if there are
        # matches in boolean context.
        if not in_database and sample_period_max is not None:
            matches = EventModel.objects.filter(
                tstart__gt=(event_model.tstart - sample_period_max),
                tstart__lt=(event_model.tstart + sample_period_max),
            )
            if matches:
                in_database = True

        if in_database:
            logger.debug(
                "Skipping {} at {}: already in database".format(
                    cls_name, event["start"]
                )
            )
        else:
            events.append(event)
            event_models.append(event_model)

    return event_models, events


def get_sample_period_max(msid: str) -> float:
    """
    Get max telemetry sample period for ``msid``.

    This is used to allow switching between CXC and MAUDE telemetry as the input.
    CXC telemetry uses the time of first VCDU in the sample period, while MAUDE
    telemetry uses the time of the minor frame where the telemetry is sampled.

    This method returns the max sample period (32.8 sec / min sample rate) over the
    available telmeetry formats.

    Parameters
    ----------
    msid : str
        MSID name

    Returns
    -------
    sample_period_max : float
        Maximum sample period (secs)
    """
    # Start of event always corresponds to a transition in event_msids[0].
    tdb_msid = ska_tdb.msids[msid]
    sample_rate_min = np.min(tdb_msid.Tsmpl["SAMPLE_RATE"])  # samples / 128 mnf
    sample_period_max = 128 / sample_rate_min * 0.25625  # sec
    return sample_period_max


def get_all_telemetry_event_msids() -> set[str]:
    """
    Get all telemetry event MSIDs.

    Returns
    -------
    event_msids : list of str
        List of telemetry event MSIDs
    """
    from kadi.events import models

    event_msids = set()
    for event_model_cls in models.get_event_models().values():
        if issubclass(event_model_cls, models.TlmEvent):
            event_msids.update(event_model_cls.event_msids)
            event_msids.update(event_model_cls.aux_msids or [])

    return event_msids


def check_maude_server_has_new_telemetry(stop) -> bool:
    """
    Check if the MAUDE server has new telemetry since the last run of update_events.

    This also saves the last telemetry date in the `Update` model of the database using
    the `maude_latest_backorbit` name.

    Parameters
    ----------
    stop : CxoTimeLike
        Stop date for event update processing.

    Returns
    -------
    has_new_telemetry : bool
        True if new telemetry is available, else False.
    """
    from django.core.exceptions import ObjectDoesNotExist

    from kadi.events import models

    stop_date = DateTime(stop).date
    update_name = "maude_latest_backorbit"
    msids = get_all_telemetry_event_msids()
    date_last_telemetry = maude.get_last_backorbit_date(msids)

    # Normally opt.stop is NOW and telemetry cannot be in the future. But if the user
    # provided an earlier stop date, clip the telemetry date to that stop date. This is
    # for testing.
    if date_last_telemetry > stop_date:
        date_last_telemetry = stop_date
        logger.info(f"Clipping MAUDE telemetry date to {stop_date=}")

    try:
        update_model = models.Update.objects.get(name=update_name)
    except ObjectDoesNotExist:
        has_new = True
        update_model = models.Update(name=update_name, date=date_last_telemetry)
        message = (
            f"Adding initial entry for MAUDE telemetry date: {date_last_telemetry=}"
        )
    else:
        has_new = date_last_telemetry > update_model.date
        status = "New" if has_new else "No new"
        message = (
            f"{status} telemetry in MAUDE: "
            f"{update_model.date=} -> {date_last_telemetry=}"
        )

    logger.info(message)
    if has_new:
        update_model.date = date_last_telemetry
        logger.info(f"Updating Update.{update_name} date to {date_last_telemetry}")
        try4times(update_model.save)

    return has_new


def main(args=None):
    opt = get_opt_parser().parse_args(args)

    if opt.delete_from_start and opt.start is None:
        raise ValueError("Must specify --start when using --delete-from-start")

    logger.setLevel(opt.log_level)
    log_run_info(logger.info, opt)

    # Set the global root data directory.  This gets used in the django
    # setup to find the sqlite3 database file.
    os.environ["KADI"] = os.path.abspath(opt.data_root)
    from kadi.paths import EVENTS_DB_PATH

    logger.info("Event database : {}".format(EVENTS_DB_PATH()))
    logger.info("")

    from kadi.events import models

    if (
        opt.maude
        and not opt.models
        and not opt.start
        and not check_maude_server_has_new_telemetry(opt.stop)
    ):
        # Normal processing run (i.e. not testing or regenerating the database) with
        # MAUDE data source. In this case check if there is any new telemetry in MAUDE
        # since the last run. Since the MAUDE server updates reliably with each comm
        # pass, we can skip all event updates (including non-telem events) if MAUDE has
        # not updated. This means events3.db3 is not changed at all.
        logger.info("No new telemetry in MAUDE, skipping event updates")
        return

    # Allow for a cmd line option --start.  If supplied then loop the
    # effective value of opt.stop from start to the cmd line
    # --stop in steps of --loop-days
    if opt.start is None:
        date_stops = [opt.stop]
    else:
        t_starts = np.arange(
            DateTime(opt.start).secs, DateTime(opt.stop).secs, opt.loop_days * 86400.0
        )
        date_stops = [DateTime(t).date for t in t_starts]
        date_stops.append(opt.stop)

    # Get the event classes in models module
    EventModels = [
        event_model
        for event_model in models.get_event_models().values()
        if hasattr(event_model, "get_events")
    ]

    # Filter on ---model command line arg(s)
    if opt.models:
        EventModels = [
            event_model_cls
            for event_model_cls in EventModels
            if any(
                re.match(model_name, event_model_cls.__name__)
                for model_name in opt.models
            )
        ]

    # Update priority (higher priority value means earlier in processing)
    EventModels = sorted(EventModels, key=lambda x: x.update_priority, reverse=True)

    cheta_data_source = "maude allow_subset=False" if opt.maude else "cxc"

    for EventModel in EventModels:
        # Since MAUDE is slow but reliably available, we can use a shorter lookback for
        # telemetry events (default of 7 days instead of 21). The issue relates to the
        # max duration of an event since the lookback needs to span the event. The most
        # credible case here is a very long safe mode.
        if opt.maude and issubclass(EventModel, models.TlmEvent):
            logger.info(
                f"Setting {EventModel.__name__} lookback to {opt.maude_tlm_lookback} days"
            )
            EventModel.lookback = opt.maude_tlm_lookback

        try:
            if opt.delete_from_start:
                delete_from_date(EventModel, opt.start)

            for date_stop in date_stops:
                with fetch.data_source(cheta_data_source):
                    update_event_model(EventModel, date_stop, opt.maude)
        except Exception:
            # Something went wrong, but press on with processing other EventModels
            import traceback

            logger.error(f"ERROR in processing {EventModel.__name__}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
