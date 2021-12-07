# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Update the events database"""

import os
import re
import argparse
import time

import numpy as np

import pyyaks.logger
from Chandra.Time import DateTime
from ska_helpers.run_info import log_run_info
from kadi import __version__  # noqa

logger = None  # for pyflakes


def get_opt(args=None):
    parser = argparse.ArgumentParser(description='Update the events database')
    parser.add_argument("--stop",
                        default=DateTime().date,
                        help="Processing stop date (default=NOW)")
    parser.add_argument("--start",
                        default=None,
                        help=("Processing start date (loops by --loop-days "
                              "until --stop date if set)"))
    parser.add_argument("--delete-from-start",
                        action='store_true',
                        help=("Delete events after --start and reset update time to --start"))
    parser.add_argument("--loop-days",
                        default=100,
                        type=int,
                        help=("Number of days in interval for looping (default=100)"))
    parser.add_argument("--log-level",
                        type=int,
                        default=pyyaks.logger.INFO,
                        help=("Logging level"))
    parser.add_argument("--model",
                        action='append',
                        dest='models',
                        help="Model class name to process [match regex] (default = all)")
    parser.add_argument("--data-root",
                        default=".",
                        help="Root data directory (default='.')")

    args = parser.parse_args(args)
    return args


def try4times(func, *arg, **kwarg):
    """
    Work around problems with sqlite3 database getting locked out from writing,
    presumably due to read activity.  Not completely understood.

    This function will try to run func(*arg, **kwarg) a total of 4 times with an
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
            if 'database is locked' in str(err):
                # Locked DB, issue informational warning
                logger.info('Warning: locked database, waiting {} seconds'.format(delay))
            else:
                # Something else so just re-raise
                raise
        else:
            # Success, jump out of loop
            break

    else:
        # After 4 tries bail out with an exception
        raise OperationalError('database is locked')


def delete_from_date(EventModel, start, set_update_date=True):
    from .events import models

    date_start = DateTime(start).date
    cls_name = EventModel.__name__

    if set_update_date:
        update = models.Update.objects.get(name=cls_name)
        logger.info('Updating {} date from {} to {}'.format(cls_name, update.date, date_start))
        update.date = date_start
        try4times(update.save)

    events = EventModel.objects.filter(start__gte=date_start)
    logger.info('Deleting {} {} events after {}'.format(events.count(), cls_name, date_start))
    try4times(events.delete)


def update(EventModel, date_stop):
    import django.db
    from django.core.exceptions import ObjectDoesNotExist
    from .events import models

    date_stop = DateTime(date_stop)
    cls_name = EventModel.__name__

    try:
        update = models.Update.objects.get(name=cls_name)
    except ObjectDoesNotExist:
        logger.info('No previous update for {} found'.format(cls_name))
        duration = EventModel.lookback
        update = models.Update(name=cls_name, date=date_stop.date)
        date_start = date_stop - EventModel.lookback
    else:
        duration = date_stop - DateTime(update.date)
        date_start = DateTime(update.date) - EventModel.lookback
        update.date = date_stop.date

        # Some events like LoadSegment or DsnComm might change in the database after
        # having been ingested.  Use lookback_delete (less than lookback) to
        # always remove events in that range and re-ingest.
        if duration >= 0.5 and hasattr(EventModel, 'lookback_delete'):
            delete_date = DateTime(update.date) - EventModel.lookback_delete
            delete_from_date(EventModel, delete_date, set_update_date=False)

    if duration < 0.5:
        logger.info('Skipping {} events because update duration={:.1f} is < 0.5 day'
                    .format(cls_name, duration))
        return

    # Some events like LoadSegment, DsnComm are defined into the future, so
    # modify date_stop accordingly.  Note that update.date is set to the
    # nominal date_stop (typically NOW), and this refers more to the last date
    # of processing rather than the actual last date in the archive.
    if hasattr(EventModel, 'lookforward'):
        date_stop = date_stop + EventModel.lookforward

    logger.info('Updating {} events from {} to {}'
                .format(cls_name, date_start.date[:-4], date_stop.date[:-4]))

    # Get events for this model from telemetry.  This is returned as a list
    # of dicts with key/val pairs corresponding to model fields.
    events_in_dates = EventModel.get_events(date_start, date_stop)

    # Determine which of the events is not already in the database and
    # put them in a list for saving.
    event_models, events = get_events_and_event_models(EventModel, cls_name, events_in_dates)

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
                logger.warn(f'WARNING: IntegrityError skipping {event_model}')
                logger.warn(f'Event dict:\n{event}')
                logger.warn(f'Traceback:\n{traceback.format_exc()}')
                continue

        # If processing got here with no exceptions then save the event update
        # information to database
        update.save()


def save_event_to_database(cls_name, event, event_model, models):
    try4times(event_model.save)
    logger.info('Added {} {}'.format(cls_name, event_model))
    if 'dur' in event and event['dur'] < 0:
        logger.info('WARNING: negative event duration for {} {}'
                    .format(cls_name, event_model))
    # Add any foreign rows (many to one)
    for foreign_cls_name, rows in event.get('foreign', {}).items():
        ForeignModel = getattr(models, foreign_cls_name)
        if isinstance(rows, np.ndarray):
            rows = [{key: row[key].tolist() for key in row.dtype.names} for row in rows]
        for row in rows:
            # Convert to a plain dict if row is structured array
            foreign_model = ForeignModel.from_dict(row, logger)
            setattr(foreign_model, event_model.model_name, event_model)
            logger.verbose('Adding {}'.format(foreign_model))
            try4times(foreign_model.save)


def get_events_and_event_models(EventModel, cls_name, events_in_dates):
    # Determine which of the events is not already in the database and
    # put them in a list for saving.
    events = []
    event_models = []
    for event in events_in_dates:
        event_model = EventModel.from_dict(event, logger)
        try:
            EventModel.objects.get(pk=event_model.pk)
        except EventModel.DoesNotExist:
            events.append(event)
            event_models.append(event_model)
        else:
            logger.verbose('Skipping {} at {}: already in database'
                           .format(cls_name, event['start']))
    return event_models, events


def main():
    global logger

    opt = get_opt()

    logger = pyyaks.logger.get_logger(name=__name__, level=opt.log_level,
                                      format="%(asctime)s %(message)s")
    log_run_info(logger.info, opt)

    # Set the global root data directory.  This gets used in the django
    # setup to find the sqlite3 database file.
    os.environ['KADI'] = os.path.abspath(opt.data_root)
    from .paths import EVENTS_DB_PATH

    logger.info('Event database : {}'.format(EVENTS_DB_PATH()))
    logger.info('')

    from .events import models

    # Allow for a cmd line option --start.  If supplied then loop the
    # effective value of opt.stop from start to the cmd line
    # --stop in steps of --max-lookback-time
    if opt.start is None:
        date_stops = [opt.stop]
    else:
        t_starts = np.arange(DateTime(opt.start).secs,
                             DateTime(opt.stop).secs,
                             opt.loop_days * 86400.)
        date_stops = [DateTime(t).date for t in t_starts]
        date_stops.append(opt.stop)

    # Get the event classes in models module
    EventModels = [Model for name, Model in vars(models).items()
                   if (isinstance(Model, type)  # is a class
                       and issubclass(Model, models.BaseEvent)  # is a BaseEvent subclass
                       and 'Meta' not in Model.__dict__  # is not a base class
                       and hasattr(Model, 'get_events')  # can get events
                       )]

    # Filter on ---model command line arg(s)
    if opt.models:
        EventModels = [x for x in EventModels
                       if any(re.match(y, x.__name__) for y in opt.models)]

    # Update priority (higher priority value means earlier in processing)
    EventModels = sorted(EventModels, key=lambda x: x.update_priority, reverse=True)

    for EventModel in EventModels:
        try:
            if opt.delete_from_start and opt.start is not None:
                delete_from_date(EventModel, opt.start)

            for date_stop in date_stops:
                update(EventModel, date_stop)
        except Exception:
            # Something went wrong, but press on with processing other EventModels
            import traceback
            logger.error(f'ERROR in processing {EventModel}')
            logger.error(f'Traceback:\n{traceback.format_exc()}')


if __name__ == '__main__':
    main()
