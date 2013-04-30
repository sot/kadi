"""Update the events database"""

import os
import re
import types
import argparse

import numpy as np

import pyyaks.logger
from Chandra.Time import DateTime

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


def update(EventModel, date_stop):
    import django.db
    from django.core.exceptions import ObjectDoesNotExist
    from .events import models

    date_stop = DateTime(date_stop)
    cls_name = EventModel.__name__

    try:
        update = models.Update.objects.get(name=cls_name)
        duration = date_stop - DateTime(update.date)
        date_start = DateTime(update.date) - EventModel.lookback
        update.date = date_stop.date
    except ObjectDoesNotExist:
        logger.info('No previous update for {} found'.format(cls_name))
        duration = EventModel.lookback
        update = models.Update(name=cls_name, date=date_stop.date)
        date_start = date_stop - EventModel.lookback

    if duration < 0.5:
        logger.info('Skipping {} events because update duration={:.1f} is < 0.5 day'
                    .format(cls_name, duration))
        return

    logger.info('Updating {} events from {} to {}'
                .format(cls_name, date_start.date[:-4], date_stop.date[:-4]))

    # Get events for this model from telemetry.  This is returned as a list
    # of dicts with key/val pairs corresponding to model fields
    events = EventModel.get_events(date_start, date_stop)

    with django.db.transaction.commit_on_success():
        for event in events:
            # Try to save event.  Use force_insert=True because otherwise django will
            # update if the event primary key already exists.  In this case we want to
            # force an exception and move on to the next event.
            try:
                event_model = EventModel.from_dict(event, logger)
                event_model.save(force_insert=True)
            except django.db.utils.IntegrityError:
                logger.verbose('Skipping {} at {}: already in database'
                               .format(cls_name, event['start']))
                continue

            logger.info('Added {} {}'.format(cls_name, event_model))

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
                    foreign_model.save()

    # If processing got here with no exceptions then save the event update
    # information to database
    update.save()


def main():
    global logger

    opt = get_opt()

    # Set the global root data directory.  This gets used in the django
    # setup to find the sqlite3 database file.
    os.environ['KADI'] = os.path.abspath(opt.data_root)
    from .events import models

    logger = pyyaks.logger.get_logger(name='events', level=opt.log_level,
                                      format="%(asctime)s %(message)s")

    # Allow for a cmd line option --date-start.  If supplied then loop the
    # effective value of opt.stop from date_start to the cmd line
    # --date-now in steps of --max-lookback-time

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
                   if (isinstance(Model, (type, types.ClassType))  # is a class
                       and issubclass(Model, models.BaseEvent)  # is a BaseEvent subclass
                       and 'Meta' not in Model.__dict__  # is not a base class
                       and hasattr(Model, 'get_events')  # can get events
                       )]

    if opt.models:
        EventModels = [x for x in EventModels
                       if any(re.match(y, x.__name__) for y in opt.models)]

    for EventModel in EventModels:
        for date_stop in date_stops:
            update(EventModel, date_stop)

if __name__ == '__main__':
    main()
