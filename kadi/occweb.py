"""
Provide an interface for getting pages from the OCCweb.  For good measure leave
this off of github.
"""

import re
import os
import configobj
import hashlib
import time
from collections import OrderedDict as odict

import numpy as np
import requests

from Chandra.Time import DateTime
import asciitable

CACHE_DIR = 'cache'
CACHE_TIME = 86000
TIMEOUT = 60


ROOTURL = 'http://occweb.cfa.harvard.edu'
URLS = {'fdb_major_events':  '/occweb/web/fdb_web/Major_Events.html',
        'fot_major_events':  '/occweb/web/fot_web/eng/reports/Chandra_major_events.htm',
        'ifot': '/occweb/web/webapps/ifot/ifot.php',
        }


def get_auth():
        authfile = '/proj/sot/ska/data/aspect_authorization/occweb-{}'.format(os.environ['USER'])
        config = configobj.ConfigObj(authfile)
        username = config.get('username')
        password = config.get('password')

        return (username, password)


def get_url(page, timeout=TIMEOUT):
    """
    Get the HTML for a web `page` on OCCweb.  This caches the result if the CACHE_DIR is
    available.  The cachefile is used if it is less than a day old.  This is mostly for
    testing, in production the cron job runs once a day.
    """
    url = ROOTURL + URLS[page]
    cachefile = os.path.join(CACHE_DIR, hashlib.sha1(url).hexdigest())
    now = time.time()
    if os.path.exists(cachefile) and now - os.stat(cachefile).st_mtime < CACHE_TIME:
        with open(cachefile, 'r') as f:
            html = f.read().decode('utf8')
    else:
        response = requests.get(url, auth=get_auth(), timeout=timeout)
        html = response.text

        if os.path.exists(CACHE_DIR):
            with open(cachefile, 'w') as f:
                f.write(html.encode('utf8'))

    return html


def get_ifot(event_type, start=None, stop=None, props=[], columns=[], timeout=TIMEOUT, types={}):
    start = DateTime('1998:001' if start is None else start)
    stop = DateTime(stop)
    event_props = '.'.join([event_type] + props)

    params = odict(r='home',
                   t='qserver',
                   format='tsv',
                   tstart=start.date,
                   tstop=stop.date,
                   e=event_props,
                   ul='7',
                   )
    if columns:
        params['columns'] = ','.join(columns)

    # Get the TSV data for the iFOT event table
    url = ROOTURL + URLS['ifot']
    response = requests.get(url, auth=get_auth(), params=params, timeout=timeout)

    text = response.text.encode('ascii', 'ignore')
    text = re.sub(r'\r\n', ' ', text)
    lines = [x for x in text.split('\t\n') if x.strip()]

    converters = {key: [asciitable.convert_numpy(getattr(np, type_))]
                  for key, type_ in types.items()}
    dat = asciitable.read(lines, Reader=asciitable.Tab, guess=False, converters=converters)
    return dat
