# Licensed under a 3-clause BSD style license - see LICENSE.rst
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
from astropy.io import ascii

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

    converters = {key: [ascii.convert_numpy(getattr(np, type_))]
                  for key, type_ in types.items()}
    dat = ascii.read(lines, format='tab', guess=False, converters=converters,
                     fill_values=None)
    return dat


def ftp_put_to_lucky(ftp_dirname, local_files, user=None, logger=None):
    """Put the ``local_files`` onto lucky in /``user``/``ftp_dirname``. First put it at the top
    level, then when complete move it into a subdir eng_archive.  This lets the OCC side
    just watch for fully-uploaded files in that directory.

    The directory paths of ``local_files`` are stripped off so they all wind up
    in a flat structure within ``ftp_dirname``.
    """
    import Ska.File
    import Ska.ftp
    import uuid

    ftp = Ska.ftp.SFTP('lucky', logger=logger, user=user)
    if user is None:
        user = ftp.ftp.get_channel().transport.get_username()
    ftp.cd('/home/{}'.format(user))
    files = ftp.ls()

    if ftp_dirname not in files:
        ftp.mkdir(ftp_dirname)
    dir_files = ftp.ls(ftp_dirname)

    for local_file in local_files:
        file_dir, file_base = os.path.split(os.path.abspath(local_file))
        ftp_file = str(uuid.uuid4())  # random unique identifier
        with Ska.File.chdir(file_dir):
            ftp.put(file_base, ftp_file)
            destination_file = "{}/{}".format(ftp_dirname, file_base)
            # If final destination of file already exists, delete that file.
            if file_base in dir_files:
                ftp.delete(destination_file)
            # Rename the temp/uniq-id file to the final destination
            ftp.rename(ftp_file, destination_file)

    ftp.close()


def ftp_get_from_lucky(ftp_dirname, local_files, user=None, logger=None):
    """
    Get files from lucky.  This looks in remote ``ftp_dirname`` for files that
    have basenames matching those of ``local_files``.  The remote files
    are copied to the corresponding local_files names.  This is the converse
    of ftp_put_to_lucky and thus requires unique basenames for all files.
    """
    import Ska.ftp
    import Ska.File

    ftp = Ska.ftp.SFTP('lucky', logger=logger, user=user)
    if user is None:
        user = ftp.ftp.get_channel().transport.get_username()
    ftp.cd('/home/{}/{}'.format(user, ftp_dirname))
    for local_file in local_files:
        file_dir, file_base = os.path.split(os.path.abspath(local_file))
        if file_base in ftp.ls():
            with Ska.File.chdir(file_dir):
                ftp.get(file_base)
                ftp.delete(file_base)

    ftp.close()
