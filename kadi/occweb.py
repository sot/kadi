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
from pathlib import Path

import numpy as np
import requests

from Chandra.Time import DateTime
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import download_file
import pyyaks.logger

# This is for deprecated functionality to cache to a local directory
CACHE_DIR = 'cache'
CACHE_TIME = 86000
TIMEOUT = 60

ROOTURL = 'https://occweb.cfa.harvard.edu'
URLS = {'fdb_major_events': '/occweb/web/fdb_web/Major_Events.html',
        'fot_major_events': '/occweb/web/fot_web/eng/reports/Chandra_major_events.htm',
        'ifot': '/occweb/web/webapps/ifot/ifot.php',
        }

# Initialize 'kadi.occweb' logger with WARNING level. Logging in this
# module is INFO.
logger = pyyaks.logger.get_logger(
    __name__, format='%(asctime)s %(funcName)s - %(message)s',
    level=pyyaks.logger.WARNING)
for handler in logger.handlers:
    handler.setLevel(pyyaks.logger.DEBUG)


def get_auth(username=None, password=None):
    if username and password:
        return (username, password)

    ska = os.environ.get('SKA')
    if ska:
        user = username or os.environ.get('USER') or os.environ.get('LOGNAME')
        authfile = Path(ska, 'data', 'aspect_authorization', f'occweb-{user}')
        config = configobj.ConfigObj(str(authfile))
        username = config.get('username')
        password = config.get('password')

    # If $SKA doesn't have occweb credentials try .netrc.
    if username is None:
        try:
            import Ska.ftp
            # Do this as a tuple so the operation is atomic
            username, password = (Ska.ftp.parse_netrc()['occweb']['login'],
                                  Ska.ftp.parse_netrc()['occweb']['password'])
        except Exception:
            pass

    return (username, password)


def get_url(page, timeout=TIMEOUT):
    """Get the HTML for a web `page` on OCCweb.

    DEPRECATED for new code, use get_occweb_page() instead.

    This caches the result if the CACHE_DIR is available.  The cachefile is used
    if it is less than a day old.  This is mostly for testing, in production the
    cron job runs once a day.
    """
    url = ROOTURL + URLS[page]

    cachefile = os.path.join(CACHE_DIR, hashlib.sha1(url.encode('utf-8')).hexdigest())
    now = time.time()
    if os.path.exists(cachefile) and now - os.stat(cachefile).st_mtime < CACHE_TIME:
        with open(cachefile, 'rb') as f:
            html = f.read().decode('utf8')
    else:
        response = requests.get(url, auth=get_auth(), timeout=timeout)
        html = response.text

        if os.path.exists(CACHE_DIR):
            with open(cachefile, 'wb') as f:
                f.write(html.encode('utf8'))

    return html


def get_ifot(event_type, start=None, stop=None, props=[], columns=[], timeout=TIMEOUT, types={}):
    start = DateTime('1998:001:12:00:00' if start is None else start)
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

    # For Py2 convert from unicode to ASCII str
    text = response.text
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


def get_auth_http_headers(username, password):
    """Get HTTP header for basic authentication"""
    from base64 import b64encode
    authorization = ('Basic '
                     + b64encode(f'{username}:{password}'.encode('utf-8')).decode('utf-8'))
    headers = {'Authorization': authorization}
    return headers


def get_occweb_page(path, timeout=30, cache=False, binary=False,
                    user=None, password=None):
    """Get contents of ``path`` on OCCweb.

    This returns the contents of the OCCweb page at ``path`` where it assumed
    that ``path`` is either a URL (which starts with 'https://occweb') or a
    relative path::

      https://occweb.cfa.harvard.edu/occweb/<path>

    If ``user`` and ``password`` are not provided then it is required that
    credentials are stored in the file ``~/.netrc``. See the ``Ska.ftp`` package
    for details.

    :param path: str, Path
        Relative path on OCCweb or an OCCweb URL
    :param timeout: int
        Timeout in seconds for the request
    :param cache: bool
        If True, cache the result and check cache for subsequent calls
    :param binary: bool
        If True, return the binary contents of the page
    :param user: str, optional
        Username for OCCweb authentication
    :param password: str, optional
        Password for OCCweb authentication
    :returns: str, bytes
        File contents (str if ``binary`` is False, bytes if ``binary`` is True)
    """
    if isinstance(path, str) and path.startswith('https://occweb'):
        url = path
    else:
        url = ROOTURL + (Path('/occweb') / path).as_posix()

    logger.verbose(f'Getting OCCweb {path} with {cache=}')
    if cache:
        user, password = get_auth(user, password)
        headers = get_auth_http_headers(user, password)
        cachefile = download_file(url, cache=cache, show_progress=False,
                                  http_headers=headers, timeout=timeout,
                                  pkgname='kadi')
        pth = Path(cachefile)
        out = pth.read_bytes() if binary else pth.read_text()
    else:
        # Not caching so don't write to a file (per download_file) just get it.
        auth = get_auth(user, password)
        req = requests.get(url, auth=auth, timeout=timeout)
        req.raise_for_status()  # raise exception if not 200

        out = req.content if binary else req.text

    return out


def get_occweb_dir(path, timeout=30, cache=False, user=None, password=None):
    """Get directory listing for ``path`` on OCCweb.

    This returns the directory for ``path`` on OCCweb, where it assumed that
    ``path`` is a relative path::

      https://occweb.cfa.harvard.edu/occweb/<path>

    If ``user`` and ``password`` are not provided then it is required that
    credentials are stored in the file ``~/.netrc``. See the ``Ska.ftp`` package
    for details.

    :param path: str, Path
        Relative path on OCCweb
    :param timeout: int
        Timeout in seconds for the request
    :param cache: bool
        If True, cache the result and check cache for subsequent calls
    :param user: str, optional
        Username for OCCweb authentication
    :param password: str, optional
        Password for OCCweb authentication
    :returns: astropy Table
        Table of directory entries
    """
    html = get_occweb_page(path, timeout=timeout, cache=cache)
    out = Table.read(html, format='ascii.html', guess=False)
    del out['col0']
    del out['Description']
    return out
