# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Provide an interface for getting pages from the OCCweb.
"""

import hashlib
import logging
import os
import re
import time
from collections import OrderedDict as odict
from pathlib import Path

import configobj
import requests
from astropy.io import ascii
from astropy.table import Table
from astropy.utils.data import download_file
from chandra_time import DateTime
from ska_helpers.retry import retry_func

# This is for deprecated functionality to cache to a local directory
CACHE_DIR = "cache"
CACHE_TIME = 86000
TIMEOUT = 60

ROOTURL = "https://occweb.cfa.harvard.edu"
URLS = {
    "fdb_major_events": "/occweb/web/fdb_web/Major_Events.html",
    "fot_major_events": "/occweb/web/fot_web/eng/reports/Chandra_major_events.htm",
    "ifot": "/occweb/web/webapps/ifot/ifot.php",
}
LUCKY = "lucky.cfa.harvard.edu"

NOODLE_OCCWEB_MAP = {
    "FOT": "occweb/FOT",
    "GRETA/mission/Backstop": "Backstop",
    "vweb": "occweb/web",
}

# Initialize 'kadi.occweb' logger.
logger = logging.getLogger(__name__)


def get_auth(username=None, password=None):
    if username and password:
        return (username, password)

    ska = os.environ.get("SKA")
    if ska:
        user = username or os.environ.get("USER") or os.environ.get("LOGNAME")
        authfile = Path(ska, "data", "aspect_authorization", f"occweb-{user}")
        config = configobj.ConfigObj(str(authfile))
        username = config.get("username")
        password = config.get("password")

    # If $SKA doesn't have occweb credentials try .netrc.
    if username is None:
        try:
            import ska_ftp

            # Do this as a tuple so the operation is atomic
            username, password = (
                ska_ftp.parse_netrc()["occweb"]["login"],
                ska_ftp.parse_netrc()["occweb"]["password"],
            )
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

    cachefile = os.path.join(CACHE_DIR, hashlib.sha1(url.encode("utf-8")).hexdigest())
    now = time.time()
    if os.path.exists(cachefile) and now - os.stat(cachefile).st_mtime < CACHE_TIME:
        with open(cachefile, "rb") as f:
            html = f.read().decode("utf8")
    else:
        response = retry_func(requests.get)(url, auth=get_auth(), timeout=timeout)
        html = response.text

        if os.path.exists(CACHE_DIR):
            with open(cachefile, "wb") as f:
                f.write(html.encode("utf8"))

    return html


def get_ifot(
    event_type,
    start=None,
    stop=None,
    props=None,
    columns=None,
    timeout=TIMEOUT,
    types=None,
):
    """Get the iFOT event table for a given event type.

    Parameters
    ----------
    event_type : str
        Event type (e.g. "ECLIPSE", "LOADSEG", "CAP")
    start : str, CxoTimeLike
        Start time for query
    stop : str, CxoTimeLike
        Stop time for query
    props : list of str
        List of iFOT properties to return
    columns : list of str
        List of columns to return
    timeout : float
        Timeout for the request
    types : dict
        Dictionary of column types

    Returns
    -------
    dat : astropy.table.Table
        Table of iFOT events
    """
    start = DateTime("1998:001:12:00:00" if start is None else start)
    stop = DateTime(stop)
    if props is None:
        props = []
    if columns is None:
        columns = []
    if types is None:
        types = {}

    event_props = ".".join([event_type] + props)

    params = odict(
        r="home",
        t="qserver",
        format="tsv",
        tstart=start.date,
        tstop=stop.date,
        e=event_props,
        ul="7",
    )
    if columns:
        params["columns"] = ",".join(columns)

    # Get the TSV data for the iFOT event table
    url = ROOTURL + URLS["ifot"]
    response = retry_func(requests.get)(
        url, auth=get_auth(), params=params, timeout=timeout
    )

    # For Py2 convert from unicode to ASCII str
    text = response.text
    text = re.sub(r"\r\n", " ", text)
    lines = [x for x in text.split("\t\n") if x.strip()]

    dat = ascii.read(
        lines, format="tab", guess=False, converters=types, fill_values=None
    )
    return dat


def ftp_put_to_lucky(ftp_dirname, local_files, user=None, logger=None):
    """Put the ``local_files`` onto lucky in /``user``/``ftp_dirname``.

    First put it at the top level, then when complete move it into a subdir
    eng_archive.  This lets the OCC side just watch for fully-uploaded files in
    that directory.

    The directory paths of ``local_files`` are stripped off so they all wind up
    in a flat structure within ``ftp_dirname``.
    """
    import uuid

    import ska_file
    import ska_ftp

    ftp = ska_ftp.SFTP(LUCKY, logger=logger, user=user)
    if user is None:
        user = ftp.ftp.get_channel().transport.get_username()
    ftp.cd("/home/{}".format(user))
    files = ftp.ls()

    if ftp_dirname not in files:
        ftp.mkdir(ftp_dirname)
    dir_files = ftp.ls(ftp_dirname)

    for local_file in local_files:
        file_dir, file_base = os.path.split(os.path.abspath(local_file))
        ftp_file = str(uuid.uuid4())  # random unique identifier
        with ska_file.chdir(file_dir):
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
    Get files from lucky.

    This looks in remote ``ftp_dirname`` for files that have basenames matching those of
    ``local_files``.  The remote files are copied to the corresponding local_files
    names.  This is the converse of ftp_put_to_lucky and thus requires unique basenames
    for all files.
    """
    import ska_file
    import ska_ftp

    ftp = ska_ftp.SFTP(LUCKY, logger=logger, user=user)
    if user is None:
        user = ftp.ftp.get_channel().transport.get_username()
    ftp.cd("/home/{}/{}".format(user, ftp_dirname))
    for local_file in local_files:
        file_dir, file_base = os.path.split(os.path.abspath(local_file))
        if file_base in ftp.ls():
            with ska_file.chdir(file_dir):
                ftp.get(file_base)
                ftp.delete(file_base)

    ftp.close()


def get_auth_http_headers(username, password):
    """Get HTTP header for basic authentication"""
    from base64 import b64encode

    authorization = "Basic " + b64encode(
        f"{username}:{password}".encode("utf-8")
    ).decode("utf-8")
    headers = {"Authorization": authorization}
    return headers


def get_occweb_page(
    path, timeout=30, cache=False, binary=False, user=None, password=None
):
    r"""Get contents of ``path`` on OCCweb.

    This returns the contents of the OCCweb page at ``path`` where it assumed
    that ``path`` is either a URL (which starts with 'https://occweb') or a
    relative path::

      https://occweb.cfa.harvard.edu/occweb/<path>

    In addition ``path`` can be a file or directory path starting with one of
    the following Noodle directory paths. These get translated to an equivalent
    location on OCCweb::

      \\noodle\FOT
      \\noodle\GRETA\mission\Backstop
      \\noodle\vweb

    If ``user`` and ``password`` are not provided then it is required that
    credentials are stored in the file ``~/.netrc``. See the ``ska_ftp`` package
    for details.

    If the page is not found then a ``requests.exceptions.HTTPError`` is raised.

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
    if isinstance(path, str):
        path = path.replace("\\", "/")
        if path.startswith("//noodle/"):
            for noodle_prefix, occweb_prefix in NOODLE_OCCWEB_MAP.items():
                noodle = "//noodle/" + noodle_prefix
                if re.match(noodle, path, re.IGNORECASE):
                    path = re.sub(
                        noodle, ROOTURL + "/" + occweb_prefix, path, flags=re.IGNORECASE
                    )
                    break
            else:
                raise ValueError(f"unrecognized noodle path: {path}")

    if isinstance(path, str) and path.startswith("https://occweb"):
        url = path
    else:
        path = Path(path)
        # Handle the case of providing an absolute path into the OCCweb tree, e.g.
        # /FOT/mission_planning/Backstop instead of FOT/mission_planning/Backstop.
        if path.is_absolute():
            path = path.relative_to(path.anchor)
        url = ROOTURL + (Path("/occweb") / path).as_posix()

    logger.info(f"Getting OCCweb {path} with {cache=}")
    if cache:
        from urllib.request import HTTPError

        user, password = get_auth(user, password)
        headers = get_auth_http_headers(user, password)
        try:
            cachefile = retry_func(download_file)(
                url,
                cache=cache,
                show_progress=False,
                http_headers=headers,
                timeout=timeout,
                pkgname="kadi",
            )
        except HTTPError as err:
            # Re-raise so caller can handle it with one exception class
            raise requests.exceptions.HTTPError(str(err)) from err
        pth = Path(cachefile)
        out = pth.read_bytes() if binary else pth.read_text()
    else:
        # Not caching so don't write to a file (per download_file) just get it.
        auth = get_auth(user, password)
        req = retry_func(requests.get)(url, auth=auth, timeout=timeout)
        req.raise_for_status()  # raise exception if not 200

        out = req.content if binary else req.text

    return out


def get_occweb_dir(path, timeout=30, cache=False, user=None, password=None):
    """Get directory listing for ``path`` on OCCweb.

    This returns the directory for ``path`` on OCCweb, where it assumed that
    ``path`` is a relative path::

      https://occweb.cfa.harvard.edu/occweb/<path>

    If ``user`` and ``password`` are not provided then it is required that
    credentials are stored in the file ``~/.netrc``. See the ``ska_ftp`` package
    for details.

    Parameters
    ----------
    path : str, Path
        Relative path on OCCweb
    timeout : int
        Timeout in seconds for the request
    cache : bool
        If True, cache the result and check cache for subsequent calls
    user : str, optional
        Username for OCCweb authentication
    password : str, optional
        Password for OCCweb authentication

    Returns
    -------
    out
        astropy Table
        Table of directory entries
    """
    html = get_occweb_page(
        path, timeout=timeout, cache=cache, user=user, password=password
    )
    out = Table.read(html, format="ascii.html", guess=False)
    del out["col0"]
    del out["Description"]
    return out
