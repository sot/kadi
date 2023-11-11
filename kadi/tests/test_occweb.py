# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import uuid
from pathlib import Path

import pytest
import requests
import Ska.File
import Ska.ftp

from kadi import occweb

try:
    Ska.ftp.parse_netrc()["lucky"]["login"]
    lucky = Ska.ftp.SFTP(occweb.LUCKY)
except Exception:
    HAS_LUCKY = False
else:
    HAS_LUCKY = True
    lucky.close()


def _test_put_get(user):
    filenames = ["test.dat", "test2.dat"]

    # Make a local temp dir and put files there
    local_tmpdir = Ska.File.TempDir()
    with Ska.File.chdir(local_tmpdir.name):
        for filename in filenames:
            open(filename, "w").write(filename)
        local_filenames = [os.path.abspath(x) for x in os.listdir(local_tmpdir.name)]

    remote_tmpdir = str(uuid.uuid4())  # random remote dir name
    occweb.ftp_put_to_lucky(remote_tmpdir, local_filenames, user=user)

    # Make a new local temp dir for the return
    local_tmpdir2 = Ska.File.TempDir()
    local_filenames = [os.path.join(local_tmpdir2.name, x) for x in filenames]
    occweb.ftp_get_from_lucky(remote_tmpdir, local_filenames, user=user)

    # Clean up remote temp dir
    lucky = Ska.ftp.SFTP(occweb.LUCKY)
    if user is None:
        user = lucky.ftp.get_channel().transport.get_username()
    lucky.rmdir("/home/{}/{}".format(user, remote_tmpdir))
    lucky.close()

    # Make sure round-tripped files are the same
    with Ska.File.chdir(local_tmpdir2.name):
        for filename in filenames:
            assert open(filename).read() == filename


@pytest.mark.skipif(not HAS_LUCKY, reason="No access to lucky FTP server")
def test_put_get_user_none():
    # Test the user=None code branch (gets username back from SFTP object, which
    # had previously gotten it from the netrc file).
    #
    # NOTE: this test CANNOT be run within 60 seconds of a previously run test
    # that accesses lucky.  This is due to a network security restriction for
    # the OCC.  If this fails unexpectedly then wait for at least 60 seconds.
    _test_put_get(user=None)


# The occweb login information should also be in the netrc (for maude) but kadi.occweb.get_auth
# uses the /proj/sot/ska/data/aspect_authorization area, so existence of the right file
# in there can just be checked by the get_auth routine
user, passwd = occweb.get_auth()
HAS_OCCWEB = bool(user is not None)


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
def test_ifot_fetch():
    events = occweb.get_ifot(
        "LOADSEG", start="2008:001:12:00:00", stop="2008:003:12:00:00"
    )
    assert len(events) == 1
    assert events[0]["tstart"] == "2008:002:21:00:00.000"


# Looks like there isn't a way to check status (HTTP codes), but these
# items should stay on these event pages
@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
def test_get_fdb_major_events():
    page = occweb.get_url("fdb_major_events")
    assert "Aspect Camera First Star Solution" in page


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
def test_get_fot_major_events():
    page = occweb.get_url("fot_major_events")
    assert "ACA Dark Current Calibration" in page


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
@pytest.mark.parametrize("str_or_Path", [str, Path])
@pytest.mark.parametrize("cache", [False, "update", True])
def test_get_occweb_dir(str_or_Path, cache):
    """Test get_occweb_dir and get_occweb_page (which is called in the process)"""
    path = str_or_Path("FOT/mission_planning/PRODUCTS/APPR_LOADS/2000/MAR/")
    url = f"https://occweb.cfa.harvard.edu/occweb/{path}"
    files_path = occweb.get_occweb_dir(path, cache=cache)
    files_url = occweb.get_occweb_dir(url, cache=cache)
    exp = [
        "      Name        Last modified   Size",
        "---------------- ---------------- ----",
        "Parent Directory               --    -",
        "       MAR0500D/ 2002-04-30 13:38    -",
        "       MAR1200D/ 2002-04-30 13:38    -",
        "       MAR1900E/ 2004-03-18 13:44    -",
        "       MAR2600C/ 2002-04-30 13:38    -",
    ]
    assert files_path.pformat_all() == exp
    assert files_url.pformat_all() == exp


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
@pytest.mark.parametrize("lowercase", [False, True])
@pytest.mark.parametrize("backslash", [True, False])
def test_get_occweb_noodle(lowercase, backslash):
    """Test get_occweb_dir and get_occweb_page (which is called in the process)
    for a noodle directory"""
    noodle = "//noodle/FOT"
    if lowercase:
        noodle = noodle.lower()
    path = f"{noodle}/mission_planning/PRODUCTS/APPR_LOADS/2000/MAR"
    if backslash:
        path = path.replace("/", "\\")
    files_path = occweb.get_occweb_dir(path)
    exp = [
        "      Name        Last modified   Size",
        "---------------- ---------------- ----",
        "Parent Directory               --    -",
        "       MAR0500D/ 2002-04-30 13:38    -",
        "       MAR1200D/ 2002-04-30 13:38    -",
        "       MAR1900E/ 2004-03-18 13:44    -",
        "       MAR2600C/ 2002-04-30 13:38    -",
    ]
    assert files_path.pformat_all() == exp


@pytest.mark.parametrize("noodle_path", occweb.NOODLE_OCCWEB_MAP)
def test_all_get_occweb_noodle(noodle_path):
    """Make sure we can get a directory from noodle for all available
    translations"""
    noodle = f"//noodle/{noodle_path}".lower()
    files = occweb.get_occweb_dir(noodle)
    assert len(files) > 0


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
@pytest.mark.parametrize("cache", [False, True])
def test_get_occweb_dir_fail(cache):
    """Test get_occweb_dir and get_occweb_page (which is called in the process)"""
    path = Path("FOT/mission_planning/PRODUCTS/APPR_LOADS/FAIL-NOT-THERE")
    with pytest.raises(requests.exceptions.HTTPError):
        occweb.get_occweb_dir(path, cache=cache)


@pytest.mark.skipif(not HAS_OCCWEB, reason="No access to OCCweb")
@pytest.mark.parametrize("cache", [False, "update", True])
def test_get_occweb_page_binary(cache):
    """Test get_occweb_page binary"""
    path = "FOT/mission_planning/PRODUCTS/APPR_LOADS/2000/MAR/"
    content_bytes = occweb.get_occweb_page(path, binary=True, cache=cache)
    content_str = occweb.get_occweb_page(path, binary=False, cache=cache)

    assert content_bytes.decode("utf-8") == content_str
