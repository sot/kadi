import os
import uuid

# See https://github.com/paramiko/paramiko/issues/735.  Without this
# hack the module-level call to Ska.ftp.SFTP('lucky') hangs during
# test collection by pytest.  Note that this does not work with
# paramiko 2.0.0.
from paramiko import py3compat
py3compat.u('dirty hack')

import Ska.ftp
import Ska.File
import pytest

from kadi import occweb
import pyyaks.logger

logger = pyyaks.logger.get_logger()


try:
    Ska.ftp.parse_netrc()['lucky']['user']
    lucky = Ska.ftp.SFTP('lucky')
except Exception:
    HAS_LUCKY = False
else:
    HAS_LUCKY = True
    lucky.close()


def _test_put_get(user):
    filenames = ['test.dat', 'test2.dat']

    # Make a local temp dir and put files there
    local_tmpdir = Ska.File.TempDir()
    with Ska.File.chdir(local_tmpdir.name):
        for filename in filenames:
            open(filename, 'w').write(filename)
        local_filenames = [os.path.abspath(x) for x in os.listdir(local_tmpdir.name)]

    remote_tmpdir = str(uuid.uuid4())  # random remote dir name
    occweb.ftp_put_to_lucky(remote_tmpdir, local_filenames, user=user)

    # Make a new local temp dir for the return
    local_tmpdir2 = Ska.File.TempDir()
    local_filenames = [os.path.join(local_tmpdir2.name, x) for x in filenames]
    occweb.ftp_get_from_lucky(remote_tmpdir, local_filenames, user=user)

    # Clean up remote temp dir
    print('connecting to lucky')
    lucky = Ska.ftp.SFTP('lucky', logger=logger)
    if user is None:
        user = lucky.ftp.get_channel().transport.get_username()
    lucky.rmdir('/home/{}/{}'.format(user, remote_tmpdir))
    lucky.close()

    # Make sure round-tripped files are the same
    with Ska.File.chdir(local_tmpdir2.name):
        for filename in filenames:
            assert open(filename).read() == filename


@pytest.mark.skipif('not HAS_LUCKY')
def test_put_get_user_from_netrc():
    # Get the remote user name for lucky
    netrc = Ska.ftp.parse_netrc()
    user = netrc['lucky']['login']
    _test_put_get(user=user)


@pytest.mark.skipif('not HAS_LUCKY')
def test_put_get_user_none():
    # Test the user=None code branch (gets username back from SFTP object, which
    # had previously gotten it from the netrc file).
    _test_put_get(user=None)
