import os
import uuid

import Ska.ftp
import Ska.File

from kadi import occweb
import pyyaks.logger

logger = pyyaks.logger.get_logger()


def test_put_get():
    filenames = ['test.dat', 'test2.dat']

    # Make a local temp dir and put files there
    local_tmpdir = Ska.File.TempDir()
    with Ska.File.chdir(local_tmpdir.name):
        for filename in filenames:
            open(filename, 'w').write(filename)
        local_filenames = [os.path.abspath(x) for x in os.listdir(local_tmpdir.name)]

    # Get the remote user name for lucky
    netrc = Ska.ftp.parse_netrc()
    user = netrc['lucky']['login']

    remote_tmpdir = str(uuid.uuid4())  # random remote dir name
    occweb.ftp_put_to_lucky(remote_tmpdir, local_filenames, user=user)

    # Make a new local temp dir for the return
    local_tmpdir2 = Ska.File.TempDir()
    local_filenames = [os.path.join(local_tmpdir2.name, x) for x in filenames]
    occweb.ftp_get_from_lucky(remote_tmpdir, local_filenames, user=user)

    # Clean up remote temp dir
    lucky = Ska.ftp.SFTP('lucky')
    lucky.rmdir('/home/{}/{}'.format(user, remote_tmpdir))
    lucky.close()

    # Make sure round-tripped files are the same
    with Ska.File.chdir(local_tmpdir2.name):
        for filename in filenames:
            assert open(filename).read() == filename
