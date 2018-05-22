Installing new versions of kadi or other apps to the production server

Local Testing
--------------
Initial setup
^^^^^^^^^^^^^^
::

  WEB_KADI=/proj/web-kadi

Test install
~~~~~~~~~~~~~
::

  ska # get into Ska environment

  TEST_PREFIX=$HOME/tmp/web-kadi  # or wherever
  mkdir -p $TEST_PREFIX/lib/python2.7/site-packages
  export PYTHONPATH=$TEST_PREFIX/lib/python2.7/site-packages:$WEB_KADI/lib/python2.7/site-packages

  cd ~/git/kadi

  git branch  # confirm correct web branch
  git status  # confirm no stray modifications

  cp ./manage.py ~/tmp

  rm -rf build
  rm -rf $TEST_PREFIX/lib/python2.7/site-packages/*

  # Install into TEST_PREFIX with flat kadi/ dir (without using egg)
  python setup.py install --prefix=$TEST_PREFIX --old-and-unmanageable

  # Run self tests (in particular note that the xdg_config test gives expected results)
  python setup.py test --args='-k test_events -s -v'

Install other packages (e.g. mica) to TEST_PREFIX if necessary

Run server and test
^^^^^^^^^^^^^^^^^^^^
::

  cd ~/tmp
  ./manage.py runserver
  # Check it out.
  # Look also at admin site: http://127.0.0.1:8000/admin

Production installation
-----------------------
::

  ska  # Enter ska flight environment on HEAD

  cd ~/git/kadi
  git branch  # confirm correct web branch
  git status  # confirm no stray modifications

  # Local (Apache) PREFIX and PYTHONPATH for web application packages.
  # Note that there is no Python installed at PREFIX.
  export PYTHONPATH=${WEB_KADI}/lib/python2.7/site-packages

  # Remove project and kadi.events app if needed
  ls -ld $PYTHONPATH/kadi*
  rm -rf $PYTHONPATH/kadi*.egg-info
  rm -rf $PYTHONPATH/kadi-bak

  # fast
  mv $PYTHONPATH/kadi{,-bak}
  python setup.py install --prefix=$WEB_KADI --old-and-unmanageable

  ls -ld $PYTHONPATH/kadi*

  # Run self tests (in particular note that the xdg_config test gives expected results)
  python setup.py test --args='-k test_events -s -v'

Restart::

  sudo /etc/rc.d/init.d/httpd-kadi restart
  # Check it out  http://kadi.cfa.harvard.edu
  # Look also at admin site: http://kadi.cfa.harvard.edu/admin
