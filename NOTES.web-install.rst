Installing new versions of kadi or other apps to the production server

Local Testing
--------------
Initial setup
^^^^^^^^^^^^^^
::

  conda create -n test-web-kadi python django=1.6 numpy astropy pyyaks jinja2
  source activate test-web-kadi
  PREFIX=`python -c 'import sys; print(sys.prefix)'`
  cd ~/git/kadi
  cp ./manage.py ~/tmp


Kadi
^^^^
::

  cd ~/git/kadi
  git branch  # confirm correct web branch
  git status  # confirm no stray modifications
  rm -rf build
  rm -rf $PREFIX/lib/python2.7/site-packages/kadi*
  python setup.py install

Mica
^^^^^
::

  cd ~/git/mica
  git branch  # confirm correct web branch
  git status  # confirm no stray modifications
  rm -rf build
  rm -rf $PREFIX/lib/python2.7/site-packages/mica*
  python setup.py install

Run server and test
^^^^^^^^^^^^^^^^^^^^
::

  cd ~/tmp
  ./manage.py runserver
  # Check it out.
  # Look also at admin site: http://127.0.0.1:8000/admin

Production installation
-----------------------
Basic setup::

  ska  # Enter ska flight environment on HEAD

  # Local (Apache) PREFIX and PYTHONPATH for web application packages.
  # Note that there is no Python installed at PREFIX.
  PREFIX=/proj/web-kadi
  export PYTHONPATH=${PREFIX}/lib/python2.7/site-packages

Kadi
^^^^^
As needed::

  cd ~/git/kadi
  git branch  # confirm correct web branch
  git status  # confirm no stray modifications

  # Remove project and kadi.events app if needed
  ls -ld $PYTHONPATH/kadi*
  rm -rf $PYTHONPATH/kadi*.egg-info
  rf -rf $PYTHONPATH/kadi-bak

  # fast
  mv $PYTHONPATH/kadi{,-bak}
  python setup.py install --prefix=$PREFIX

  ls -ld $PYTHONPATH/kadi*

Restart::

  sudo /etc/rc.d/init.d/httpd-kadi restart
  # Check it out  http://kadi.cfa.harvard.edu
  # Look also at admin site: http://kadi.cfa.harvard.edu/admin

Mica
^^^^^
As needed::

  cd ~/git/mica
  git branch  # confirm correct web branch
  git status  # confirm no stray modifications

  ls -ld $PYTHONPATH/mica*
  rm -rf $PYTHONPATH/mica*
  python setup.py install --prefix=$PREFIX



