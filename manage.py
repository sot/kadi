#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import sys

if __name__ == "__main__":
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "kadi.settings")

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)
