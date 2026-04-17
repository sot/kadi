#################################################################
Maintain kadi commands
#################################################################

Rebuild commands files from scratch
###################################
See ``utils/migrate_cmds_to_cmds2.py`` for the process.

Rebuild recently-archived cmds
###############################

This process has been used in the case where a command event from the sheet was incorrect
but also more than 30 days old.  Most recent case was repairing cmds for the day
2026:065 HRC activity. A typo in the command event sheet caused a command related
to the activity to appear on day 2026:064. This was fixed in the command event sheet
too late to get in to the archived cmds (more than 30 days after the fact)
So, on day 2026:103, this process was used to fix the cmds for 2026:065.
The steps were:

1. Edit the command event sheet to fix the error
2. As unprivileged user or on non-HEAD machine, create a empty directory and copy
   cmds2.h5, cmds2.pkl, cmds3.h5, cmds3.pkl from
   flight archive to that directory.
3. Using an environment with at least kadi cmds 7.19.0
   kadi_update_cmds --kadi-cmds-version=2 --lookback 50
   kadi_update_cmds --kadi-cmds-version=3 --lookback 50
4. Review the new commands to confirm the fix is correct and that no other changes were made.
5. Either stop the flight update cron task, or just wait until it has just run
6. Copy the new cmds files (cmds2.h5 cmds2.pkl cmds3.h5 cmds3.pkl) to the flight
   archive as the aca user.
7. Restart the update cron task if it was stopped.

Example of validation steps for the 2026:065 fix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Set the ``KADI`` environment variable to the directory with the copied cmds files.
2. Confirm that ``CMD_EVT`` commands include only the corrected commands. Use
   ``KADI_CMDS_VERSION`` to confirm this for both the v2 and v3 versions of the cmds
   files. 
3. Run ``kadi_validate_states`` to confirm no new validation errors were introduced.
4. Write the new and flight ``cmds3`` files as text files after day 2026:050 and confirm
   by diff that the only changes are the corrected commands.
```
diff cmds3_flight.csv cmds3_test.csv
4175d4174
< 267969,2026:064:13:40:06.000,COMMAND_HW,2SPHVOF,0,0,889105275.184,CMD_EVT,-1
4502a4502
> 269067,2026:065:13:40:06.000,COMMAND_HW,2SPHVOF,0,0,889191675.184,CMD_EVT,-1
```


