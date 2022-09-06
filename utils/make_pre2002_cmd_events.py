"""Make initial version of pre2002/cmd_events.csv from kadi events.

This has then been hand-edited to fix a few things.
"""
from astropy.table import Table
from cxotime import CxoTime

from kadi import events

# State,Date,Event,Params,Author,Reviewer,Comment

print(events.normal_suns.filter("1999:339", "2002:007"))
print(events.safe_suns.filter("1999:339", "2002:007"))
print(events.scs107s.filter("1999:339", "2002:007"))

start = CxoTime("1999:339")
stop = CxoTime("2002:007")

rows = []
for event_func, name in (
    ("safe_suns", "Safe mode"),
    ("normal_suns", "Normal sun mode"),
    ("scs107s", "SCS-107"),
):
    for evt in getattr(events, event_func).filter(start, stop):
        row = {
            "State": "Definitive",
            "Date": evt.start,
            "Event": name,
            "Params": "",
            "Author": "Tom Aldcroft",
            "Reviewer": "",
            "Comment": "",
        }
        rows.append(row)

cmd_events = Table(rows)
cmd_events.sort("Date", reverse=True)

cmd_events.write("pre2002/cmd_events.csv", format="ascii.csv", overwrite=True)
