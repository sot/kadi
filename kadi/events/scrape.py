# Licensed under a 3-clause BSD style license - see LICENSE.rst
import re

from bs4 import BeautifulSoup as parse_html

from Chandra.Time import DateTime
from .. import occweb

REPLACES = (
    (r"&#150", "-"),
    (r"&amp;", "&"),
    (r"&nbsp;", " "),
    (r"&quot;", '"'),
    (r"&gt;", ">"),
    (r"&lt;", "<"),
    (r"&#14[678];", ""),
)

# Cache of events dicts to prevent hitting the web page or re-parsing multiple times
# (which is also slow).
evts_cache = {}


def cleantext(text):
    """
    Clean up some HTML or unicode special characters and replace with ASCII equivalents.
    """
    # Convert any non-ASCII characters to XML like '&#40960;abcd&#1972;'
    text = text.encode("ascii", "xmlcharrefreplace").decode("ascii")
    lines = [x.strip() for x in text.splitlines()]
    text = " ".join(lines)
    for match, out in REPLACES:
        text = re.sub(match, out, text)
    text = text.strip()  # some spaces may have been generated

    return text


def get_table_rows(table):
    """
    Use BeautifulSoup to extract table entries as a list of lists.
    """
    rows = []
    trs = table.findAll("tr")
    for tr in trs:
        tds = tr.findAll("td")
        if "colspan" in dict(tds[0].attrs):
            continue
        vals = tuple(cleantext(td.text) for td in tds)
        rows.append(vals)

    return rows


# TODO refactor these two mostly-similar routines


def get_fdb_major_events():
    """
    Get the FDB major events.

    Some events in the early mission have multiple CAPs, which causes run-together CAP
    numbers.
    """
    page = "fdb_major_events"
    if page in evts_cache:
        return evts_cache[page]

    html = occweb.get_url(page)
    soup = parse_html(html, "lxml")
    table = soup.find("table")
    rows = get_table_rows(table)
    evts = []

    mons = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
    for i, row_vals in enumerate(rows[:0:-1]):
        try:
            mm, dd, yy = (int(x) for x in row_vals[0].split("/"))
        except ValueError:
            continue
        yyyy = yy + (1900 if yy > 97 else 2000)
        start = DateTime("{}{}{:02d} at 12:00:00".format(yyyy, mons[mm - 1], dd))
        notes = []
        if row_vals[2]:
            notes.append("CAP {}".format(row_vals[2]))
        if row_vals[3]:
            notes.append("FSW PR {}".format(row_vals[3]))
        evt = dict(
            start=start.date[:8],
            date="{}-{}-{:02d}".format(yyyy, mons[mm - 1], dd),
            tstart=start.secs + i + 1000,  # add i + 1000 for later sorting
            descr=row_vals[1],
            note=", ".join(notes),
            source="FDB",
        )
        evts.append(evt)

    evts_cache[page] = evts

    return evts


def get_fot_major_events():
    """
    Get the FOT major events.

    There is a known issue with words/phrases enclosed in <span class=SpellE></span>
    which ends up causing words run together.  E.g. Lunar eclipse 2000:302 that
    "Required Solar arrayoffpointing".
    """
    page = "fot_major_events"
    if page in evts_cache:
        return evts_cache[page]

    html = occweb.get_url(page)
    soup = parse_html(html, "lxml")
    table = soup.find("table", attrs={"class": "MsoNormalTable"})
    rows = get_table_rows(table)
    evts = []

    for i, row_vals in enumerate(rows[1:]):
        start = DateTime(row_vals[0])
        caldate = start.caldate
        evt = dict(
            start=start.date[:8],
            date="{}-{}-{}".format(caldate[:4], caldate[4:7], caldate[7:9]),
            tstart=start.secs + i,  # add i for later sorting
            descr=row_vals[1],
            note=row_vals[2],
            source="FOT",
        )
        evts.append(evt)

    evts_cache[page] = evts

    return evts
