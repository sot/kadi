from pathlib import Path

import numpy as np
import tables

# Use snapshot from aug08 before the last update that broke things.
with tables.open_file("cmds_aug08.h5") as h5:
    cmds = h5.root.data[:]

print(cmds.dtype)
# [('idx', '<u2'), ('date', 'S21'), ('type', 'S12'), ('tlmsid', 'S10'),
#  ('scs', 'u1'), ('step', '<u2'), ('timeline_id', '<u4'), ('vcdu', '<i4')]

new_dtype = [
    ("idx", "<i4"),
    ("date", "S21"),
    ("type", "S12"),
    ("tlmsid", "S10"),
    ("scs", "u1"),
    ("step", "<u2"),
    ("timeline_id", "<u4"),
    ("vcdu", "<i4"),
]
new_cmds = cmds.astype(new_dtype)

for name in cmds.dtype.names:
    assert np.all(cmds[name] == new_cmds[name])

cmds_h5 = Path("cmds.h5")
if cmds_h5.exists():
    cmds_h5.unlink()

with tables.open_file("cmds.h5", mode="a") as h5:
    h5.create_table(h5.root, "data", new_cmds, "cmds", expectedrows=2e6)

# Make sure the new file is really the same except the dtype
with tables.open_file("cmds.h5") as h5:
    new_cmds = h5.root.data[:]

for name in cmds.dtype.names:
    assert np.all(cmds[name] == new_cmds[name])
    if name != "idx":
        assert cmds[name].dtype == new_cmds[name].dtype

assert new_cmds["idx"].dtype.str == "<i4"
