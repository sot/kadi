import time

import kadi
from kadi.commands import get_cmds
from kadi.commands.states import get_states

print(f"{kadi.__version__=}")

start, stop = "2021:001", "2022:001"
cmds = get_cmds(start, stop, scenario="flight")

t0 = time.time()
states = get_states(start, stop, scenario="flight")
print(f"get_states took {time.time() - t0:.1f} sec")

t0 = time.time()
states = get_states(start, stop, scenario="flight")
print(f"2nd get_states took {time.time() - t0:.1f} sec")