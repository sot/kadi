# %%
import numpy as np
from cxotime import CxoTime
import astropy.units as u
from kadi.commands import observations
from kadi import events

# %%
def get_trend_observations(start=None, stop=None, require_starcat=False):
    pad = 2 * u.day
    obss = observations.get_observations(
        start=CxoTime(start) - pad,
        stop=CxoTime(stop) + pad)
    starcats = observations.get_starcats(start=CxoTime(start) - pad,
                                     stop=stop)
    s_dates = np.array([cat.date for cat in starcats])
    manvrs = events.manvrs.filter(start=CxoTime(start) - pad, stop=stop)
    m_times = np.array([m.tstart for m in manvrs])
    for i in range(1, len(obss) - 1):
        obs = obss[i]
        obs['prev_obs_start'] = obss[i-1]['obs_start']
        obs['next_obs_start'] = obss[i+1]['obs_start']
        # get the matching manvr
        m_idx = int(np.argmin(np.abs(CxoTime(obs['manvr_start']).secs - m_times)))
        m = manvrs[m_idx]
        if abs(CxoTime(obs['manvr_start']).secs - m.tstart) < 60:
            obs['manvr'] = m

        # get the matching starcat
        if 'starcat_date' in obs:
            s_idx = np.nonzero(s_dates == obs['starcat_date'])[0]
            if len(s_idx) == 1:
                obs['starcat'] = starcats[s_idx[0]]

    for obs in obss:
        if 'manvr' in obs:
            obs['npnt_start'] = obs['manvr'].npnt_start
            obs['npnt_stop'] = obs['manvr'].npnt_stop

    out = []
    for obs in obss:
        if require_starcat and 'starcat' not in obs:
            continue
        if obs['obs_stop'] > start and obs['obs_start'] < stop:
            out.append(obs)
    return out

# %%



