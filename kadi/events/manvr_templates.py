# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
Define templates of previously seen maneuver sequences.  These cover
sequences seen at least twice as of ~Mar 2012.  See:

http://occweb.cfa.harvard.edu/twiki/Aspect/RarePcadSequences
"""


def get_manvr_templates():
    """Return a list of (name, template) pairs, where each template is a
    list of strings like <msid>_<val0>_<val>.  This is ordered by frequency.
    """
    templates = []
    dat_str = get_dat_str()
    tables = dat_str.split('---------------------------------------------------')
    for table in tables:
        lines = [x.split('|')[0].strip() for x in table.splitlines() if x.strip()]
        name = lines[0].split()[0]
        templates.append((name, lines[1:]))
    return templates


def get_dat_str():
    dat = """
normal : Instances in mission: 17238
aofattmd_MNVR_STDY | 2012:086:18:15:50.729 -10214.1
aofattmd_STDY_MNVR | 2012:086:20:40:30.429  -1534.4
aofattmd_MNVR_STDY | 2012:086:21:06:05.880      1.0
aopcadmd_NMAN_NPNT | 2012:086:21:06:21.255     16.4
aoacaseq_KALM_AQXN | 2012:086:21:06:21.255     16.4
aoacaseq_AQXN_GUID | 2012:086:21:07:00.205     55.4
aoacaseq_GUID_KALM | 2012:086:21:07:45.305    100.5
aopcadmd_NPNT_NMAN | 2012:086:22:44:20.655   5895.8
aofattmd_STDY_MNVR | 2012:086:22:44:30.905   5906.1
---------------------------------------------------
nman_dwell : Instances in mission: 1143
aofattmd_MNVR_STDY | 2009:012:12:37:56.088 -13204.1
aofattmd_STDY_MNVR | 2009:012:15:45:45.963  -1934.2
aofattmd_MNVR_STDY | 2009:012:16:18:01.164      1.0
aopcadmd_NMAN_NPNT | 2009:012:16:18:16.539     16.4
aoacaseq_KALM_AQXN | 2009:012:16:18:16.539     16.4
aoacaseq_AQXN_GUID | 2009:012:16:18:52.414     52.3
aoacaseq_GUID_KALM | 2009:012:16:19:41.614    101.5
aopcadmd_NPNT_NMAN | 2009:012:19:08:59.364  10259.2
aopcadmd_NMAN_NPNT | 2009:012:19:16:35.489  10715.4
aoacaseq_KALM_AQXN | 2009:012:19:16:35.489  10715.4
aoacaseq_AQXN_GUID | 2009:012:19:17:17.514  10757.4
aoacaseq_GUID_KALM | 2009:012:19:17:38.014  10777.9
aopcadmd_NPNT_NMAN | 2009:012:19:23:59.314  11159.2
aofattmd_STDY_MNVR | 2009:012:19:24:09.564  11169.4
---------------------------------------------------
interm_att : Instances in mission: 613
aofattmd_MNVR_STDY | 2012:033:21:46:41.237  -1439.1
aofattmd_STDY_MNVR | 2012:033:22:06:40.487   -239.9
aofattmd_MNVR_STDY | 2012:033:22:10:41.362      1.0
aofattmd_STDY_MNVR | 2012:033:22:25:41.312    901.0
---------------------------------------------------
two_acq : Instances in mission: 85
aofattmd_MNVR_STDY | 2011:117:07:48:26.771  -6546.7
aofattmd_STDY_MNVR | 2011:117:08:59:36.922  -2276.5
aofattmd_MNVR_STDY | 2011:117:09:37:34.472      1.0
aopcadmd_NMAN_NPNT | 2011:117:09:37:49.847     16.4
aoacaseq_KALM_AQXN | 2011:117:09:37:49.847     16.4
aoacaseq_AQXN_GUID | 2011:117:09:38:27.772     54.3
aoacaseq_GUID_KALM | 2011:117:09:39:16.972    103.5
aoacaseq_KALM_AQXN | 2011:117:09:40:24.622    171.2
aoacaseq_AQXN_GUID | 2011:117:09:40:59.472    206.0
aoacaseq_GUID_KALM | 2011:117:09:41:15.872    222.4
aopcadmd_NPNT_NMAN | 2011:117:19:30:14.799  35561.4
aofattmd_STDY_MNVR | 2011:117:19:30:24.024  35570.6
---------------------------------------------------
three_acq : Instances in mission: 39
aofattmd_MNVR_STDY | 2012:057:22:55:49.122 -10777.9
aofattmd_STDY_MNVR | 2012:058:01:31:55.573  -1411.4
aofattmd_MNVR_STDY | 2012:058:01:55:28.023      1.0
aopcadmd_NMAN_NPNT | 2012:058:01:55:43.398     16.4
aoacaseq_KALM_AQXN | 2012:058:01:55:43.398     16.4
aoacaseq_AQXN_GUID | 2012:058:01:56:23.373     56.4
aoacaseq_GUID_KALM | 2012:058:01:56:47.973     81.0
aoacaseq_KALM_AQXN | 2012:058:03:26:07.698   5440.7
aoacaseq_AQXN_GUID | 2012:058:03:28:50.673   5603.7
aoacaseq_GUID_KALM | 2012:058:03:29:11.173   5624.2
aoacaseq_KALM_AQXN | 2012:058:03:29:52.173   5665.2
aoacaseq_AQXN_GUID | 2012:058:03:32:36.173   5829.2
aoacaseq_GUID_KALM | 2012:058:03:32:56.673   5849.7
aopcadmd_NPNT_NMAN | 2012:058:07:44:33.899  20946.9
aofattmd_STDY_MNVR | 2012:058:07:44:44.149  20957.2
---------------------------------------------------
two_acq_nman : Instances in mission: 13
aofattmd_MNVR_STDY | 2006:151:07:31:56.674 -15786.0
aofattmd_STDY_MNVR | 2006:151:11:46:27.124   -515.6
aofattmd_MNVR_STDY | 2006:151:11:55:03.724      1.0
aopcadmd_NMAN_NPNT | 2006:151:11:55:20.124     17.4
aoacaseq_KALM_AQXN | 2006:151:11:55:20.124     17.4
aoacaseq_AQXN_GUID | 2006:151:11:56:01.124     58.4
aoacaseq_GUID_KALM | 2006:151:11:56:21.624     78.9
aoacaseq_KALM_AQXN | 2006:151:11:57:53.874    171.2
aoacaseq_AQXN_GUID | 2006:151:11:58:32.824    210.1
aoacaseq_GUID_KALM | 2006:151:11:58:53.324    230.6
aopcadmd_NPNT_NMAN | 2006:151:13:14:36.125   4773.4
aopcadmd_NMAN_NPNT | 2006:151:13:22:12.250   5229.6
aoacaseq_KALM_AQXN | 2006:151:13:22:12.250   5229.6
aoacaseq_AQXN_GUID | 2006:151:13:22:52.225   5269.5
aoacaseq_GUID_KALM | 2006:151:13:23:04.525   5281.8
aopcadmd_NPNT_NMAN | 2006:151:13:29:36.075   5673.4
aofattmd_STDY_MNVR | 2006:151:13:29:46.325   5683.6
---------------------------------------------------
three_acq_nman : Instances in mission: 3
aofattmd_MNVR_STDY | 2000:167:00:56:12.805  -2248.9
aofattmd_STDY_MNVR | 2000:167:01:00:52.630  -1969.0
aofattmd_MNVR_STDY | 2000:167:01:33:42.680      1.0
aopcadmd_NMAN_NPNT | 2000:167:01:33:58.055     16.4
aoacaseq_KALM_AQXN | 2000:167:01:33:58.055     16.4
aoacaseq_AQXN_GUID | 2000:167:01:34:26.755     45.1
aoacaseq_GUID_KALM | 2000:167:01:35:44.655    123.0
aopcadmd_NPNT_NMAN | 2000:167:01:44:56.105    674.5
aopcadmd_NMAN_NPNT | 2000:167:02:22:28.030   2926.4
aoacaseq_KALM_AQXN | 2000:167:02:22:28.030   2926.4
aoacaseq_AQXN_GUID | 2000:167:02:22:53.655   2952.0
aoacaseq_GUID_KALM | 2000:167:02:23:10.055   2968.4
aopcadmd_NPNT_NMAN | 2000:167:06:09:36.431  16554.8
aopcadmd_NMAN_NPNT | 2000:167:06:20:02.706  17181.1
aoacaseq_KALM_AQXN | 2000:167:06:20:02.706  17181.1
aoacaseq_AQXN_GUID | 2000:167:06:20:29.356  17207.7
aoacaseq_GUID_KALM | 2000:167:06:20:49.856  17228.2
aopcadmd_NPNT_NMAN | 2000:167:06:24:41.506  17459.9
aofattmd_STDY_MNVR | 2000:167:06:24:51.756  17470.1
---------------------------------------------------
nsun_anom : Instances in mission: 3 (ANOMALY)
aofattmd_MNVR_STDY | 2000:027:00:56:48.808  -7299.0
aofattmd_STDY_MNVR | 2000:027:02:54:29.008   -238.8
aofattmd_MNVR_STDY | 2000:027:02:58:28.858      1.0
aopcadmd_NMAN_NPNT | 2000:027:02:58:44.233     16.4
aoacaseq_KALM_AQXN | 2000:027:02:58:44.233     16.4
aoacaseq_AQXN_GUID | 2000:027:02:59:13.958     46.1
aoacaseq_GUID_KALM | 2000:027:02:59:22.158     54.3
aopcadmd_NPNT_NMAN | 2000:027:13:32:54.284  38066.5
aopcadmd_NMAN_NSUN | 2000:027:13:33:39.384  38111.6
aofattmd_STDY_NULL | 2000:027:13:33:39.384  38111.6
aopcadmd_NSUN_NMAN | 2000:027:13:50:39.259  39131.4
aofattmd_NULL_STDY | 2000:027:13:50:39.259  39131.4
aofattmd_STDY_MNVR | 2000:027:13:54:31.935  39364.1
---------------------------------------------------
bsh_anom : Instances in mission: 3 (ANOMALY)
aofattmd_MNVR_STDY | 2001:265:10:30:10.369  -4564.3
aofattmd_STDY_MNVR | 2001:265:11:37:05.294   -549.4
aofattmd_MNVR_STDY | 2001:265:11:46:15.719      1.0
aopcadmd_NMAN_NPNT | 2001:265:11:46:31.094     16.4
aoacaseq_KALM_AQXN | 2001:265:11:46:31.094     16.4
aoacaseq_AQXN_BRIT | 2001:265:11:47:36.694     82.0
aoacaseq_BRIT_KALM | 2001:265:11:48:50.494    155.8
aopcadmd_NPNT_NMAN | 2001:266:02:23:09.996  52615.3
aofattmd_STDY_MNVR | 2001:266:02:36:47.946  53433.3
---------------------------------------------------
four_acq: Instances in mission: 2
aofattmd_MNVR_STDY | 2000:157:04:11:15.024 -11103.8
aofattmd_STDY_MNVR | 2000:157:07:02:44.999   -813.9
aofattmd_MNVR_STDY | 2000:157:07:16:19.874      1.0
aopcadmd_NMAN_NPNT | 2000:157:07:16:36.274     17.4
aoacaseq_KALM_AQXN | 2000:157:07:16:36.274     17.4
aoacaseq_AQXN_GUID | 2000:157:07:17:02.924     44.1
aoacaseq_GUID_KALM | 2000:157:07:17:15.224     56.4
aoacaseq_KALM_AQXN | 2000:157:13:01:55.625  20736.8
aoacaseq_AQXN_GUID | 2000:157:13:05:24.725  20945.9
aoacaseq_GUID_KALM | 2000:157:13:05:37.025  20958.2
aoacaseq_KALM_AQXN | 2000:157:13:07:57.450  21098.6
aoacaseq_AQXN_GUID | 2000:157:13:10:15.825  21237.0
aoacaseq_GUID_KALM | 2000:157:13:12:47.525  21388.7
aoacaseq_KALM_AQXN | 2000:157:13:13:20.325  21421.5
aoacaseq_AQXN_GUID | 2000:157:13:13:49.025  21450.2
aoacaseq_GUID_KALM | 2000:157:13:14:46.425  21507.6
aopcadmd_NPNT_NMAN | 2000:158:03:59:48.602  74609.8
aofattmd_STDY_MNVR | 2000:158:03:59:58.852  74620.0
---------------------------------------------------
delayed_npnt : Instances in mission: 2
aofattmd_MNVR_STDY | 2000:014:13:41:19.444 -17370.7
aofattmd_STDY_MNVR | 2000:014:18:00:38.944  -1811.2
aofattmd_MNVR_STDY | 2000:014:18:30:51.144      1.0
aoacaseq_KALM_AQXN | 2000:014:18:32:03.919     73.8
aopcadmd_NMAN_NPNT | 2000:014:18:32:08.019     77.9
aoacaseq_AQXN_GUID | 2000:014:18:32:28.519     98.4
aoacaseq_GUID_KALM | 2000:014:18:33:46.419    176.3
aopcadmd_NPNT_NMAN | 2000:015:05:46:41.171  40551.1
aofattmd_STDY_MNVR | 2000:015:05:46:51.421  40561.3
"""
    return dat

if __name__ == '__main__':
    get_manvr_templates()
