from .. import events


def test_query_event_intervals():
    intervals = (events.manvrs & events.tsc_moves).intervals('2012:001', '2012:002')
    assert intervals == [('2012:001:18:21:31.715', '2012:001:18:22:04.515'),
                         ('2012:002:02:53:03.804', '2012:002:02:54:50.917')]


def test_basic_query():
    rad_zones = events.rad_zones.filter('2013:001', '2013:007')
    assert str(rad_zones).splitlines() == [
        '<RadZone: 1852 2013:003:16:19:36 2013:004:02:21:34 dur=36.1 ksec>',
        '<RadZone: 1853 2013:006:08:22:22 2013:006:17:58:48 dur=34.6 ksec>']
    assert rad_zones.table.pformat(max_width=-1) == [
        '        start                  stop             tstart        tstop          dur      orbit orbit_num        perigee       ',
        '--------------------- --------------------- ------------- ------------- ------------- ----- --------- ---------------------',
        '2013:003:16:19:36.289 2013:004:02:21:34.289 473617243.473 473653361.473 36118.0000001  1852      1852 2013:003:22:29:59.302',
        '2013:006:08:22:22.982 2013:006:17:58:48.982 473847810.166 473882396.166 34586.0000001  1853      1853 2013:006:13:58:21.389']


def test_zero_length_query():
    rad_zones = events.rad_zones.filter('2013:001', '2013:002')
    assert len(rad_zones) == 0
    assert len(rad_zones.table) == 0


def test_short_query():
    """
    Short duration queries that test that filter will return partially
    included intervals.
    """
    dwells = events.dwells.filter('2012:002:00:00:00', '2012:002:00:00:01')
    assert len(dwells) == 1
    dwells = events.dwells.filter('2012:001:18:40:00', '2012:001:18:42:00')
    assert len(dwells) == 1
    dwells = events.dwells.filter('2012:002:02:49:00', '2012:002:02:50:00')
    assert len(dwells) == 1
