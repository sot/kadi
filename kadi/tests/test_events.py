from .. import events


def test_query_event():
    intervals = (events.manvrs & events.tsc_moves).intervals('2012:001', '2012:002')
    assert intervals == [('2012:001:18:21:31.715', '2012:001:18:22:04.515'),
                         ('2012:002:02:53:03.804', '2012:002:02:54:50.917')]
