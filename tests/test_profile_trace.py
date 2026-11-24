import pytest

from ep.archive_profile import inspect_events


def test_each_profile_step_requires_device_execution():
    events = [{"ph": "M", "pid": 2, "name": "process_name", "args": {"name": "Ascend Hardware"}}]
    for step in range(3):
        for name, offset, duration in [(f"ProfilerStep#{step+1}", 0, 9),
                                       ("ep_forward", 1, 2), ("ep_backward", 4, 2)]:
            events.append(dict(ph="X", name=name, ts=step*10+offset, dur=duration, pid=1))
        events.append(dict(ph="X", name="kernel", ts=step*10+2, dur=3, pid=2))
    assert inspect_events(events, "rank0")["device_events"] == 3
    with pytest.raises(ValueError, match="incomplete ProfilerStep#3"):
        inspect_events(events[:-1], "rank0")
