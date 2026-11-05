from hepose.live import LatestFrame, live_options


def test_latest_slot_discards_only_unsent_frames():
    slot = LatestFrame()
    slot.put("first")
    assert slot.take() == "first"
    slot.put("second")
    slot.put("third")
    assert slot.take() == "third"
    assert slot.take() is None
    assert slot.dropped == 1


def test_live_explicit_values_override_config_and_defaults():
    from hepose.cli import parser

    args = parser().parse_args(
        ["live", "--model", "yolo26", "--fps", "20", "--remote", "lab", "--remote-root", "/project"]
    )
    options = live_options(args, {"fps": 12, "width": 800, "remote": "other"})
    assert options.fps == 20
    assert options.width == 800
    assert options.remote == "lab"
