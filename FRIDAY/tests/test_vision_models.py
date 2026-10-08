from vision.models import ScreenTarget, ScreenObservation

def test_target_round_trip_values():
    target = ScreenTarget("Search", 100, 200, 0.91, "search box")
    assert target.label == "Search"
    assert target.x == 100
    assert target.y == 200
    assert target.confidence == 0.91

def test_observation_defaults():
    obs = ScreenObservation(1920, 1080)
    assert obs.targets == []
    assert obs.summary == ""
