from vision.client import parse_target_response

def test_parser_accepts_json():
    data = parse_target_response('{"summary":"browser","targets":[{"label":"YouTube","x":10,"y":20,"confidence":0.95}]}')
    assert data["targets"][0]["x"] == 10

def test_parser_extracts_wrapped_json():
    data = parse_target_response('Here is the result: {"summary":"ok","targets":[]}')
    assert data["summary"] == "ok"
