import pytest
from browser.controller import BrowserController

def test_rejects_non_http_url():
    with pytest.raises(ValueError):
        BrowserController().open_url("file:///C:/secret.txt")

def test_rejects_bad_search_engine():
    with pytest.raises(ValueError):
        BrowserController().search("hello", "javascript:alert(1)")
