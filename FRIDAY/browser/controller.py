from __future__ import annotations
import time
import webbrowser
from urllib.parse import quote_plus

class BrowserController:
    """Conservative browser navigation helpers."""

    ALLOWED_SCHEMES = {"http", "https"}

    def open_url(self, url: str) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Only http/https URLs are allowed.")
        webbrowser.open(url)

    def search(self, query: str, engine: str = "https://www.google.com/search?q=") -> None:
        if not engine.startswith(("http://", "https://")):
            raise ValueError("Search engine must use http/https.")
        webbrowser.open(engine + quote_plus(query))

    def wait_for_page(self, seconds: float = 2.0) -> None:
        time.sleep(max(0.0, min(seconds, 10.0)))
