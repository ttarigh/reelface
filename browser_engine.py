"""
browser_engine.py: Playwright browser controller with anti-bot stealth & session persistence.
"""

import os
import time
import random
from typing import Optional
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page

USER_DATA_DIR = os.path.join(os.path.dirname(__file__), ".browser_session")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "WebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

class BrowserEngine:
    def __init__(self, headless: bool = False, user_data_dir: str = USER_DATA_DIR):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.playwright = None
        self.context: Optional[BrowserContext] = None
        self._is_started = False

    def start(self):
        """Launch persistent browser context."""
        if self._is_started:
            return

        os.makedirs(self.user_data_dir, exist_ok=True)
        self.playwright = sync_playwright().start()

        launch_args = [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-infobars",
            "--window-size=1280,900"
        ]

        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=self.headless,
            viewport={"width": 1280, "height": 900},
            user_agent=DEFAULT_USER_AGENT,
            locale="en-US",
            timezone_id="America/New_York",
            args=launch_args,
            bypass_csp=True,
            ignore_https_errors=True
        )

        # Inject stealth evasions into all pages
        self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
            Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
            Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
        """)

        self._is_started = True

    def new_page(self) -> Page:
        """Create a new page in the persistent context with video streaming blocked for maximum speed."""
        if not self._is_started:
            self.start()
        page = self.context.new_page()
        try:
            page.route("**/*.{mp4,webm}", lambda route: route.abort())
        except Exception:
            pass
        return page

    def human_scroll(self, page: Page, times: int = 5, min_delay: float = 1.0, max_delay: float = 2.0):
        """Perform natural-looking human scroll gestures to trigger infinite loading."""
        for _ in range(times):
            scroll_distance = random.randint(500, 1100)
            page.evaluate(f"window.scrollBy(0, {scroll_distance});")
            time.sleep(random.uniform(min_delay, max_delay))

    def close(self):
        """Cleanly close browser and playwright instances."""
        try:
            if self.context:
                self.context.close()
            if self.playwright:
                self.playwright.stop()
        except Exception:
            pass
        finally:
            self._is_started = False
