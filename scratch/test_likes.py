import asyncio, os, re
from playwright.async_api import async_playwright
from scrapers.base import parse_views_string

USER_DATA_DIR = os.path.join(os.getcwd(), '.browser_session')

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(USER_DATA_DIR, headless=True)
        page = await ctx.new_page()
        await page.goto("https://www.instagram.com/popular/dentist/?utm_source=explore_tag", timeout=8000)
        links = await page.evaluate('''() => {
            return Array.from(document.querySelectorAll('a[href*="/reel/"], a[href*="/p/"]')).map(a => a.href);
        }''')
        for l in links[:6]:
            await page.goto(l, timeout=6000)
            desc = await page.evaluate('''() => document.querySelector('meta[property="og:description"]')?.content''') or ""
            m = re.search(r'([\d\.,]+[kmb]?)\s+likes', desc, re.IGNORECASE)
            likes = parse_views_string(m.group(1)) if m else 0
            code = l.split('/')[-2]
            print(f'{code} -> {likes:,} likes | {desc[:60]}')
        await b.close()

asyncio.run(main())
