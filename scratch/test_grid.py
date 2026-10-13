import asyncio, os, re
from playwright.async_api import async_playwright
from face_verifier import FaceVerifier
from scrapers.base import parse_views_string, is_english_speaking, is_podcast_or_interview, is_theme_page_or_montage

USER_DATA_DIR = os.path.join(os.getcwd(), '.browser_session')
verifier = FaceVerifier()

async def main():
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(USER_DATA_DIR, headless=True)
        page = await ctx.new_page()
        await page.goto('https://www.instagram.com/popular/doctor/?utm_source=explore_tag', timeout=8000, wait_until='domcontentloaded')
        
        cards = await page.evaluate(r"""() => {
            const results = [];
            const anchors = document.querySelectorAll('a[href*="/reel/"], a[href*="/p/"]');
            anchors.forEach(a => {
                const img = a.querySelector('img');
                const text = a.innerText || '';
                const parts = text.split('\n');
                const author = parts[0] || '';
                const viewsStr = parts[1] || '';
                const caption = parts.slice(2).join(' ') || img?.getAttribute('alt') || '';
                results.push({
                    href: a.href,
                    author: author.trim(),
                    viewsStr: viewsStr.trim(),
                    caption: caption.trim(),
                    imgSrc: img?.getAttribute('src') || ''
                });
            });
            return results;
        }""")
        
        print(f"Extracted {len(cards)} cards from grid:")
        for c in cards:
            views = parse_views_string(c['viewsStr']) if c['viewsStr'] else 1000000
            likes = int(views / 10)
            is_face, metrics = verifier.verify_image_url(c['imgSrc']) if c['imgSrc'] else (False, {})
            print(f"  @{c['author']} | Views: {views:,} (~{likes:,} likes) | Face: {is_face} ({metrics.get('confidence',0)*100:.0f}%) | {c['caption'][:40]}")
        await ctx.close()

asyncio.run(main())
