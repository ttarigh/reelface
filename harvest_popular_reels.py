import asyncio
import re
import json
import os
from playwright.async_api import async_playwright

TAGS = [
    'storytime', 'grwm', 'careeradvice', 'personalfinance', 'psychology',
    'deepthoughts', 'mindsetshift', 'booktok', 'motivationalspeech', 'lifeadvice',
    'publicspeaking', 'communication', 'philosophy', 'datingadvice', 'relationshiptips',
    'wellness', 'techreviews', 'productivity', 'confessions', 'selfimprovement',
    'historyfacts', 'businessadvice', 'startup', 'reading', 'writing',
    'traveldiaries', 'expatlife', 'fitnesstips', 'nutritionadvice', 'habits',
    'leadership', 'mentalhealth', 'parentingtips', 'moneytok', 'careergoals'
]

OUTPUT_FILE = 'fresh_popular_candidates.json'

async def harvest():
    print(f"=== Starting Fast Explore Harvester across {len(TAGS)} Tags ===")

    # Load existing seen video IDs
    seen = set()
    if os.path.exists('assets/verified_50k_manifest.json'):
        with open('assets/verified_50k_manifest.json') as f:
            m = json.load(f)
            for v in m.values():
                if v.get('video_id'): seen.add(v['video_id'])

    if os.path.exists('assets/rejected_blacklist.json'):
        with open('assets/rejected_blacklist.json') as f:
            bl = json.load(f)
            seen.update(bl.get('video_ids', []))

    all_harvested = []
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE) as f:
                all_harvested = json.load(f)
                for item in all_harvested:
                    v = item.get('video_id')
                    if v: seen.add(v)
        except: pass

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        context = await browser.new_context(viewport={'width': 800, 'height': 1200})
        page = await context.new_page()

        for tag in TAGS:
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            print(f"Scraping #{tag} -> {url}...", end=" ", flush=True)
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=10000)
                await asyncio.sleep(1.0)

                tag_vids = set()
                for _ in range(4):
                    content = await page.content()
                    for v in re.findall(r'/(?:reel|p)/([a-zA-Z0-9_-]{10,12})', content):
                        if v not in seen:
                            seen.add(v)
                            tag_vids.add(v)
                            all_harvested.append({'video_id': v, 'tag': tag})
                    await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
                    await asyncio.sleep(1.0)

                print(f"Found {len(tag_vids)} fresh reels! (Total pool: {len(all_harvested)})", flush=True)

                # Save incrementally
                with open(OUTPUT_FILE, 'w') as f:
                    json.dump(all_harvested, f, indent=2)

            except Exception as e:
                print(f"Failed: {e}")

        await browser.close()

    print(f"\n✓ Harvest Complete! Saved {len(all_harvested)} fresh candidate reels to {OUTPUT_FILE}!")

if __name__ == '__main__':
    asyncio.run(harvest())
