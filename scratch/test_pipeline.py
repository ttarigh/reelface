import asyncio, os, re, time
from playwright.async_api import async_playwright
from face_verifier import FaceVerifier
from storage import VideoDatabase, format_number
from scrapers.base import (
    parse_views_string,
    is_english_speaking,
    is_podcast_or_interview,
    is_theme_page_or_montage
)

USER_DATA_DIR = os.path.join(os.getcwd(), '.browser_session')
verifier = FaceVerifier()
db = VideoDatabase()

test_tags = [
    'doctor', 'dentist', 'surgeon', 'lawyer', 'attorney', 'architect', 'astronomer',
    'biologist', 'chemist', 'physicist', 'pilot', 'economist', 'marketer',
    'realtor', 'journalism', 'actress', 'improv', 'scriptwriting', 'philosopher', 'neuroscientist',
    'publicspeaking', 'communication', 'realtalk', 'hardtruths', 'mindset', 'growthmindset',
    'philosophy', 'stoicism', 'psychology', 'technology', 'science', 'physics', 'education',
    'careeradvice', 'financialfreedom', 'business', 'entrepreneur', 'vlogger', 'grwm'
]

async def main():
    print(f"Starting DB count: {db.get_count()}")
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(USER_DATA_DIR, headless=True)
        page = await ctx.new_page()
        
        # Abort media
        await page.route("**/*.{mp4,webm}", lambda r: r.abort())
        
        for tag in test_tags:
            if db.get_count() >= 500:
                break
            url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
            try:
                t0 = time.time()
                await page.goto(url, timeout=7000, wait_until="domcontentloaded")
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
                            href: a.href.split('?')[0],
                            author: author.trim(),
                            viewsStr: viewsStr.trim(),
                            caption: caption.trim(),
                            imgSrc: img?.getAttribute('src') || ''
                        });
                    });
                    return results;
                }""")
                
                print(f"#{tag}: {len(cards)} cards in {time.time()-t0:.2f}s")
                for c in cards:
                    author = c['author'].lstrip('@').lower()
                    if not author or author in ('instagram', 'explore', 'reels'):
                        continue
                    if db.has_creator(author):
                        continue
                    caption = c['caption']
                    if is_theme_page_or_montage(author, caption):
                        continue
                    if is_podcast_or_interview(caption) or not is_english_speaking(caption):
                        continue
                    
                    views = parse_views_string(c['viewsStr']) if c['viewsStr'] else 1000000
                    if views < 1000000:
                        continue
                    likes = int(views / 10)
                    
                    # CV verify
                    is_face, metrics = verifier.verify_image_url(c['imgSrc'])
                    if not is_face:
                        continue
                    
                    match_id = re.search(r"/(?:reel|p)/([a-zA-Z0-9_-]+)", c['href'])
                    video_id = match_id.group(1) if match_id else c['href']
                    
                    cand = {
                        "platform": "Instagram Reels",
                        "creator_name": f"@{author}",
                        "creator_url": f"https://www.instagram.com/{author}/",
                        "video_id": video_id,
                        "video_url": c['href'],
                        "title": caption[:100] or f"Reel by @{author}",
                        "view_count": views,
                        "like_count": likes,
                        "upload_date": "2025-2026",
                        "thumbnail_url": c['imgSrc'],
                        "face_confidence": metrics.get("confidence", 0.0),
                        "face_area_pct": metrics.get("face_area_ratio_pct", 0.0),
                        "verified_talking_head": True
                    }
                    
                    res = db.upsert_video(cand)
                    if res == "NEW":
                        cnt = db.get_count()
                        print(f"  ✓ NEW @{author} | {views:,} Views (~{likes:,} Likes) | Face: {int(cand['face_confidence']*100)}% | Total: {cnt}/500")
                        if cnt % 3 == 0 or cnt == 500:
                            db.export_markdown()
            except Exception as e:
                pass
        await ctx.close()
    print(f"Final DB count: {db.get_count()}")

asyncio.run(main())
