import asyncio
import os
import re
import json
import time
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFIED_MANIFEST = os.path.join(BASE_DIR, 'assets', 'verified_50k_manifest.json')
CREATORS_JSON = os.path.join(BASE_DIR, 'assets', 'creators.json')
WEB_CREATORS = os.path.join(BASE_DIR, 'web', 'assets', 'creators.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'talking_head_photos_100')
BLACKLIST_PATH = os.path.join(BASE_DIR, 'assets', 'rejected_blacklist.json')
MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
LBF_MODEL = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")

TARGET_PENDING = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("scratch", exist_ok=True)

ctx_ssl = ssl._create_unverified_context()

INDIC_SCRIPT_REGEX = re.compile(r'[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F\u0A80-\u0AFF\u0980-\u09FF\u0A00-\u0A7F\u0D00-\u0D7F]')

CURATED_WESTERN_TAGS = [
    # 1. Western & Global Home Renovation, Remodel & Craft
    'hgtv', 'thisoldhouse', 'fixerupperstyle', 'modernfarmhouse', 'farmhouserenovation',
    'traditionalwoodworking', 'timberframe', 'customwoodworking', 'americanwoodworker',
    'finehomebuilding', 'joannagaines', 'diyhomerenovation', 'diywoodworking', 'diydecor',
    'renovationjourney', 'cottagestyle', 'countryliving', 'homemakeover', 'roomtransformation',
    'restorationhardware', 'woodcraft', 'interiordesignideas', 'remodelista', 'housegoals',

    # 2. Silicon Valley, US/UK Creative Tech, AI & Design
    'siliconvalley', 'nyctech', 'sftech', 'designsystems', 'techstartup', 'ycombinator',
    'productdesign', 'figmadesign', 'indiehacker', 'creativecoder', 'creativecodeart',
    'touchdesigner', 'interactiondesign', 'designinspiration', 'webdesignerlife',
    'womenintech', 'techcommunity', 'coderlife', 'programmerlife', 'uxdesign', 'uidesign',
    'creativecoding', 'generativeai', 'aiart', 'blender3d', 'motiondesign',

    # 3. US/UK Creator Life & Instagram Growth Strategy
    'creatorculture', 'growonreels', 'socialmediatipsdaily', 'contentcreatoruk', 'creatorsusa',
    'contentcreationtips', 'influencerstrategy', 'contentqueen', 'socialmediacoach',
    'igtipsforcreators', 'creatortips', 'instagrammarketingstrategy', 'reelstipsandtricks',
    'contentstrategist', 'personalbrandingcoach', 'contentcreatortips', 'creatorcommunity',

    # 4. Solo High-Yield Western Cultural Talking Heads
    'corporatebaddie', 'corporatemillennial', 'corporategirlies', 'datingadviceforwomen',
    'storytimeuk', 'storytimeusa', 'girldinner', 'londonlife', 'nyclife', 'parisianstyle',
    'aussiecreators', 'canadiancreators', 'booktok', 'readingcommunity', 'therapistsofinstagram',
    'mentalhealthadvocate', 'careeradviceforwomen', 'negotiationskills', 'corporatehumor',
    'workplaceculture', 'girlboss', 'mindsetcoach', 'lifeadvice'
]

REGIONAL_BLOCK_KEYWORDS = [
    'india', 'indian', 'delhi', 'mumbai', 'bangalore', 'punjab', 'desi', 'hindi', 'bhai', 'vines',
    'urdu', 'pakistan', 'kolkata', 'chennai', 'hyderabad', 'bhojpuri', 'malayalam', 'kerala',
    '_in$', 'official_in', 'media_in'
]

GENERAL_BLOCKED_KEYWORDS = [
    'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
    'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
    'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
    'derm', 'dermatolog', 'dr.', 'dr_', 'dr', 'accent', 'pronounc', 'esl', 'medical', 'md', 'pharma',
    'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german', 'japanese', 'chinese',
    'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language', 'podcast', 'news', 'music', 'singer',
    'radio', 'band', 'guitar', 'vogue', 'hollywood', 'unlocked', 'official', 'memes', 'dailymemes'
]

def ear(pts):
    r_h1 = np.linalg.norm(pts[37] - pts[41])
    r_h2 = np.linalg.norm(pts[38] - pts[40])
    r_w  = np.linalg.norm(pts[36] - pts[39])
    r_ear = (r_h1 + r_h2) / (2.0 * r_w + 1e-5)
    l_h1 = np.linalg.norm(pts[43] - pts[47])
    l_h2 = np.linalg.norm(pts[44] - pts[46])
    l_w  = np.linalg.norm(pts[42] - pts[45])
    l_ear = (l_h1 + l_h2) / (2.0 * l_w + 1e-5)
    return (r_ear + l_ear) / 2.0

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0

    top_bar = fr[:int(h*0.10), :]
    bot_bar = fr[int(h*0.90):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0

    sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    primary = sorted_faces[0]
    conf = float(primary[-1])
    if conf < 0.78: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Portrait framing
    if area < 0.025 or area > 0.42: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.28: return False, 0
    if fy < int(h*0.04) or fy > int(h*0.65): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 35: return False, 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.30: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.16: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.12 or m_ratio > 0.95: return False, 0

    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

def analyze_video_url(v_url, worker_id, vid, detector, facemark):
    temp_path = f"scratch/div_w{worker_id}_{vid}.mp4"
    try:
        req = urllib.request.Request(v_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=5.0, context=ctx_ssl) as u, open(temp_path, 'wb') as f:
            f.write(u.read())

        cap = cv2.VideoCapture(temp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            if os.path.exists(temp_path): os.remove(temp_path)
            return None, -1.0

        step = max(1, total // 25)
        best_f = None
        best_s = -1.0
        for idx in range(6, total, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, fr = cap.read()
            if not ret or fr is None: continue
            ok, sc = eval_frame(fr, detector, facemark)
            if ok and sc > best_s:
                best_s = sc
                best_f = fr.copy()
        cap.release()
        if os.path.exists(temp_path): os.remove(temp_path)
        return best_f, best_s
    except Exception:
        if os.path.exists(temp_path): os.remove(temp_path)
        return None, -1.0

async def worker(worker_id, queue, browser, stop_flag, lock):
    detector, facemark = create_models()
    context = await browser.new_context(viewport={'width': 600, 'height': 850})
    page = await context.new_page()
    await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")

    processed_count = 0

    while not stop_flag['stop']:
        try:
            cand = await asyncio.wait_for(queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            if stop_flag['stop']:
                break
            continue

        if cand is None:
            queue.task_done()
            break

        processed_count += 1
        if processed_count % 25 == 0:
            try:
                await page.close()
                page = await context.new_page()
                await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")
            except: pass

        vid = cand.get('video_id')
        cat = cand.get('category', 'general')

        try:
            embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
            await page.goto(embed_url, wait_until='domcontentloaded', timeout=6000)
            await asyncio.sleep(0.35)

            data = await page.evaluate(r'''() => {
                let likes = null;
                let handle = null;
                let fullText = document.body ? document.body.innerText : '';
                const els = Array.from(document.querySelectorAll('a, span, div'));
                for (const el of els) {
                    const m = el.innerText && el.innerText.match(/([0-9,.]+)\s*([KkMmBbn]?)\s+likes/i);
                    if (m && likes === null) {
                        let val = parseFloat(m[1].replace(/,/g, ''));
                        const unit = (m[2] || '').toUpperCase();
                        if (unit === 'K') val *= 1000;
                        else if (unit === 'M') val *= 1000000;
                        likes = Math.round(val);
                    }
                }
                for (const a of document.querySelectorAll('a')) {
                    if (a.href.includes('utm_source=ig_embed') && !a.href.includes('/reel/') && !a.href.includes('/stories/') && a.innerText.trim().length > 0 && !a.innerText.includes('View') && !a.innerText.includes('more')) {
                        handle = a.innerText.trim();
                        break;
                    }
                }
                const v = document.querySelector('video');
                const img = document.querySelector('.EmbeddedMediaImage');
                return { likes, handle, video_src: v ? v.src : null, img_src: img ? img.src : null, fullText };
            }''')

            real_likes = data.get('likes')
            html = ""
            if real_likes is None:
                html = await page.content()
                m_likes = re.search(r'edge_liked_by[\\\"\':\s]+count[\\\"\':\s]+([0-9]+)', html)
                if m_likes:
                    real_likes = int(m_likes.group(1))

            if real_likes is None or real_likes < 50000:
                queue.task_done()
                continue

            raw_handle = data.get('handle')
            if not raw_handle or raw_handle.lower() in ['instagram', 'view profile', 'watch on instagram']:
                if not html: html = await page.content()
                m_auth = re.search(r'\"owner\":\s*\{\"id\":[^\}]*\"username\":\s*\"([^\"]+)\"', html)
                if m_auth: raw_handle = m_auth.group(1)

            if not raw_handle:
                queue.task_done()
                continue

            clean_handle = re.sub(r'[^a-zA-Z0-9._-]', '', raw_handle.split('\n')[0].strip())[:25]
            clean_lower = clean_handle.lower()

            # Regional balancing filter (skip Indic scripts and regional keywords)
            full_text = data.get('fullText', '')
            if INDIC_SCRIPT_REGEX.search(full_text) or INDIC_SCRIPT_REGEX.search(html):
                queue.task_done()
                continue

            if any(b in clean_lower for b in REGIONAL_BLOCK_KEYWORDS) or any(b in full_text.lower() for b in REGIONAL_BLOCK_KEYWORDS):
                queue.task_done()
                continue

            if any(b in clean_lower for b in GENERAL_BLOCKED_KEYWORDS):
                queue.task_done()
                continue

            async with lock:
                if os.path.exists(BLACKLIST_PATH):
                    with open(BLACKLIST_PATH) as bf:
                        bl_data = json.load(bf)
                    if vid in set(bl_data.get("video_ids", [])) or clean_lower in {h.lower() for h in bl_data.get("handles", [])}:
                        queue.task_done()
                        continue

                with open(VERIFIED_MANIFEST) as mf:
                    curr_manifest = json.load(mf)
                if clean_lower in {v['handle'].lower() for v in curr_manifest.values()} or vid in {v.get('video_id') for v in curr_manifest.values()}:
                    queue.task_done()
                    continue

            best_f = None
            best_s = -1.0

            img_src = data.get('img_src')
            if img_src:
                try:
                    req = urllib.request.Request(img_src, headers={'User-Agent': 'Mozilla/5.0'})
                    with urllib.request.urlopen(req, timeout=4.0, context=ctx_ssl) as u:
                        arr = np.frombuffer(u.read(), dtype=np.uint8)
                        fr = cv2.imdecode(arr, -1)
                    if fr is not None:
                        ok, sc = eval_frame(fr, detector, facemark)
                        if ok:
                            best_f = fr
                            best_s = sc
                except: pass

            v_src = data.get('video_src')
            if best_f is None and v_src:
                best_f, best_s = analyze_video_url(v_src, worker_id, vid, detector, facemark)

            if best_f is None:
                queue.task_done()
                continue

            # Atomic save under lock
            async with lock:
                with open(VERIFIED_MANIFEST) as mf:
                    manifest = json.load(mf)

                if clean_lower in {v['handle'].lower() for v in manifest.values()} or vid in {v.get('video_id') for v in manifest.values()}:
                    queue.task_done()
                    continue

                existing_ranks = [int(k) for k in manifest.keys() if str(k).isdigit()]
                next_rank = max(existing_ranks + [0]) + 1
                rank_str = f"{next_rank:04d}"
                out_name = f"{rank_str}_{clean_handle}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, best_f, [cv2.IMWRITE_JPEG_QUALITY, 95])

                reel_url = f"https://www.instagram.com/reel/{vid}/"
                manifest[rank_str] = {
                    'rank': rank_str,
                    'handle': clean_handle,
                    'video_id': vid,
                    'reelUrl': reel_url,
                    'likes': int(real_likes),
                    'category': cat,
                    'photo': out_name,
                    'score': float(round(best_s, 1)),
                    'verified': True,
                    'approved': False
                }

                tmp_mf = f"{VERIFIED_MANIFEST}.div_{time.time()}.tmp"
                with open(tmp_mf, 'w') as mf:
                    json.dump(manifest, mf, indent=2)
                os.replace(tmp_mf, VERIFIED_MANIFEST)

                creators_list = list(manifest.values())
                tmp_cf = f"{CREATORS_JSON}.div_{time.time()}.tmp"
                with open(tmp_cf, 'w') as cf:
                    json.dump(creators_list, cf, indent=2)
                os.replace(tmp_cf, CREATORS_JSON)

                if os.path.exists(os.path.dirname(WEB_CREATORS)):
                    tmp_wf = f"{WEB_CREATORS}.div_{time.time()}.tmp"
                    with open(tmp_wf, 'w') as wf:
                        json.dump(creators_list, wf, indent=2)
                    os.replace(tmp_wf, WEB_CREATORS)

                approved_total = sum(1 for v in manifest.values() if v.get('approved'))
                pending_total = sum(1 for v in manifest.values() if not v.get('approved'))
                print(f"  ★ [W{worker_id}] [{rank_str}] STAGED ({cat}): @{clean_handle} -> {real_likes:,} REAL LIKES ({reel_url}) | Approved: {approved_total}, Pending: {pending_total}/{TARGET_PENDING}", flush=True)

                if pending_total >= TARGET_PENDING:
                    print(f"\n[GOAL REACHED] Staged target of {TARGET_PENDING} pending candidates! Workers stopping...", flush=True)
                    stop_flag['stop'] = True

        except Exception as e:
            pass

        queue.task_done()

    await context.close()

async def main():
    print(f"=== Diverse & Western Finish Harvester (Targeting {TARGET_PENDING} Pending Candidates) ===\n", flush=True)

    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    seen_vids = set(v.get('video_id') for v in manifest.values() if v.get('video_id')) | blacklisted_vids

    queue = asyncio.LifoQueue()
    stop_flag = {'stop': False}
    lock = asyncio.Lock()
    num_workers = 6

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--disable-dev-shm-usage', '--no-sandbox']
        )
        crawler_page = await browser.new_page()

        workers = [asyncio.create_task(worker(i, queue, browser, stop_flag, lock)) for i in range(num_workers)]

        async def crawl_live():
            round_num = 1
            while not stop_flag['stop']:
                print(f"\n--- Starting Crawl Round {round_num} ---", flush=True)
                for tag in CURATED_WESTERN_TAGS:
                    if stop_flag['stop']: break
                    tag_url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                    try:
                        await crawler_page.goto(tag_url, wait_until='domcontentloaded', timeout=8000)
                        await asyncio.sleep(0.35)
                        for _ in range(6):
                            await crawler_page.evaluate('window.scrollBy(0, 1000)')
                            await asyncio.sleep(0.25)

                        hrefs = await crawler_page.evaluate('''() => {
                            return Array.from(document.querySelectorAll('a[href*="/reel/"]')).map(a => a.href);
                        }''')

                        added = 0
                        for h in hrefs:
                            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', h)
                            if m:
                                vid = m.group(1)
                                if vid not in seen_vids:
                                    seen_vids.add(vid)
                                    await queue.put({'video_id': vid, 'category': tag})
                                    added += 1
                        print(f"[Crawler] #{tag} -> Found {len(hrefs)} reels ({added} brand new). Queue: {queue.qsize()}", flush=True)
                    except Exception as e:
                        pass
                round_num += 1

            for _ in range(num_workers):
                await queue.put(None)

        crawler_task = asyncio.create_task(crawl_live())
        await asyncio.gather(crawler_task, *workers)
        await crawler_page.close()
        await browser.close()

    print("\n=== Diverse Harvester Completed! ===", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
