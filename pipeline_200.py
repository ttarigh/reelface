import asyncio
import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CANDIDATES_FILE = "candidates_200.json"
MANIFEST_FILE = "talking_heads_100_manifest.json"

DIVERSE_TAGS = [
    # Tech & Software
    'softwareengineer', 'techcreator', 'codinglife', 'techfounder',
    # Filmmaking & Gear
    'filmmaking', 'cinematography', 'cameragear',
    # Fitness & Athletics
    'fitnesstrainer', 'workoutadvice', 'calisthenics',
    # Finance & Real Estate
    'personalfinance', 'realestateinvesting', 'stockmarketadvice',
    # Books & Literature
    'booktok', 'bookreview', 'writingtips',
    # Science & Space & History
    'sciencefacts', 'spacefacts', 'historybuff',
    # Food & Culinary
    'chefsoftiktok', 'culinaryarts', 'cheftips',
    # Language & Culture
    'learnenglish', 'learnspanish', 'polyglot',
    # Comedy Monologue & Satire
    'standupcomedy', 'observationalhumor',
    # Music & Audio
    'musicproducer', 'audioengineer'
]

ctx_ssl = ssl._create_unverified_context()

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def norm_handle(h):
    if not h: return ""
    clean = re.sub(r'[^a-zA-Z0-9]', '', h.lower())
    return clean[:6]

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

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    top = fr[:int(h*0.06), :]
    bot = fr[int(h*0.94):, :]
    if (np.mean(top) < 15 and np.std(top) < 10) or (np.mean(bot) < 15 and np.std(bot) < 10):
        return False, 0
        
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return False, 0
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area = (fw*fh)/(w*h)
    if area < 0.040 or area > 0.17: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.17: return False, 0
    if fy < int(h*0.10) or fy > int(h*0.50): return False, 0
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    eye_span = abs(le_x - re_x)
    if eye_span <= 14: return False, 0
    if abs((nt_x - re_x) - (le_x - nt_x)) / eye_span > 0.18: return False, 0
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    eye_ratio = ear(pts)
    if eye_ratio < 0.27: return False, 0
    
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.29 or m_ratio > 0.80: return False, 0
    
    chin_y = int(pts[8, 1])
    chin_x = int(pts[8, 0])
    neck_h = min(h - chin_y, 70)
    if neck_h > 20:
        neck = gray[chin_y:chin_y+neck_h, max(0, chin_x-40):min(w, chin_x+40)]
        if np.max(np.abs(cv2.Sobel(neck, cv2.CV_64F, 0, 1, ksize=3))) > 1050:
            return False, 0
            
    score = conf*100 + m_ratio*50 + eye_ratio*50
    return True, score

async def scrape_diverse_candidates():
    print("=== Phase 1: Scraping Diverse Explore Tags ===")
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        page = await ctx.new_page()
        reels = []
        for tag in DIVERSE_TAGS:
            try:
                url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                await page.goto(url, wait_until='domcontentloaded', timeout=4000)
                await asyncio.sleep(0.1)
                for _ in range(2):
                    await page.evaluate('window.scrollBy(0, 1500)')
                    await asyncio.sleep(0.2)
                found = await page.evaluate(r'''() => {
                    return Array.from(document.querySelectorAll('a[href*="/reel/"]')).map(a => ({
                        href: a.href.split('?')[0],
                        author: (a.innerText || '').split('\n')[0].trim()
                    }));
                }''')
                print(f"  #{tag:22}: {len(found)} reels")
                for r in found:
                    r['tag'] = tag
                reels.extend(found)
            except Exception as e:
                print(f"  #{tag:22}: Error {e}")
        await ctx.close()

    seen = set()
    unique = []
    for r in reels:
        if r['href'] not in seen and r['author']:
            seen.add(r['href'])
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
            if m:
                r['video_id'] = m.group(1)
                unique.append(r)

    with open(CANDIDATES_FILE, 'w') as f:
        json.dump(unique, f, indent=2)
    print(f"\nTotal diverse candidate reels collected: {len(unique)}\n")
    return unique

async def worker(worker_id, queue, saved_counter, seen_handles, lock):
    detector, facemark = create_models()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        while not queue.empty():
            cand = await queue.get()
            author = cand.get('author', '').split('\n')[0].strip().lstrip('@')
            clean_author = re.sub(r'[^a-zA-Z0-9._-]', '', author)[:25]
            prefix = norm_handle(author)
            
            async with lock:
                if saved_counter['count'] >= 200:
                    queue.task_done()
                    break
                if not prefix or prefix in seen_handles:
                    queue.task_done()
                    continue

            vid = cand.get('video_id')
            if not vid:
                queue.task_done()
                continue

            try:
                await page.goto(f"https://www.instagram.com/reel/{vid}/embed/", wait_until='domcontentloaded', timeout=4000)
                el = await page.wait_for_selector('video', timeout=2500)
                v_src = await el.get_attribute('src')
                if not v_src:
                    queue.task_done()
                    continue

                # Also grab real author from embed header if possible
                header_author = await page.evaluate(r'''() => {
                    const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*="/p/"], .UsernameText');
                    return el ? el.innerText.trim() : '';
                }''')
                if header_author:
                    clean_author = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25]
                    prefix = norm_handle(clean_author)

                mp4_path = f"scratch/w_{worker_id}_{vid}.mp4"
                req = urllib.request.Request(v_src, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3.5, context=ctx_ssl) as u, open(mp4_path, 'wb') as f:
                    f.write(u.read())

                cap = cv2.VideoCapture(mp4_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 30)
                best_f = None
                best_s = -1.0

                for idx in range(10, min(total, 400), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, fr = cap.read()
                    if not ret or fr is None: continue
                    ok, sc = eval_frame(fr, detector, facemark)
                    if ok and sc > best_s:
                        best_s = sc
                        best_f = fr.copy()
                cap.release()
                if os.path.exists(mp4_path): os.remove(mp4_path)

                if best_f is not None:
                    async with lock:
                        if saved_counter['count'] < 200 and prefix not in seen_handles:
                            saved_counter['count'] += 1
                            count = saved_counter['count']
                            out_name = f"{count:03d}_{clean_author}.jpg"
                            out_path = os.path.join(OUTPUT_DIR, out_name)
                            cv2.imwrite(out_path, best_f)
                            seen_handles.add(prefix)
                            print(f"  ✓ [{count:03d}/200] SAVED (#{cand.get('tag')}): @{clean_author} -> {out_name} (score: {best_s:.1f})")
            except Exception:
                pass
            finally:
                queue.task_done()

        await browser.close()

async def harvest_to_200():
    candidates = await scrape_diverse_candidates()
    existing_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith('composited') and not f.startswith('procrustes')])
    current_count = len(existing_files)
    print(f"=== Phase 2: Parallel Harvesting from {current_count}/200 ===")

    seen_handles = set()
    for f in existing_files:
        h = f.split('_', 1)[1].replace('.jpg', '')
        seen_handles.add(norm_handle(h))

    saved_counter = {'count': current_count}
    lock = asyncio.Lock()
    queue = asyncio.Queue()

    for c in candidates:
        queue.put_nowait(c)

    workers = [asyncio.create_task(worker(i, queue, saved_counter, seen_handles, lock)) for i in range(4)]
    await asyncio.gather(*workers)

    final_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith('composited') and not f.startswith('procrustes')])
    print(f"\nHarvesting to 200 finished! Total in {OUTPUT_DIR}: {len(final_files)}/200")

if __name__ == "__main__":
    asyncio.run(harvest_to_200())
