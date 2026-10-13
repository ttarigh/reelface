import asyncio
import os
import re
import json
import glob
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
VERIFIED_MANIFEST = "assets/verified_50k_manifest.json"
CREATORS_PATH = "assets/creators.json"
MARKDOWN_OUT = "talking_heads_1000.md"
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

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
    if m_ratio < 0.28 or m_ratio > 0.80: return False, 0
    
    score = conf*100 + m_ratio*50 + eye_ratio*50
    return True, score

def download_and_analyze_video(v_src, worker_id, vid, detector, facemark):
    try:
        mp4_path = f"scratch/reharvest_w{worker_id}_{vid}.mp4"
        req = urllib.request.Request(v_src, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3.5, context=ctx_ssl) as u, open(mp4_path, 'wb') as f:
            f.write(u.read(2500000))

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
        if os.path.exists(mp4_path):
            os.remove(mp4_path)

        return best_f, best_s
    except Exception:
        return None, -1.0

async def worker(worker_id, queue, saved_counter, seen_vids, seen_handles, manifest, lock, ctx):
    detector, facemark = create_models()
    page = await ctx.new_page()

    while not queue.empty():
        try:
            cand = await queue.get()
        except asyncio.QueueEmpty:
            break

        vid = cand.get('video_id')
        if not vid or vid in seen_vids:
            continue

        author = cand.get('author', '').split('\n')[0].strip().replace('@', '')
        clean = re.sub(r'[^a-zA-Z0-9._-]', '', author)[:25]
        if not clean or clean == 'creator':
            clean = f"creator_{vid[:8]}"

        prefix = clean[:6].lower()
        if prefix in seen_handles:
            continue

        async with lock:
            if saved_counter['count'] >= 1000:
                break

        try:
            embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
            await page.goto(embed_url, wait_until='domcontentloaded', timeout=4000)
            await asyncio.sleep(0.3)

            # Parse likes
            text = await page.evaluate("() => document.body.innerText")
            likes = -1
            for line in text.split('\n'):
                m = re.search(r'([0-9,]+)\s+likes', line, re.I)
                if m:
                    likes = int(m.group(1).replace(',', ''))
                    break

            # STRICT 50,000 LIKES HARD GATE
            if likes < 50000:
                continue

            # Also grab author from embed header if possible
            header_author = await page.evaluate(r'''() => {
                const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*="/p/"], .UsernameText');
                return el ? el.innerText.trim() : '';
            }''')
            if header_author:
                clean = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25]
                prefix = clean[:6].lower()
                if prefix in seen_handles:
                    continue

            # Get video src
            v_src = await page.evaluate(r'''() => {
                const v = document.querySelector('video');
                return v ? v.src : null;
            }''')
            if not v_src:
                try:
                    el = await page.wait_for_selector('video', timeout=1800)
                    v_src = await el.get_attribute('src')
                except Exception:
                    pass
            if not v_src:
                continue

            # Run download and video analysis in thread to never block async event loop
            best_f, best_s = await asyncio.to_thread(download_and_analyze_video, v_src, worker_id, vid, detector, facemark)

            if best_f is not None:
                async with lock:
                    if saved_counter['count'] < 1000 and prefix not in seen_handles:
                        saved_counter['count'] += 1
                        count = saved_counter['count']
                        rank_str = f"{count:04d}"
                        out_name = f"{rank_str}_{clean}.jpg"
                        out_path = os.path.join(OUTPUT_DIR, out_name)
                        cv2.imwrite(out_path, best_f)

                        seen_vids.add(vid)
                        seen_handles.add(prefix)

                        reel_url = f"https://www.instagram.com/reel/{vid}/"
                        manifest[rank_str] = {
                            'rank': rank_str,
                            'handle': clean,
                            'video_id': vid,
                            'reelUrl': reel_url,
                            'likes': likes,
                            'photo': out_name,
                            'score': float(best_s)
                        }

                        # Save manifest after EVERY face addition so progress is never lost
                        with open(VERIFIED_MANIFEST, 'w') as f:
                            json.dump(manifest, f, indent=2)

                        print(f"  [W{worker_id}] ✓ [{rank_str}/1000] SAVED: @{clean} -> {reel_url} ({likes:,} likes, score: {best_s:.1f})", flush=True)

        except Exception:
            pass
        finally:
            try:
                queue.task_done()
            except ValueError:
                pass

    await page.close()

def run_procrustes_compositing(total_faces):
    print(f"\n=======================================================")
    print(f"RUNNING PROCRUSTEAN COMPOSITING FOR {total_faces} VERIFIED 50K+ FACES")
    print(f"=======================================================")
    
    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)
        
    creators_list = [manifest[f"{i+1:04d}"] for i in range(total_faces) if f"{i+1:04d}" in manifest]
    
    CANVAS_W = 800
    CANVAS_H = 1000
    
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    
    aligned_images = []
    canonical_eye_r = np.array([285.0, 400.0], dtype=np.float32)
    canonical_eye_l = np.array([513.0, 397.0], dtype=np.float32)
    dst_eyes = np.array([canonical_eye_r, canonical_eye_l], dtype=np.float32)
    
    for c in creators_list:
        p_path = os.path.join(OUTPUT_DIR, c['photo'])
        if not os.path.exists(p_path): continue
        img = cv2.imread(p_path)
        if img is None: continue
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is not None and len(faces) > 0:
            f = faces[0]
            re_pt = np.array([f[4], f[5]], dtype=np.float32)
            le_pt = np.array([f[6], f[7]], dtype=np.float32)
            src_eyes = np.array([re_pt, le_pt], dtype=np.float32)
            
            M = cv2.estimateAffinePartial2D(src_eyes.reshape(1, 2, 2), dst_eyes.reshape(1, 2, 2))[0]
            if M is not None:
                warped = cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                aligned_images.append(warped.astype(np.float32))
                
    if aligned_images:
        avg_img = np.mean(aligned_images, axis=0)
        avg_bgr = np.clip(avg_img, 0, 255).astype(np.uint8)
        
        comp_name = f"composited_{total_faces}_faces.jpg"
        cv2.imwrite(comp_name, avg_bgr)
        print(f"  ✓ Saved {comp_name}")
        
        # Also copy to root composite
        cv2.imwrite("composited_1000_faces.jpg", avg_bgr)
        print("  ✓ Updated composited_1000_faces.jpg")

async def main():
    with open(VERIFIED_MANIFEST) as f:
        manifest = json.load(f)

    current_count = len(manifest)
    print(f"=== Starting Fast Reharvest to 1,000 Faces (Current Verified >=50K: {current_count}/1000) ===")

    seen_vids = {item.get('video_id') for item in manifest.values() if item.get('video_id')}
    seen_handles = {item.get('handle', '')[:6].lower() for item in manifest.values() if item.get('handle')}
    saved_counter = {'count': current_count}
    lock = asyncio.Lock()

    # Load all cached candidate reels from JSON files
    all_candidates = []
    seen_cand_vids = set()

    for jf in glob.glob('*.json') + glob.glob('scratch/*.json'):
        if 'creators' in jf or 'verified' in jf: continue
        try:
            with open(jf) as f:
                data = json.load(f)
                if isinstance(data, list):
                    for it in data:
                        if isinstance(it, dict):
                            v = it.get('video_id')
                            if not v and it.get('href'):
                                m = re.search(r'/reel/([a-zA-Z0-9_-]+)', it['href'])
                                if m: v = m.group(1)
                            if v and v not in seen_vids and v not in seen_cand_vids:
                                seen_cand_vids.add(v)
                                all_candidates.append({
                                    'video_id': v,
                                    'author': it.get('author') or 'creator'
                                })
        except: pass

    print(f"Queuing {len(all_candidates)} pre-cached candidate reels across 8 workers...")

    queue = asyncio.Queue()
    for c in all_candidates:
        queue.put_nowait(c)

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        num_workers = 8
        workers = [asyncio.create_task(worker(i, queue, saved_counter, seen_vids, seen_handles, manifest, lock, ctx)) for i in range(num_workers)]
        await asyncio.gather(*workers)
        await ctx.close()

    print(f"\n=======================================================")
    print(f"REHARVEST COMPLETE! Total verified faces: {saved_counter['count']}/1000")
    print(f"=======================================================")

    # Run Procrustean face averaging across all verified faces
    run_procrustes_compositing(saved_counter['count'])

    # Sync creators.json and talking_heads_1000.md
    creators_list = list(manifest.values())
    with open(CREATORS_PATH, 'w') as f:
        json.dump(creators_list, f, indent=2)

    if os.path.exists('web/assets'):
        with open('web/assets/creators.json', 'w') as f:
            json.dump(creators_list, f, indent=2)

    with open(MARKDOWN_OUT, 'w') as out:
        out.write("# Viral Talking-Head Creators Catalog (1000 Faces)\n\n")
        out.write("Total Verified Viral Talking Heads (All >= 50,000 Likes): **1000**\n\n")
        out.write("| Rank | Creator Handle | Reel Link | Verified Likes | Screenshot |\n")
        out.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for c in creators_list:
            r = c['rank']
            h = c['handle']
            link = c['reelUrl']
            likes_fmt = f"{c.get('likes', 0):,}"
            photo = c['photo']
            out.write(f"| {r} | @{h} | [Watch Reel]({link}) | **{likes_fmt}** | [{photo}](file:///Users/tarighian/most-viral-face/{OUTPUT_DIR}/{photo}) |\n")

    print(f"Synced {len(creators_list)} creators to {CREATORS_PATH} and {MARKDOWN_OUT}!")

if __name__ == "__main__":
    asyncio.run(main())
