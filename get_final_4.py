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
ctx_ssl = ssl._create_unverified_context()

def load_models():
    detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("models/lbfmodel.yaml")
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
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    eye_ratio = ear(pts)
    if eye_ratio < 0.28: return False, 0
    
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.30 or m_ratio > 0.80: return False, 0
    
    # Cheek color and texture check (reject CGI/drawings)
    cx = fx + int(fw * 0.7)
    cy = fy + int(fh * 0.6)
    if cy+25 < h and cx+25 < w and cy-25 > 0 and cx-25 > 0:
        patch = gray[cy-25:cy+25, cx-25:cx+25]
        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
        if lap_var < 50: return False, 0
    
    return True, conf*100 + m_ratio*50 + eye_ratio*50

async def worker(worker_id, queue, saved_counter, seen, lock):
    detector, facemark = load_models()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        while not queue.empty():
            cand = await queue.get()
            vid = cand.get('video_id')
            author = cand.get('author', '').split('\n')[0].strip()
            clean = re.sub(r'[^a-zA-Z0-9._-]', '', author)[:25]
            
            async with lock:
                if saved_counter['count'] >= 200:
                    queue.task_done()
                    break
                if not clean or clean[:6] in seen:
                    queue.task_done()
                    continue

            try:
                await page.goto(f'https://www.instagram.com/reel/{vid}/embed/', wait_until='domcontentloaded', timeout=4000)
                el = await page.wait_for_selector('video', timeout=2500)
                v_src = await el.get_attribute('src')
                if not v_src:
                    queue.task_done()
                    continue

                header_author = await page.evaluate(r'''() => {
                    const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*=\"/p/\"], .UsernameText');
                    return el ? el.innerText.trim() : '';
                }''')
                if header_author:
                    clean = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25]
                    async with lock:
                        if clean[:6] in seen:
                            queue.task_done()
                            continue

                mp4 = f'scratch/final4_{worker_id}_{vid}.mp4'
                req = urllib.request.Request(v_src, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=3.5, context=ctx_ssl) as u, open(mp4, 'wb') as f:
                    f.write(u.read())

                cap = cv2.VideoCapture(mp4)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, total // 30)
                best_f = None
                best_s = -1
                for idx in range(10, min(total, 400), step):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, fr = cap.read()
                    if not ret or fr is None: continue
                    ok, sc = eval_frame(fr, detector, facemark)
                    if ok and sc > best_s:
                        best_s = sc
                        best_f = fr.copy()
                cap.release()
                if os.path.exists(mp4): os.remove(mp4)

                if best_f is not None:
                    async with lock:
                        if saved_counter['count'] < 200 and clean[:6] not in seen:
                            saved_counter['count'] += 1
                            count = saved_counter['count']
                            out_name = f"{count:03d}_{clean}.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), best_f)
                            seen.add(clean[:6])
                            print(f"  ✓ [{count:03d}/200] SAVED: @{clean} -> {out_name} (score: {best_s:.1f})")
            except Exception:
                pass
            finally:
                queue.task_done()
                
        await browser.close()

async def main():
    existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone'))])
    current_count = len(existing)
    seen = {f.split('_', 1)[1].replace('.jpg', '')[:6] for f in existing}
    
    with open('fast_3.json') as f:
        candidates = json.load(f)
    queue = asyncio.Queue()
    for c in candidates:
        queue.put_nowait(c)
        
    saved_counter = {'count': current_count}
    lock = asyncio.Lock()
    
    print(f"Harvesting final {200 - current_count} faces to reach exactly 200...")
    workers = [asyncio.create_task(worker(i, queue, saved_counter, seen, lock)) for i in range(4)]
    await asyncio.gather(*workers)
    
    final_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone'))])
    print(f"Done! Final count in {OUTPUT_DIR}: {len(final_files)}/200")

if __name__ == '__main__':
    asyncio.run(main())
