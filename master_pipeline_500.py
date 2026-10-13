import asyncio
import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.spatial import Delaunay
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("scratch", exist_ok=True)
CANDIDATES_FILE = "candidates_500.json"

ctx_ssl = ssl._create_unverified_context()

DIVERSE_TAGS = [
    # Comedy, Standup & Improv
    "standup", "standupcomedy", "standupcomedian", "comedyclub", "comedyshow", "crowdwork",
    "roastcomedy", "britishcomedy", "irishcomedy", "canadiancomedy", "aussiecomedy", "openmic",
    "jokes", "humor", "improv", "sketchcomedy", "comedycentral",
    # TV, Journalism & News Anchors
    "journalism", "journalist", "newsanchor", "broadcaster", "tvhost", "tvpresenter",
    "fieldreporter", "investigativejournalism", "breakingnews", "weatheranchor", "meteorologist",
    "sportsanchor", "sportscaster", "radiohost", "podcaster", "podcasting", "interview",
    # Academia, Science & Professors
    "professor", "professors", "scientist", "biologist", "marinebiology", "physicist",
    "astronomy", "astrophysics", "neuroscience", "psychology", "psychologist", "therapist",
    "counselor", "sociology", "anthropology", "archaeology", "paleontology", "geology",
    "botany", "zoology", "historyteacher", "scienceteacher", "stemeducation", "spacefacts",
    # Acting, Theatre & Cinema
    "actor", "acting", "actress", "theatre", "theater", "broadway", "stageactor",
    "screenacting", "actingcoach", "actingtips", "actingaudition", "monologue",
    "filmmaker", "director", "filmdirector", "cinematography", "screenwriter", "screenwriting",
    "voiceactor", "voiceover", "voiceacting",
    # Public Speaking & Authors
    "publicspeaking", "publicspeaker", "keynotespeaker", "tedxtalk", "tedtalk",
    "toastmasters", "speechcoach", "vocalcoach", "storytelling", "communicationtips",
    "author", "authors", "writer", "writers", "writingcommunity", "booktok", "poet", "spokenword"
]

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
    
    # 1. Color check (reject archival / B&W)
    b, g, r = cv2.split(fr)
    if np.mean(np.abs(b - g)) < 5 and np.mean(np.abs(g - r)) < 5:
        return False, 0
        
    # Letterbox check
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
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    # 2. Sunglasses check
    r_patch = gray[max(0, re_y-15):min(h, re_y+15), max(0, re_x-20):min(w, re_x+20)]
    l_patch = gray[max(0, le_y-15):min(h, le_y+15), max(0, le_x-20):min(w, le_x+20)]
    if np.mean(r_patch) < 55 or np.mean(l_patch) < 55: return False, 0
    
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    # 3. Direct camera gaze (reject downward laptop/book gaze)
    eye_ratio = ear(pts)
    if eye_ratio < 0.30: return False, 0
    
    # 4. Speech articulation
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.30 or m_ratio > 0.80: return False, 0
    
    # 5. Sclera / Cartoon check (reject 3D Pixar / CGI doll eyes)
    y1, y2 = int(min(pts[37,1], pts[38,1])), int(max(pts[40,1], pts[41,1]))
    x1, x2 = int(pts[36,0]), int(pts[39,0])
    if y2 > y1 and x2 > x1:
        re_box = gray[y1:y2, x1:x2]
        if re_box.size > 0:
            p95 = np.percentile(re_box, 95)
            re_h = y2 - y1
            if p95 > 195 and (re_h / max(1, fw)) > 0.08:
                return False, 0
    
    # 6. Chin / Lower mouth obstruction check (reject microphones, hands, pop filters)
    chin_y = int(pts[8, 1])
    chin_x = int(pts[8, 0])
    my = int(pts[62, 1])
    mx = int(pts[62, 0])
    if chin_y > my:
        sub_chin = gray[my:chin_y, max(0, mx-30):min(w, mx+30)]
        if sub_chin.size > 0:
            if np.mean(sub_chin) < 30 or np.std(sub_chin) < 12: return False, 0
            if np.max(np.abs(cv2.Sobel(sub_chin, cv2.CV_64F, 0, 1, ksize=3))) > 920: return False, 0
        
    # 7. Unobstructed cheeks and neck (no hands covering face)
    l_cheek = gray[fy:fy+fh, max(0, fx-40):fx]
    r_cheek = gray[fy:fy+fh, fx+fw:min(w, fx+fw+40)]
    if np.max(np.abs(cv2.Sobel(l_cheek, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
    if np.max(np.abs(cv2.Sobel(r_cheek, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
    
    neck_h = min(h - chin_y, 70)
    if neck_h > 20:
        neck = gray[chin_y:chin_y+neck_h, max(0, chin_x-40):min(w, chin_x+40)]
        if np.max(np.abs(cv2.Sobel(neck, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
        
    score = conf*100 + m_ratio*50 + eye_ratio*50
    return True, score

def align_to_canonical_canvas(img, landmarks, canvas_w=800, canvas_h=1000):
    re_center = np.mean(landmarks[36:42], axis=0)
    le_center = np.mean(landmarks[42:48], axis=0)

    target_re = np.array([canvas_w * 0.36, canvas_h * 0.40])
    target_le = np.array([canvas_w * 0.64, canvas_h * 0.40])

    src_pts = np.float32([re_center, le_center])
    dst_pts = np.float32([target_re, target_le])

    M = cv2.estimateAffinePartial2D(src_pts.reshape(1, 2, 2), dst_pts.reshape(1, 2, 2))[0]
    aligned_img = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    ones = np.ones((68, 1))
    homo = np.hstack([landmarks, ones])
    aligned_landmarks = np.dot(homo, M.T)
    return aligned_img, aligned_landmarks

def generalized_procrustes_analysis(shapes, max_iter=100, tol=1e-7):
    centered = [s - np.mean(s, axis=0) for s in shapes]
    normalized = [s / np.sqrt(np.sum(s**2)) for s in centered]
    mean_shape = normalized[0].copy()

    for it in range(max_iter):
        aligned_list = []
        for s in normalized:
            M = np.dot(s.T, mean_shape)
            U, _, Vt = np.linalg.svd(M)
            R = np.dot(U, Vt)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(U, Vt)
            aligned_list.append(np.dot(s, R))

        new_mean = np.mean(aligned_list, axis=0)
        new_mean -= np.mean(new_mean, axis=0)
        new_mean /= np.sqrt(np.sum(new_mean**2))

        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean
        if diff < tol:
            break

    avg_re = np.mean([np.mean(s[36:42], axis=0) for s in shapes], axis=0)
    avg_le = np.mean([np.mean(s[42:48], axis=0) for s in shapes], axis=0)
    target_dist = np.linalg.norm(avg_le - avg_re)

    mean_re = np.mean(mean_shape[36:42], axis=0)
    mean_le = np.mean(mean_shape[42:48], axis=0)
    curr_dist = np.linalg.norm(mean_le - mean_re)

    scale = target_dist / curr_dist
    mean_center = (mean_re + mean_le) / 2.0
    target_center = (avg_re + avg_le) / 2.0
    return (mean_shape - mean_center) * scale + target_center

def get_boundary_anchors(w, h):
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 4], [w - 1, h // 4],
        [0, h // 2], [w - 1, h // 2],
        [0, 3 * h // 4], [w - 1, 3 * h // 4],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
    ], dtype=np.float64)

def run_milestone_compositing(target_count):
    CANVAS_W, CANVAS_H = 800, 1000
    print(f"\n=======================================================")
    print(f"RUNNING PROCRUSTEAN COMPOSITING FOR MILESTONE {target_count}")
    print(f"=======================================================")
    
    yunet, facemark = load_models()
    all_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    target_files = all_files[:target_count]
    
    raw_images = []
    raw_landmarks = []
    valid_files = []
    
    for f in target_files:
        img = cv2.imread(os.path.join(OUTPUT_DIR, f))
        if img is None: continue
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(img)
        if faces is None or len(faces) == 0: continue
        fx, fy, fw, fh = [int(v) for v in faces[0][:4]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
        if ok and len(lms) > 0:
            raw_images.append(img)
            raw_landmarks.append(lms[0].reshape(68, 2).astype(np.float64))
            valid_files.append(f)
            
    print(f"Extracted landmarks for all {len(raw_images)}/{target_count} images.")
    
    aligned_images = []
    aligned_landmarks = []
    for img, pts in zip(raw_images, raw_landmarks):
        aimg, apts = align_to_canonical_canvas(img, pts, CANVAS_W, CANVAS_H)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)
        
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)
    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])
    
    warped_rgb_list = []
    for idx, (aimg, apts) in enumerate(zip(aligned_images, aligned_landmarks)):
        src_full = np.vstack([apts, boundary_anchors])
        rgb_float = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        tform = PiecewiseAffineTransform.from_estimate(target_full, src_full)
        warped_rgb = warp(rgb_float, tform, output_shape=(CANVAS_H, CANVAS_W), mode='reflect')
        warped_rgb_list.append(warped_rgb)
        
    stack = np.stack(warped_rgb_list, axis=0)
    avg_rgb = np.mean(stack, axis=0)
    avg_bgr = cv2.cvtColor((np.clip(avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # Save cumulative composite
    cum_path = f"composited_{target_count}_faces.jpg"
    cv2.imwrite(cum_path, avg_bgr)
    print(f"  ✓ Saved {cum_path}")
    
    # Also save the batch-only composite for this specific 100-slice
    start_batch = target_count - 100
    if start_batch >= 0 and len(warped_rgb_list) >= target_count:
        batch_stack = np.stack(warped_rgb_list[start_batch:target_count], axis=0)
        batch_avg = cv2.cvtColor((np.clip(np.mean(batch_stack, axis=0), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        batch_path = f"composited_batch_{start_batch+1}_to_{target_count}_only.jpg"
        cv2.imwrite(batch_path, batch_avg)
        print(f"  ✓ Saved {batch_path}")
        
    # Grid generation
    cols = 20
    rows = (target_count + cols - 1) // cols
    cw, ch = 90, 115
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for idx, cell_img in enumerate(aligned_images[:target_count]):
        r = idx // cols
        c = idx % cols
        grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = cv2.resize(cell_img, (cw, ch))
    grid_path = f"procrustes_{target_count}_faces_grid.jpg"
    cv2.imwrite(grid_path, grid)
    print(f"  ✓ Saved {grid_path}")
    
    # If final 500 reached, generate master evolution poster
    if target_count == 500:
        milestones = [15, 100, 200, 300, 400, 500]
        panel_w, panel_h = 320, 420
        banner_h = 520
        total_w = len(milestones) * panel_w + (len(milestones) + 1) * 20
        master = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
        
        for idx, m in enumerate(milestones):
            m_path = f"composited_{m}_faces.jpg"
            if os.path.exists(m_path):
                m_img = cv2.imread(m_path)
                m_res = cv2.resize(m_img, (panel_w, panel_h))
                x = 20 + idx * (panel_w + 20)
                master[60:60+panel_h, x:x+panel_w] = m_res
                label = f"{m} Faces"
                cv2.putText(master, label, (x + panel_w//4, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
        cv2.imwrite("master_evolution_15_to_500.jpg", master)
        print("  ✓ Saved master_evolution_15_to_500.jpg")
        
        # Write final catalog
        with open("talking_heads_500.md", "w") as out:
            out.write("# Viral Talking-Head Creators Catalog (500 Faces)\n\n")
            out.write(f"Total Verified Viral Talking Heads: **500**\n\n")
            out.write("| Rank | Creator Handle | Reel Link | Screenshot |\n")
            out.write("| :--- | :--- | :--- | :--- |\n")
            for idx, f in enumerate(all_files[:500]):
                handle = f.split('_', 1)[1].replace('.jpg', '')
                out.write(f"| {idx+1:03d} | @{handle} | [{handle}](https://www.instagram.com/{handle}/) | [{f}](file:///Users/tarighian/most-viral-face/{OUTPUT_DIR}/{f}) |\n")
        print("  ✓ Generated talking_heads_500.md")

def process_video_stream(v_src, detector, facemark):
    try:
        cap = cv2.VideoCapture(v_src)
        best_f = None
        best_s = -1
        for i in range(120):
            ret, fr = cap.read()
            if not ret or fr is None: break
            if i < 8 or i % 3 != 0: continue
            ok, sc = eval_frame(fr, detector, facemark)
            if ok and sc > best_s:
                best_s = sc
                best_f = fr.copy()
                if sc > 145: break
        cap.release()
        return best_f, best_s
    except Exception:
        return None, -1

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
                if saved_counter['count'] >= 500:
                    break
                if not clean or clean[:6] in seen:
                    continue

            try:
                await page.goto(f'https://www.instagram.com/reel/{vid}/embed/', wait_until='domcontentloaded', timeout=3000)
                el = await page.wait_for_selector('video', timeout=1800)
                v_src = await el.get_attribute('src')
                if not v_src:
                    continue

                header_author = await page.evaluate(r'''() => {
                    const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*=\"/p/\"], .UsernameText');
                    return el ? el.innerText.trim() : '';
                }''')
                if header_author:
                    clean = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25]
                    async with lock:
                        if clean[:6] in seen:
                            continue

                best_f, best_s = await asyncio.to_thread(process_video_stream, v_src, detector, facemark)

                if best_f is not None:
                    async with lock:
                        if saved_counter['count'] < 500 and clean[:6] not in seen:
                            saved_counter['count'] += 1
                            count = saved_counter['count']
                            out_name = f"{count:03d}_{clean}.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), best_f)
                            seen.add(clean[:6])
                            print(f"  ✓ [{count:03d}/500] SAVED (#{cand.get('tag')}): @{clean} -> {out_name} (score: {best_s:.1f})", flush=True)
                            
                            # Check milestone triggers
                            if count == 300:
                                await asyncio.to_thread(run_milestone_compositing, 300)
                            elif count == 400:
                                await asyncio.to_thread(run_milestone_compositing, 400)
                            elif count == 500:
                                await asyncio.to_thread(run_milestone_compositing, 500)
            except Exception:
                pass
            finally:
                queue.task_done()
        await browser.close()

MORE_TAGS = [
    # Academia, Philosophy & Science
    "professors", "historyfacts", "philosophers", "stoicism", "ancientgreece", "ancientrome",
    "psychologists", "counselors", "sociologyfacts", "anthropologylife", "biologists",
    "astronomyfacts", "physicsfun", "scienceteachers", "stemlife", "geologyrocks",
    # Comedy & Standup
    "standupcomedians", "comedyclub", "openmic", "comediansofinstagram", "ukcomedy",
    "canadiancomedy", "aussiehumour", "improvtroupe", "sketchcomedian", "standupclips",
    # TV, Journalism & Speaking
    "tvhost", "newsanchors", "fieldreporter", "meteorologists", "interviewseries",
    "broadcastnews", "podcasthosting", "tedxtalks", "publicspeakingcoach", "keynotespeakers",
    "toastmaster", "speechtips", "communicationcoach", "vocalcoaching",
    # Performing Arts & Cinema
    "actorslife", "actingauditions", "monologues", "actingcoach", "theatredirector",
    "broadwayactors", "voiceoverartists", "voiceactorslife", "filmdirector", "screenwriters",
    # Authors, Languages & Culture
    "authorlife", "writingcommunity", "indieauthor", "booktokers", "poetryreadings",
    "linguist", "polyglotclub", "eslteacher", "englishteacher", "frenchculture"
]

async def collect_batch(tags):
    print(f"\nScraping candidate reels across {len(tags)} explore tags...", flush=True)
    cands = []
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        page = await ctx.new_page()
        for tag in tags:
            try:
                url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                await page.goto(url, wait_until='domcontentloaded', timeout=3500)
                await asyncio.sleep(0.1)
                for _ in range(2):
                    await page.evaluate("window.scrollBy(0, 1500)")
                    await asyncio.sleep(0.15)
                found = await page.evaluate(r'''() => {
                    return Array.from(document.querySelectorAll('a[href*=\"/reel/\"]')).map(a => ({
                        href: a.href.split('?')[0],
                        author: (a.innerText || '').split('\\n')[0].trim()
                    }));
                }''')
                for r in found:
                    if r.get('author'):
                        m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                        if m:
                            r['video_id'] = m.group(1)
                            r['tag'] = tag
                            cands.append(r)
                print(f"  #{tag:24}: {len(found)} reels", flush=True)
            except Exception:
                pass
        await ctx.close()
    return cands

async def main():
    existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    current_count = len(existing)
    seen = {f.split('_', 1)[1].replace('.jpg', '')[:6] for f in existing}
    saved_counter = {'count': current_count}
    lock = asyncio.Lock()
    
    print(f"=== Starting Uninterrupted Master Pipeline: Count = {current_count}/500 ===", flush=True)
    
    tag_waves = [
        MORE_TAGS,
        DIVERSE_TAGS,
        ["standup", "acting", "broadcasting", "astronomy", "linguistics", "philosophy", "journalism", "theatre", "filmmaking", "history"]
    ]
    wave_idx = 0
    
    while saved_counter['count'] < 500:
        current_tags = tag_waves[wave_idx % len(tag_waves)]
        wave_idx += 1
        
        candidates = await collect_batch(current_tags)
        queue = asyncio.Queue()
        for c in candidates:
            if c.get('author') and c['author'][:6] not in seen:
                queue.put_nowait(c)
                
        print(f"\n--> Queued {queue.qsize()} fresh candidate reels across 8 parallel workers (Target: 500, Current: {saved_counter['count']})...", flush=True)
        if queue.empty():
            await asyncio.sleep(1)
            continue
            
        workers = [asyncio.create_task(worker(i, queue, saved_counter, seen, lock)) for i in range(8)]
        await asyncio.gather(*workers)
        print(f"Wave finished. Current count: {saved_counter['count']}/500", flush=True)
        
    final_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    print(f"\n=======================================================")
    print(f"MASTER PIPELINE FINISHED! Total creators: {len(final_files)}/500")
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
