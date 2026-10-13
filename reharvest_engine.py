import asyncio
import os
import re
import json
import sqlite3
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

async def process_video_candidate(page, vid, author, detector, facemark, worker_id):
    embed_url = f"https://www.instagram.com/reel/{vid}/embed/"
    await page.goto(embed_url, wait_until='domcontentloaded', timeout=4000)
    await asyncio.sleep(0.4)
    
    text = await page.evaluate("() => document.body.innerText")
    likes = -1
    for line in text.split('\n'):
        m = re.search(r'([0-9,]+)\s+likes', line, re.I)
        if m:
            likes = int(m.group(1).replace(',', ''))
            break
            
    # HARD 50K LIKES GATE
    if likes < 50000:
        return None, None, likes
        
    el = await page.wait_for_selector('video', timeout=2000)
    v_src = await el.get_attribute('src')
    if not v_src:
        return None, None, likes
        
    # Download first 2.5MB
    mp4_path = f"scratch/harvest_w{worker_id}_{vid}.mp4"
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
        
    return best_f, best_s, likes
