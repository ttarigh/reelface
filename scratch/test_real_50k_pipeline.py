import asyncio
import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
from playwright.async_api import async_playwright

VERIFIED_MANIFEST = 'assets/verified_50k_manifest.json'
BLACKLIST_PATH = 'assets/rejected_blacklist.json'
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

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

    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0

    sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    primary = sorted_faces[0]
    conf = float(primary[-1])
    if conf < 0.82: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    if area < 0.030 or area > 0.32: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.25: return False, 0
    if fy < int(h*0.05) or fy > int(h*0.60): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 40: return False, 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.22: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.18: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.15 or m_ratio > 0.90: return False, 0

    score = conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50
    return True, score

async def test():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)

    test_vids = ['DcWJhmiJtdY', 'Das0PNVsLCv', 'DddD70jI9rX', 'DdH2V7pIBXn', 'Dd12q65J2Q9']

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 600, 'height': 900})

        for vid in test_vids:
            url = f"https://www.instagram.com/reel/{vid}/embed/"
            try:
                await page.goto(url, wait_until='domcontentloaded', timeout=6000)
                await asyncio.sleep(0.4)

                data = await page.evaluate(r'''() => {
                    let likes = null;
                    let handle = null;
                    const links = Array.from(document.querySelectorAll('a, span, div'));
                    for (const el of links) {
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
                    return { likes, handle, v_src: v ? v.src : null, img_src: img ? img.src : null };
                }''')

                likes = data.get('likes')
                handle = data.get('handle')
                print(f"[{vid}] Handle: @{handle} | Real Likes: {likes}")

                if likes and likes >= 50000:
                    print(f"  --> PASSES 50K THRESHOLD! ({likes:,} likes)")
                else:
                    print(f"  --> Skipped (< 50k or hidden likes: {likes})")

            except Exception as e:
                print(f"[{vid}] Error: {e}")

        await browser.close()

if __name__ == '__main__':
    asyncio.run(test())
