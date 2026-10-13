import sqlite3
import requests
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()
c.execute("SELECT id, creator_name, video_id, video_url, thumbnail_url, like_count FROM videos ORDER BY like_count DESC")
videos = c.fetchall()

print(f"Total videos in DB: {len(videos)}")

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

def verify_frame(img, detector):
    if img is None or img.size == 0:
        return False, None, "Empty image", 0
    h, w = img.shape[:2]
    detector.setInputSize((w, h))
    retval, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return False, None, "No face detected", 0

    best_face = max(faces, key=lambda f: f[2] * f[3])
    conf = float(best_face[-1])
    if conf < 0.85:
        return False, None, f"Low conf {conf:.2f}", 0

    fx, fy, fw, fh = [int(v) for v in best_face[0:4]]
    re_x, re_y = int(best_face[4]), int(best_face[5])
    le_x, le_y = int(best_face[6]), int(best_face[7])
    nt_x, nt_y = int(best_face[8]), int(best_face[9])
    rm_y, lm_y = int(best_face[11]), int(best_face[13])

    # Frontal geometry check
    eyes_y = (re_y + le_y) / 2.0
    mouth_y = (rm_y + lm_y) / 2.0
    if not (eyes_y < nt_y < mouth_y):
        return False, None, "Abnormal vertical geometry", 0

    eye_span = abs(le_x - re_x)
    if eye_span <= 0:
        return False, None, "Zero eye span", 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.40:
        return False, None, f"Not frontal (offset {nose_offset:.2f})", 0

    # Eye openness check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_w, eye_h = int(fw * 0.16), int(fh * 0.12)

    def eye_stats(ex, ey):
        x1, x2 = max(0, ex - eye_w // 2), min(w, ex + eye_w // 2)
        y1, y2 = max(0, ey - eye_h // 2), min(h, ey + eye_h // 2)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0:
            return 0, 0, 0
        min_val, max_val, _, _ = cv2.minMaxLoc(patch)
        contrast = max_val - min_val
        sobel = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.mean(np.abs(sobel))
        return min_val, contrast, edge

    r_min, r_con, r_edge = eye_stats(re_x, re_y)
    l_min, l_con, l_edge = eye_stats(le_x, le_y)

    # Sharpness
    face_crop = gray[max(0, fy):min(h, fy + fh), max(0, fx):min(w, fx + fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0

    # Obstruction check: check central triangle between eyes and nose
    # If a big play button or icon exists, center of face has stark artificial white or black geometry
    # Also check if eyes are open: contrast >= 50, edge >= 18
    eyes_open = (r_con >= 48 and l_con >= 48 and r_edge >= 16 and l_edge >= 16)
    if not eyes_open:
        return False, None, f"Eyes not sufficiently open (R: con={r_con:.0f}, edge={r_edge:.0f}; L: con={l_con:.0f}, edge={l_edge:.0f})", 0

    # Obstruction check for center play button:
    # A play button is typically right in the middle of the frame / face
    # In creator___, the play button was at the eye/nose bridge
    tri_x1, tri_x2 = max(0, int(min(re_x, le_x))), min(w, int(max(re_x, le_x)))
    tri_y1, tri_y2 = max(0, int(min(re_y, le_y))), min(h, int(nt_y))
    tri_patch = gray[tri_y1:tri_y2, tri_x1:tri_x2]
    if tri_patch.size > 0:
        # A solid white play button has many pure white pixels (>245)
        white_ratio = np.sum(tri_patch > 245) / tri_patch.size
        if white_ratio > 0.25:
            return False, None, "Play button / watermark overlay obstruction", 0

    composite_score = conf * 100 + min(r_edge, l_edge) + min(r_con, l_con) * 0.2 + min(50, sharpness * 0.05) - nose_offset * 30
    return True, best_face, "OK", composite_score

def eval_single(row):
    vid_id, creator_name, video_id, video_url, thumbnail_url, likes = row
    det = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    try:
        resp = requests.get(thumbnail_url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return vid_id, creator_name, False, "HTTP error"
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        ok, face, reason, score = verify_frame(img, det)
        return vid_id, creator_name, ok, reason
    except Exception as e:
        return vid_id, creator_name, False, str(e)

print("Evaluating 500 thumbnails with ThreadPoolExecutor...")
with ThreadPoolExecutor(max_workers=12) as executor:
    results = list(executor.map(eval_single, videos))

passed = [r for r in results if r[2]]
failed = [r for r in results if not r[2]]

print(f"Thumbnails Passed: {len(passed)} / {len(videos)} ({len(passed)/len(videos)*100:.1f}%)")
print(f"Thumbnails Needing Video Frame Rescue: {len(failed)} / {len(videos)} ({len(failed)/len(videos)*100:.1f}%)")
print("Sample reasons for rescue:")
for r in failed[:10]:
    print(f"  {r[1]}: {r[3]}")
