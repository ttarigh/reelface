import sqlite3
import requests
import cv2
import numpy as np
import os

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

conn = sqlite3.connect("viral_videos.db")
c = conn.cursor()
c.execute("SELECT creator_name, video_url, thumbnail_url FROM videos ORDER BY like_count DESC LIMIT 20")
rows = c.fetchall()

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
}

os.makedirs("scratch/test_batch", exist_ok=True)

for i, (name, v_url, t_url) in enumerate(rows, 1):
    clean_name = name.lstrip("@")
    try:
        resp = requests.get(t_url, headers=headers, timeout=5)
        if resp.status_code != 200:
            print(f"[{i}/20] {name}: Failed to download thumb ({resp.status_code})")
            continue
        nparr = np.frombuffer(resp.content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{i}/20] {name}: Failed to decode")
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            print(f"[{i}/20] {name}: No face detected in thumbnail")
            continue

        # dominant face
        best_face = max(faces, key=lambda f: f[2] * f[3])
        conf = float(best_face[-1])
        fx, fy, fw, fh = [int(v) for v in best_face[0:4]]
        re_x, re_y = int(best_face[4]), int(best_face[5])
        le_x, le_y = int(best_face[6]), int(best_face[7])

        # Eye openness check
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eye_w, eye_h = int(fw * 0.16), int(fh * 0.12)

        def eye_stats(ex, ey):
            x1, x2 = max(0, ex - eye_w//2), min(w, ex + eye_w//2)
            y1, y2 = max(0, ey - eye_h//2), min(h, ey + eye_h//2)
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

        # Sharpness (Laplacian variance on face region)
        face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
        sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0

        # Eyes open check: contrast > 50, edge > 20
        eyes_open = (r_con > 50 and l_con > 50 and r_edge > 18 and l_edge > 18)

        print(f"[{i}/20] @{clean_name:18} Conf: {conf:.2f} | Sharp: {sharpness:.0f} | "
              f"R-eye: con={r_con:.0f}, edge={r_edge:.0f} | L-eye: con={l_con:.0f}, edge={l_edge:.0f} | "
              f"Eyes Open: {eyes_open}")

        # Save cropped headshot
        pad_x = int(fw * 0.35)
        pad_y_top = int(fh * 0.50)
        pad_y_bot = int(fh * 0.35)
        x1 = max(0, fx - pad_x)
        y1 = max(0, fy - pad_y_top)
        x2 = min(w, fx + fw + pad_x)
        y2 = min(h, fy + fh + pad_y_bot)
        headshot = img[y1:y2, x1:x2]
        cv2.imwrite(f"scratch/test_batch/{clean_name}.jpg", headshot)

    except Exception as e:
        print(f"[{i}/20] {name}: Error {e}")
