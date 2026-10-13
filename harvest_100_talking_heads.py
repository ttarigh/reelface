import asyncio
import os
import re
import sqlite3
import cv2
import numpy as np
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

OUTPUT_DIR = "talking_head_photos_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TAGS = [
    "publicspeaking", "speechcoach", "communicationtips", "communicationskills",
    "executivepresence", "relationshiptips", "lifecoach", "dailywisdom", "dermatologist",
    "psychologist", "mentalhealthtips", "confidencecoach", "careeradvice", "leadershipskills",
    "storytime", "stoicism", "neuroscience", "tedtalks"
]

def score_talking_head(frame):
    h, w = frame.shape[:2]
    if h <= w: return -1.0, "Landscape"
    
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or len(faces) != 1: return -1.0, "Not 1 face"
    
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return -1.0, "Low conf"
    
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area_pct = (fw * fh) / (w * h) * 100
    if area_pct < 4.0 or area_pct > 18.0: return -1.0, f"Area {area_pct:.1f}%"
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    nt_x, nt_y = int(face[8]), int(face[9])
    rm_x, rm_y = int(face[10]), int(face[11])
    lm_x, lm_y = int(face[12]), int(face[13])
    
    eye_span = abs(le_x - re_x)
    if eye_span <= 12: return -1.0, "Eye span"
    
    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
    if nose_offset > 0.22: return -1.0, f"Looking away ({nose_offset:.2f})"
    
    if re_y >= nt_y or le_y >= nt_y or nt_y >= rm_y or nt_y >= lm_y:
        return -1.0, "Landmark order"
        
    lower_face = frame[max(0, nt_y):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if lower_face.size == 0: return -1.0, "Empty lower face"
    lower_gray = cv2.cvtColor(lower_face, cv2.COLOR_BGR2GRAY)
    
    sobel_h = cv2.Sobel(lower_gray, cv2.CV_64F, 0, 1, ksize=3)
    if np.max(np.abs(sobel_h)) > 1600: return -1.0, "Occlusion edge"
    
    sobel_v = cv2.Sobel(lower_gray, cv2.CV_64F, 1, 0, ksize=3)
    if np.max(np.abs(sobel_v)) > 1600: return -1.0, "Vertical hand occlusion"

    # Talking check: mouth open
    my = (rm_y + lm_y) // 2
    lip_roi = lower_gray[max(0, my - nt_y - 8):min(lower_gray.shape[0], my - nt_y + 12), :]
    lip_std = np.std(lip_roi) if lip_roi.size > 0 else 0
    if lip_std < 18: return -1.0, "Mouth closed"

    # Eyes open check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_w, eye_h = max(4, int(fw * 0.16)), max(4, int(fh * 0.12))
    def eye_stats(ex, ey):
        x1, x2 = max(0, ex - eye_w//2), min(w, ex + eye_w//2)
        y1, y2 = max(0, ey - eye_h//2), min(h, ey + eye_h//2)
        patch = gray[y1:y2, x1:x2]
        if patch.size == 0: return 0, 0
        min_v, max_v, _, _ = cv2.minMaxLoc(patch)
        sobel = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
        return (max_v - min_v), np.mean(np.abs(sobel))
        
    r_con, r_edge = eye_stats(re_x, re_y)
    l_con, l_edge = eye_stats(le_x, le_y)
    if r_con < 40 or l_con < 40 or r_edge < 13 or l_edge < 13:
        return -1.0, "Eyes not wide open"

    face_crop = gray[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    sharpness = cv2.Laplacian(face_crop, cv2.CV_64F).var() if face_crop.size > 0 else 0
    if sharpness < 40: return -1.0, "Blurry"

    score = conf * 100 + min(r_edge, l_edge)*2.0 + min(r_con, l_con)*0.4 + min(50, sharpness*0.05) - nose_offset*50 + lip_std*1.0
    return score, "OK"

print("Script template ready.")
