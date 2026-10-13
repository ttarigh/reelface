#!/usr/bin/env python3
import os
import re
import json
import time
import cv2
import numpy as np
import concurrent.futures
from skimage.transform import PiecewiseAffineTransform, warp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'assets', 'verified_50k_manifest.json')
CREATORS_PATH = os.path.join(BASE_DIR, 'assets', 'creators.json')
WEB_CREATORS = os.path.join(BASE_DIR, 'web', 'assets', 'creators.json')
OUTPUT_DIR = os.path.join(BASE_DIR, 'talking_head_photos_100')
YUNET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
LBF_MODEL = os.path.join(BASE_DIR, "models", "lbfmodel.yaml")

CANVAS_W = 800
CANVAS_H = 1000
TARGET_COUNT = 1000

def load_models():
    detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.5)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def align_to_canonical_canvas(img, landmarks):
    re_center = np.mean(landmarks[36:42], axis=0)
    le_center = np.mean(landmarks[42:48], axis=0)

    target_re = np.array([CANVAS_W * 0.36, CANVAS_H * 0.40])
    target_le = np.array([CANVAS_W * 0.64, CANVAS_H * 0.40])

    src_pts = np.float32([re_center, le_center])
    dst_pts = np.float32([target_re, target_le])

    d_x = target_le[0] - target_re[0]
    d_y = target_le[1] - target_re[1]
    target_dist = np.sqrt(d_x**2 + d_y**2)

    s_x = le_center[0] - re_center[0]
    s_y = le_center[1] - re_center[1]
    src_dist = np.sqrt(s_x**2 + s_y**2)

    scale = target_dist / (src_dist + 1e-5)
    angle = np.degrees(np.arctan2(s_y, s_x))

    M = cv2.getRotationMatrix2D(tuple(re_center), angle, scale)
    M[0, 2] += target_re[0] - re_center[0]
    M[1, 2] += target_re[1] - re_center[1]

    aligned_img = cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    ones = np.ones((landmarks.shape[0], 1))
    pts_homo = np.hstack([landmarks, ones])
    aligned_landmarks = M.dot(pts_homo.T).T
    return aligned_img, aligned_landmarks

def generalized_procrustes_analysis(shapes_list, max_iter=20, tol=1e-5):
    shapes = np.array(shapes_list, dtype=np.float64)
    n_shapes, n_points, _ = shapes.shape

    centered = shapes - np.mean(shapes, axis=1, keepdims=True)
    scales = np.sqrt(np.sum(centered**2, axis=(1, 2), keepdims=True))
    normalized = centered / (scales + 1e-7)

    mean_shape = normalized[0].copy()
    for it in range(max_iter):
        aligned_shapes = []
        for s in normalized:
            H = np.dot(s.T, mean_shape)
            U, _, Vt = np.linalg.svd(H)
            R = np.dot(U, Vt)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(U, Vt)
            aligned_shapes.append(np.dot(s, R))
        new_mean = np.mean(aligned_shapes, axis=0)
        new_mean = new_mean / np.sqrt(np.sum(new_mean**2))
        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean
        if diff < tol:
            print(f"Procrustes shape optimization converged in {it+1} iterations (diff: {diff:.2e}).", flush=True)
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

def warp_worker(args):
    aimg, apts, target_full, boundary_anchors = args
    src_full = np.vstack([apts, boundary_anchors])
    rgb_float = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tform = PiecewiseAffineTransform.from_estimate(target_full, src_full)
    warped = warp(rgb_float, tform, output_shape=(CANVAS_H, CANVAS_W), mode='reflect')
    return warped.astype(np.float32)

def main():
    print(f"=== REMAKING THE 1,000 MOST VIRAL FACE COMPOSITE & WEBSITE ===", flush=True)
    t0_total = time.time()

    # 1. Load approved creators from manifest
    with open(MANIFEST_PATH) as f:
        mf = json.load(f)

    approved_entries = [v for v in mf.values() if v.get('approved')]
    print(f"Total approved in manifest: {len(approved_entries)}", flush=True)

    # Sort deterministically by rank (integer if possible)
    def rank_key(x):
        r = str(x.get('rank', '99999'))
        return int(r) if r.isdigit() else 99999

    approved_sorted = sorted(approved_entries, key=rank_key)
    creators_1000 = approved_sorted[:TARGET_COUNT]
    print(f"Selected exactly {len(creators_1000)} creators for the official 1,000 deck.", flush=True)

    # Re-index ranks 0001 through 1000 for perfect website alignment
    final_creators = []
    for idx, c in enumerate(creators_1000):
        c_copy = dict(c)
        c_copy['rank'] = f"{idx+1:04d}"
        final_creators.append(c_copy)

    # Save official creators.json
    with open(CREATORS_PATH, 'w') as f:
        json.dump(final_creators, f, indent=2)
    print(f"  ✓ Updated {CREATORS_PATH}")

    if os.path.exists(os.path.dirname(WEB_CREATORS)):
        with open(WEB_CREATORS, 'w') as f:
            json.dump(final_creators, f, indent=2)
        print(f"  ✓ Updated {WEB_CREATORS}")

    # 2. Extract Landmarks & Rigidly Align
    print(f"\n--- 1. Detecting Landmarks on all {len(final_creators)} Faces ---", flush=True)
    detector, facemark = load_models()

    aligned_images = []
    aligned_landmarks = []
    rigid_accum = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)

    t0_detect = time.time()
    for idx, c in enumerate(final_creators):
        p = os.path.join(OUTPUT_DIR, c.get('photo'))
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Missing photo: {p}")

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        if faces is None or len(faces) == 0:
            raise ValueError(f"No face detected in {p}")

        fx, fy, fw, fh = [int(v) for v in faces[0][:4]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
        if not ok or len(lms) == 0:
            raise ValueError(f"No landmarks found for {p}")

        pts = lms[0].reshape(68, 2).astype(np.float64)
        aimg, apts = align_to_canonical_canvas(img, pts)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)
        rigid_accum += cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if (idx + 1) % 200 == 0 or idx == len(final_creators) - 1:
            print(f"  ✓ Processed [{idx+1:04d}/{len(final_creators)}] faces ({time.time()-t0_detect:.1f}s)", flush=True)

    # 3. Generalized Procrustes Analysis
    print(f"\n--- 2. Computing Generalized Procrustes Analysis (GPA) Mean Shape ---", flush=True)
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)
    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])
    print(f"  ✓ Target Procrustean mesh constructed with {len(target_full)} vertices.", flush=True)

    # 4. Piecewise Affine Warping & Averaging
    print(f"\n--- 3. Piecewise Affine Warping & Averaging (8 Threads) ---", flush=True)
    t0_warp = time.time()
    warp_tasks = [(aimg, apts, target_full, boundary_anchors) for aimg, apts in zip(aligned_images, aligned_landmarks)]

    cum_acc = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, warped_rgb in enumerate(executor.map(warp_worker, warp_tasks)):
            cum_acc += warped_rgb
            if (idx + 1) % 200 == 0 or idx == len(warp_tasks) - 1:
                print(f"  ✓ Warped [{idx+1:04d}/{len(warp_tasks)}] faces ({time.time()-t0_warp:.1f}s)...", flush=True)

    avg_rgb = cum_acc / len(aligned_images)
    avg_bgr = cv2.cvtColor((np.clip(avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save sharp Procrustean composite
    comp_path = os.path.join(BASE_DIR, 'composited_1000_faces.jpg')
    cv2.imwrite(comp_path, avg_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\n  ✓ Saved {comp_path}", flush=True)

    comp_assets = os.path.join(BASE_DIR, 'assets', 'composited_1000_faces.jpg')
    cv2.imwrite(comp_assets, avg_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  ✓ Saved {comp_assets}", flush=True)

    comp_web = os.path.join(BASE_DIR, 'web', 'assets', 'composited_1000_faces.jpg')
    if os.path.exists(os.path.dirname(comp_web)):
        cv2.imwrite(comp_web, avg_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  ✓ Saved {comp_web}", flush=True)

    # Also save rigid composite for comparison
    rigid_avg_rgb = rigid_accum / len(aligned_images)
    rigid_avg_bgr = cv2.cvtColor((np.clip(rigid_avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(BASE_DIR, 'composited_1000_faces_rigid.jpg'), rigid_avg_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # 5. Generate 3D Mask Cutout (assets/mask_1000_cutout.png)
    print(f"\n--- 4. Generating 3D Floating Mask Cutout ---", flush=True)
    h, w = avg_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w//2, int(h*0.48)), (int(w*0.36), int(h*0.44)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (65, 65), 0)
    b, g, r = cv2.split(avg_bgr)
    rgba = cv2.merge([b, g, r, mask])

    mask_out = os.path.join(BASE_DIR, 'assets', 'mask_1000_cutout.png')
    cv2.imwrite(mask_out, rgba)
    print(f"  ✓ Saved {mask_out}", flush=True)

    root_mask = os.path.join(BASE_DIR, 'mask_1000_cutout.png')
    cv2.imwrite(root_mask, rgba)

    web_mask = os.path.join(BASE_DIR, 'web', 'assets', 'mask_1000_cutout.png')
    if os.path.exists(os.path.dirname(web_mask)):
        cv2.imwrite(web_mask, rgba)
        print(f"  ✓ Saved {web_mask}", flush=True)

    # 6. Generate 1,000-Face Background Grid (assets/grid_1000.jpg)
    print(f"\n--- 5. Generating 25x40 Tiled Grid (2000x4000) for Homepage Background ---", flush=True)
    cols = 25
    rows = 40
    cw = 80
    ch = 100
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

    for idx, (cell_img, c) in enumerate(zip(aligned_images, final_creators)):
        r_idx = idx // cols
        c_idx = idx % cols
        grid[r_idx*ch:(r_idx+1)*ch, c_idx*cw:(c_idx+1)*cw] = cv2.resize(cell_img, (cw, ch), interpolation=cv2.INTER_AREA)

    grid_out = os.path.join(BASE_DIR, 'assets', 'grid_1000.jpg')
    cv2.imwrite(grid_out, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"  ✓ Saved {grid_out} ({grid.shape[1]}x{grid.shape[0]})", flush=True)

    web_grid = os.path.join(BASE_DIR, 'web', 'assets', 'grid_1000.jpg')
    if os.path.exists(os.path.dirname(web_grid)):
        cv2.imwrite(web_grid, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  ✓ Saved {web_grid}", flush=True)

    cv2.imwrite(os.path.join(BASE_DIR, 'procrustes_1000_faces_grid.jpg'), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 7. Generate Full Markdown Catalog (talking_heads_1000.md)
    print(f"\n--- 6. Generating Official 1,000 Catalog (talking_heads_1000.md) ---", flush=True)
    cat_path = os.path.join(BASE_DIR, 'talking_heads_1000.md')
    with open(cat_path, 'w') as out:
        out.write("# The Most Viral Face — 1,000 Verified Talking-Head Creators\n\n")
        out.write(r"Every creator in this official dataset has been verified to possess $\ge 50,000$ real likes, authentic talking-head front-facing framing, and zero duplicate accounts." + "\n\n")
        out.write("| Rank | Creator Handle | Verified Real Likes | Reel Link | Photo File |\n")
        out.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for c in final_creators:
            h = c.get('handle', '')
            l = c.get('likes', 0)
            url = c.get('reelUrl', f"https://www.instagram.com/reel/{c.get('video_id')}/")
            p = c.get('photo', '')
            out.write(f"| #{c['rank']} | **@{h}** | {l:,} | [{h}]({url}) | `{p}` |\n")

    print(f"  ✓ Generated {cat_path}", flush=True)

    print(f"\n=======================================================")
    print(f"✓ 1,000-FACE REMAKE COMPLETE IN {time.time()-t0_total:.1f}s!")
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    main()
