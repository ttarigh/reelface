#!/usr/bin/env python3
import os
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
USER_REF_IMAGE = "/Users/tarighian/.gemini/jetski/brain/924fbb9c-14cc-4d2d-a2af-89516f152739/media__1791863744931.png"

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
    print("==================================================================", flush=True)
    print("STRICT COMPOSITE OF ONLY THE 1,000 APPROVED CURATION STUDIO FACES", flush=True)
    print("==================================================================", flush=True)
    t0_start = time.time()

    # 1. Load ONLY approved entries from verified_50k_manifest.json
    with open(MANIFEST_PATH) as f:
        mf = json.load(f)

    approved = [v for v in mf.values() if v.get('approved')]
    print(f"Manifest approved entries: {len(approved)} (Zero unapproved included)", flush=True)

    def rank_key(x):
        r = str(x.get('rank', '99999'))
        return int(r) if r.isdigit() else 99999

    approved_sorted = sorted(approved, key=rank_key)[:TARGET_COUNT]
    print(f"Filtered to exactly {len(approved_sorted)} approved creators.", flush=True)

    # Re-index ranks 0001 through 1000
    final_creators = []
    for idx, c in enumerate(approved_sorted):
        c_copy = dict(c)
        c_copy['rank'] = f"{idx+1:04d}"
        final_creators.append(c_copy)

    with open(CREATORS_PATH, 'w') as f:
        json.dump(final_creators, f, indent=2)
    if os.path.exists(os.path.dirname(WEB_CREATORS)):
        with open(WEB_CREATORS, 'w') as f:
            json.dump(final_creators, f, indent=2)

    # 2. Extract Landmarks & Align
    print(f"\n--- Extracting Landmarks for {len(final_creators)} Approved Faces ---", flush=True)
    detector, facemark = load_models()
    aligned_images = []
    aligned_landmarks = []

    t0_detect = time.time()
    for idx, c in enumerate(final_creators):
        p = os.path.join(OUTPUT_DIR, c.get('photo'))
        img = cv2.imread(p)
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)
        fx, fy, fw, fh = [int(v) for v in faces[0][:4]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
        pts = lms[0].reshape(68, 2).astype(np.float64)

        aimg, apts = align_to_canonical_canvas(img, pts)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)

        if (idx + 1) % 250 == 0 or idx == len(final_creators) - 1:
            print(f"  ✓ Extracted [{idx+1:04d}/{len(final_creators)}] in {time.time()-t0_detect:.1f}s", flush=True)

    # 3. GPA Mean Shape
    print(f"\n--- Calculating Generalized Procrustes Mean Shape ---", flush=True)
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)
    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])

    # 4. Piecewise Affine Warping & Milestone Slices
    print(f"\n--- Warping 1,000 Faces into Mean Shape (8 Threads) ---", flush=True)
    t0_warp = time.time()
    warp_tasks = [(aimg, apts, target_full, boundary_anchors) for aimg, apts in zip(aligned_images, aligned_landmarks)]

    milestone_targets = [15, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    milestone_composites = {}

    cum_acc = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, warped_rgb in enumerate(executor.map(warp_worker, warp_tasks)):
            cum_acc += warped_rgb
            current_num = idx + 1
            if current_num in milestone_targets:
                slice_avg = cum_acc / float(current_num)
                slice_bgr = cv2.cvtColor((np.clip(slice_avg, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                milestone_composites[current_num] = slice_bgr
                cv2.imwrite(f"composited_{current_num}_faces.jpg", slice_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if current_num % 250 == 0 or current_num == len(warp_tasks):
                print(f"  ✓ Warped [{current_num:04d}/{len(warp_tasks)}] in {time.time()-t0_warp:.1f}s", flush=True)

    final_1000_bgr = milestone_composites[1000]
    cv2.imwrite("composited_1000_faces.jpg", final_1000_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite("assets/composited_1000_faces.jpg", final_1000_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if os.path.exists("web/assets"):
        cv2.imwrite("web/assets/composited_1000_faces.jpg", final_1000_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("  ✓ Saved official composited_1000_faces.jpg", flush=True)

    # 5. Regenerate master_evolution_15_to_1000.jpg with the ACTUAL curated milestones
    print("\n--- Updating master_evolution_15_to_1000.jpg with Curated Milestone Faces ---", flush=True)
    panel_w, panel_h = 240, 310
    banner_h = 390
    total_w = len(milestone_targets) * panel_w + (len(milestone_targets) + 1) * 15
    master = np.zeros((banner_h, total_w, 3), dtype=np.uint8)

    for idx, m in enumerate(milestone_targets):
        m_img = milestone_composites[m]
        m_res = cv2.resize(m_img, (panel_w, panel_h))
        x = 15 + idx * (panel_w + 15)
        master[55:55+panel_h, x:x+panel_w] = m_res
        label = f"{m} Faces"
        cv2.putText(master, label, (x + panel_w//6, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite("master_evolution_15_to_1000.jpg", master, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  ✓ Updated master_evolution_15_to_1000.jpg with 100% Curation Studio approved faces!", flush=True)

    # 6. Apply Exact User Reference Silhouette & Eye Cutouts
    print("\n--- Applying User Reference Silhouette & Eye Cutouts to 3D Mask ---", flush=True)
    ref = cv2.imread(USER_REF_IMAGE)
    bg_color = np.array([244, 244, 244], dtype=np.float32)
    diff = np.linalg.norm(ref[:, :, :3].astype(np.float32) - bg_color, axis=2)
    raw_mask = (diff > 8).astype(np.uint8)
    flood = raw_mask.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    outer_silhouette = (flood != 2).astype(np.uint8) * 255
    outer_alpha = cv2.GaussianBlur(outer_silhouette, (5, 5), 0)

    # Eye positions in 1000 composite
    src_eyes = np.float32([[CANVAS_W * 0.36, CANVAS_H * 0.40], [CANVAS_W * 0.64, CANVAS_H * 0.40]])
    dst_eyes = np.float32([[310.6, 318.5], [732.6, 313.6]])

    M = cv2.estimateAffinePartial2D(src_eyes.reshape(1, 2, 2), dst_eyes.reshape(1, 2, 2))[0]
    warped_comp = cv2.warpAffine(final_1000_bgr, M, (ref.shape[1], ref.shape[0]), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)

    # Eye cutouts: smaller radius = 32px
    h, w = ref.shape[:2]
    r_eye = 32.0
    y_grid, x_grid = np.ogrid[:h, :w]
    dist_l = np.sqrt((x_grid - 310.6)**2 + (y_grid - 318.5)**2)
    hole_l = np.clip((dist_l - (r_eye - 1.0)) / 1.5, 0.0, 1.0)
    dist_r = np.sqrt((x_grid - 732.6)**2 + (y_grid - 313.6)**2)
    hole_r = np.clip((dist_r - (r_eye - 1.0)) / 1.5, 0.0, 1.0)
    eye_mask = hole_l * hole_r

    final_alpha = (outer_alpha.astype(np.float32) / 255.0) * eye_mask
    final_alpha = np.clip(final_alpha * 255.0, 0, 255).astype(np.uint8)

    ys, xs = np.where(outer_silhouette > 128)
    y1, y2 = ys.min(), ys.max()
    x1, x2 = xs.min(), xs.max()

    cropped_face = warped_comp[y1:y2+1, x1:x2+1]
    cropped_alpha = final_alpha[y1:y2+1, x1:x2+1]
    b, g, r = cv2.split(cropped_face)
    rgba = cv2.merge([b, g, r, cropped_alpha])

    cv2.imwrite('assets/mask_1000_cutout.png', rgba)
    cv2.imwrite('mask_1000_cutout.png', rgba)
    if os.path.exists('web/assets'):
        cv2.imwrite('web/assets/mask_1000_cutout.png', rgba)
    print("  ✓ Saved assets/mask_1000_cutout.png", flush=True)

    # 7. Update 25x40 Tiled Grid (2000x4000)
    print("\n--- Updating 25x40 Tiled Grid from Approved Faces ---", flush=True)
    cols = 25
    rows = 40
    cw = 80
    ch = 100
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)

    for idx, cell_img in enumerate(aligned_images):
        r_idx = idx // cols
        c_idx = idx % cols
        grid[r_idx*ch:(r_idx+1)*ch, c_idx*cw:(c_idx+1)*cw] = cv2.resize(cell_img, (cw, ch), interpolation=cv2.INTER_AREA)

    cv2.imwrite('assets/grid_1000.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if os.path.exists('web/assets'):
        cv2.imwrite('web/assets/grid_1000.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  ✓ Saved assets/grid_1000.jpg", flush=True)

    print(f"\n==================================================================", flush=True)
    print(f"✓ STRICT 1,000 APPROVED CURATION STUDIO COMPOSITE FINISHED IN {time.time()-t0_start:.1f}s!", flush=True)
    print("==================================================================", flush=True)

if __name__ == '__main__':
    main()
