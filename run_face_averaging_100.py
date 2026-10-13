import os
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.spatial import Delaunay

YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

def load_models():
    yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.5)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return yunet, facemark

def extract_landmarks(img, yunet, facemark):
    h, w = img.shape[:2]
    yunet.setInputSize((w, h))
    _, faces = yunet.detect(img)
    if faces is None or len(faces) == 0:
        return None
    f = faces[0]
    fx, fy, fw, fh = [int(v) for v in f[:4]]
    faces_rect = np.array([[fx, fy, fw, fh]], dtype=np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, landmarks = facemark.fit(gray, faces_rect)
    if ok and len(landmarks) > 0:
        return landmarks[0].reshape(68, 2).astype(np.float64)
    return None

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
            print(f"Procrustes shape optimization converged in {it+1} iterations (diff: {diff:.2e}).")
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

    procrustes_target_shape = (mean_shape - mean_center) * scale + target_center
    return procrustes_target_shape

def get_boundary_anchors(w, h):
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 4], [w - 1, h // 4],
        [0, h // 2], [w - 1, h // 2],
        [0, 3 * h // 4], [w - 1, 3 * h // 4],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
    ], dtype=np.float64)

def run_100_averaging():
    INPUT_DIR = "talking_head_photos_100"
    CANVAS_W = 800
    CANVAS_H = 1000

    print("--- 1. Detecting Landmarks on 100 Faces ---")
    yunet, facemark = load_models()
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg") and not f.startswith("composited") and not f.startswith("procrustes")])[:100]
    
    raw_images = []
    raw_landmarks = []
    valid_files = []

    for f in files:
        img = cv2.imread(os.path.join(INPUT_DIR, f))
        if img is None: continue
        pts = extract_landmarks(img, yunet, facemark)
        if pts is not None:
            raw_images.append(img)
            raw_landmarks.append(pts)
            valid_files.append(f)
            print(f"  ✓ Extracted [{len(valid_files)}/100]: {f}")

    N = len(raw_images)
    print(f"\nExtracted landmarks for all {N} images.")

    print("\n--- 2. Rigid Alignment to Canonical Canvas ---")
    aligned_images = []
    aligned_landmarks = []
    for img, pts in zip(raw_images, raw_landmarks):
        aimg, apts = align_to_canonical_canvas(img, pts, CANVAS_W, CANVAS_H)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)

    print("\n--- 3. Generalized Procrustes Analysis (GPA) ---")
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)

    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])

    print("\n--- 4. Piecewise Affine Warping into Procrustean Mean Shape ---")
    warped_images_rgb = []
    for idx, (aimg, apts, fname) in enumerate(zip(aligned_images, aligned_landmarks, valid_files)):
        if (idx + 1) % 10 == 0 or idx == N - 1:
            print(f"  [{idx+1}/{N}] Warping {fname}...")
        src_full = np.vstack([apts, boundary_anchors])
        rgb_float = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        tform = PiecewiseAffineTransform.from_estimate(target_full, src_full)
        warped_rgb = warp(rgb_float, tform, output_shape=(CANVAS_H, CANVAS_W), mode='reflect')
        warped_images_rgb.append(warped_rgb)

    print("\n--- 5. Averaging & Compositing ---")
    stack = np.stack(warped_images_rgb, axis=0)

    # 1. Arithmetic Mean
    avg_rgb = np.mean(stack, axis=0)
    avg_bgr = cv2.cvtColor((np.clip(avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 2. Median Blend
    med_rgb = np.median(stack, axis=0)
    med_bgr = cv2.cvtColor((np.clip(med_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 3. Portrait 9:16 Crop
    portrait_h = int(CANVAS_W * 16 / 9)
    portrait_canvas = np.zeros((portrait_h, CANVAS_W, 3), dtype=np.uint8)
    y_off = (portrait_h - CANVAS_H) // 2
    portrait_canvas[y_off:y_off+CANVAS_H, :] = avg_bgr

    # Save outputs
    cv2.imwrite("composited_100_faces.jpg", avg_bgr)
    cv2.imwrite("composited_100_faces_median.jpg", med_bgr)
    cv2.imwrite("composited_100_faces_portrait.jpg", portrait_canvas)
    cv2.imwrite(os.path.join(INPUT_DIR, "composited_100_faces.jpg"), avg_bgr)
    print("  ✓ Saved composited_100_faces.jpg")
    print("  ✓ Saved composited_100_faces_median.jpg")
    print("  ✓ Saved composited_100_faces_portrait.jpg")

    print("\n--- 6. Generating Visualizations & Grid ---")
    tri = Delaunay(target_full)
    mesh_vis = avg_bgr.copy()
    for s in tri.simplices:
        pts_tri = np.int32(target_full[s])
        cv2.polylines(mesh_vis, [pts_tri], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    for pt in procrustes_mean:
        cv2.circle(mesh_vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite("composited_100_faces_landmarks.jpg", mesh_vis)
    print("  ✓ Saved composited_100_faces_landmarks.jpg")

    # 10x10 Grid for all 100 individual faces
    grid_cols = 10
    grid_rows = 10
    cell_w, cell_h = 160, 200
    grid = np.zeros((cell_h * grid_rows, cell_w * grid_cols, 3), dtype=np.uint8)

    for idx, (cell_img, fname) in enumerate(zip(aligned_images[:100], valid_files[:100])):
        row = idx // grid_cols
        col = idx % grid_cols
        resized = cv2.resize(cell_img, (cell_w, cell_h))
        title = fname.split("_", 1)[1].replace(".jpg", "")[:12]
        cv2.rectangle(resized, (0, cell_h - 20), (cell_w, cell_h), (20, 20, 20), -1)
        cv2.putText(resized, f"{idx+1:03d} {title}", (4, cell_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (240, 240, 240), 1, cv2.LINE_AA)
        grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized

    cv2.imwrite("procrustes_100_faces_grid.jpg", grid)
    print("  ✓ Saved procrustes_100_faces_grid.jpg (100 individual talking heads)")

    # Comparison poster: 15 Faces Composite vs 100 Faces Composite
    if os.path.exists("composited_15_faces.jpg"):
        c15 = cv2.imread("composited_15_faces.jpg")
        c100 = avg_bgr
        post_w = 1200
        post_h = 750
        comp_poster = np.zeros((post_h, post_w, 3), dtype=np.uint8)
        c15_res = cv2.resize(c15, (520, 650))
        c100_res = cv2.resize(c100, (520, 650))
        comp_poster[50:700, 50:570] = c15_res
        comp_poster[50:700, 630:1150] = c100_res
        cv2.putText(comp_poster, "15 Faces Composite", (160, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(comp_poster, "100 Faces Composite", (740, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.imwrite("milestone_15_vs_100_comparison.jpg", comp_poster)
        print("  ✓ Saved milestone_15_vs_100_comparison.jpg")

    print("\nPROCRUSTEAN 100-FACE COMPOSITING COMPLETE!")

if __name__ == "__main__":
    run_100_averaging()
