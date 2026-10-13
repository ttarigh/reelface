import os
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp

# -------------------------------------------------------------
# 1. Models & Landmark Detection
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# 2. Similarity Alignment to Canonical Canvas
# -------------------------------------------------------------
def align_to_canonical_canvas(img, landmarks, canvas_w=800, canvas_h=1000):
    """
    Transforms the image and its landmarks so that the eyes are horizontally
    aligned at a fixed canonical position.
    """
    # Landmarks 36:42 are subject right eye, 42:48 are subject left eye
    re_center = np.mean(landmarks[36:42], axis=0) # [x, y]
    le_center = np.mean(landmarks[42:48], axis=0) # [x, y]

    target_re = np.array([canvas_w * 0.36, canvas_h * 0.40])
    target_le = np.array([canvas_w * 0.64, canvas_h * 0.40])

    # Compute similarity transform
    src_pts = np.float32([re_center, le_center])
    dst_pts = np.float32([target_re, target_le])

    M = cv2.estimateAffinePartial2D(src_pts.reshape(1, 2, 2), dst_pts.reshape(1, 2, 2))[0]

    # Warp image to canvas
    aligned_img = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Transform landmarks
    ones = np.ones((68, 1))
    homo = np.hstack([landmarks, ones])
    aligned_landmarks = np.dot(homo, M.T)

    return aligned_img, aligned_landmarks

# -------------------------------------------------------------
# 3. Generalized Procrustes Analysis (GPA)
# -------------------------------------------------------------
def generalized_procrustes_analysis(shapes, max_iter=100, tol=1e-7):
    """
    Computes the Procrustes average shape from pre-aligned shapes.
    """
    # Center each shape
    centered = [s - np.mean(s, axis=0) for s in shapes]
    # Unit centroid size
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

    # Scale and place mean shape back to match the average eye positions of aligned shapes
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

# -------------------------------------------------------------
# 4. Canvas Boundary Anchors
# -------------------------------------------------------------
def get_boundary_anchors(w, h):
    return np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 4], [w - 1, h // 4],
        [0, h // 2], [w - 1, h // 2],
        [0, 3 * h // 4], [w - 1, 3 * h // 4],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
    ], dtype=np.float64)

# -------------------------------------------------------------
# 5. Full Pipeline
# -------------------------------------------------------------
def run_procrustes_compositing():
    INPUT_DIR = "talking_head_photos"
    CANVAS_W = 800
    CANVAS_H = 1000

    print("--- 1. Detecting Landmarks ---")
    yunet, facemark = load_models()
    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".jpg") and not f.startswith("composited") and not f.startswith("procrustes")])
    
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
            print(f"  ✓ Extracted: {f}")

    N = len(raw_images)
    print(f"\nExtracted landmarks for all {N}/15 images.")

    print("\n--- 2. Rigid Alignment to Canonical Canvas ---")
    aligned_images = []
    aligned_landmarks = []
    for img, pts in zip(raw_images, raw_landmarks):
        aimg, apts = align_to_canonical_canvas(img, pts, CANVAS_W, CANVAS_H)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)

    print("\n--- 3. Generalized Procrustes Analysis (GPA) ---")
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)

    # Add canvas boundary anchors to Procrustes mean shape
    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])

    print("\n--- 4. Piecewise Affine Warping into Procrustean Mean Shape ---")
    warped_images_rgb = []
    for idx, (aimg, apts, fname) in enumerate(zip(aligned_images, aligned_landmarks, valid_files)):
        print(f"  [{idx+1}/{N}] Warping {fname}...")
        src_full = np.vstack([apts, boundary_anchors])

        # Convert to RGB float in [0, 1]
        rgb_float = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

        # Estimate piecewise affine transform from target (mean) to source
        tform = PiecewiseAffineTransform.from_estimate(target_full, src_full)
        warped_rgb = warp(rgb_float, tform, output_shape=(CANVAS_H, CANVAS_W), mode='reflect')
        warped_images_rgb.append(warped_rgb)

    print("\n--- 5. Averaging & Compositing ---")
    stack = np.stack(warped_images_rgb, axis=0) # (15, H, W, 3)

    # 1. Arithmetic Mean
    avg_rgb = np.mean(stack, axis=0)
    avg_bgr = cv2.cvtColor((np.clip(avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 2. Median Blend (cleaner skin textures, removes any outlier glare or shadows)
    med_rgb = np.median(stack, axis=0)
    med_bgr = cv2.cvtColor((np.clip(med_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Save outputs
    cv2.imwrite("composited_15_faces.jpg", avg_bgr)
    cv2.imwrite("composited_15_faces_median.jpg", med_bgr)
    cv2.imwrite(os.path.join(INPUT_DIR, "composited_15_faces.jpg"), avg_bgr)
    print("  ✓ Saved composited_face.jpg")
    print("  ✓ Saved composited_face_median.jpg")

    print("\n--- 6. Generating Visualizations & Grid ---")
    from scipy.spatial import Delaunay
    tri = Delaunay(target_full)
    mesh_vis = avg_bgr.copy()
    for s in tri.simplices:
        pts_tri = np.int32(target_full[s])
        cv2.polylines(mesh_vis, [pts_tri], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    for pt in procrustes_mean:
        cv2.circle(mesh_vis, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.imwrite("composited_15_faces_landmarks.jpg", mesh_vis)
    print("  ✓ Saved composited_face_landmarks.jpg (Delaunay mesh)")

    # 4x4 Grid: 15 individual aligned faces + 1 composited face
    cell_w, cell_h = 240, 300
    grid = np.zeros((cell_h * 4, cell_w * 4, 3), dtype=np.uint8)

    all_cells = aligned_images + [avg_bgr]
    titles = [f.split("_")[1].replace(".jpg", "") for f in valid_files] + ["★ COMPOSITED FACE"]

    for idx, (cell_img, title) in enumerate(zip(all_cells, titles)):
        row = idx // 4
        col = idx % 4
        resized = cv2.resize(cell_img, (cell_w, cell_h))
        if idx == 15:
            cv2.rectangle(resized, (0, 0), (cell_w - 1, cell_h - 1), (0, 215, 255), 4)
        cv2.rectangle(resized, (0, cell_h - 26), (cell_w, cell_h), (20, 20, 20), -1)
        color = (0, 215, 255) if idx == 15 else (240, 240, 240)
        cv2.putText(resized, title[:18], (6, cell_h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized

    cv2.imwrite("procrustes_15_faces_grid.jpg", grid)
    print("  ✓ Saved procrustes_grid_15.jpg (16-panel grid)")

    print("\nPROCRUSTEAN FACE AVERAGING COMPLETE!")

if __name__ == "__main__":
    run_procrustes_compositing()
