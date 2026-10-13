import os
import cv2
import numpy as np
from scipy.spatial import Delaunay

# -------------------------------------------------------------
# 1. Landmark Detection (YuNet + FacemarkLBF 68 Points)
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
    # Use the highest confidence / largest face
    f = faces[0]
    fx, fy, fw, fh = [int(v) for v in f[:4]]
    faces_rect = np.array([[fx, fy, fw, fh]], dtype=np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, landmarks = facemark.fit(gray, faces_rect)
    if ok and len(landmarks) > 0:
        return landmarks[0].reshape(68, 2).astype(np.float64)
    return None

# -------------------------------------------------------------
# 2. Generalized Procrustes Analysis (GPA)
# -------------------------------------------------------------
def align_shapes_procrustes(shapes, max_iter=100, tol=1e-6):
    """
    Generalized Procrustes Analysis to compute the canonical mean shape
    invariant to translation, uniform scale, and rotation.
    """
    n_shapes = len(shapes)
    n_points = shapes[0].shape[0]

    # Step 1: Center each shape at (0, 0) and normalize centroid size to 1.0
    centered = []
    for s in shapes:
        c = np.mean(s, axis=0)
        s_cent = s - c
        size = np.sqrt(np.sum(s_cent ** 2))
        centered.append(s_cent / size)

    # Initial estimate of mean shape is the first centered shape
    mean_shape = centered[0].copy()

    for it in range(max_iter):
        aligned_shapes = []
        for s in centered:
            # Align s to mean_shape using SVD (Ordinary Procrustes Analysis)
            # M = s^T * mean_shape
            M = np.dot(s.T, mean_shape)
            U, _, Vt = np.linalg.svd(M)
            # Rotation matrix R
            R = np.dot(U, Vt)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = np.dot(U, Vt)
            s_aligned = np.dot(s, R)
            aligned_shapes.append(s_aligned)

        # Compute new mean shape
        new_mean = np.mean(aligned_shapes, axis=0)
        # Normalize new mean shape
        new_mean -= np.mean(new_mean, axis=0)
        new_mean /= np.sqrt(np.sum(new_mean ** 2))

        # Check convergence
        diff = np.linalg.norm(new_mean - mean_shape)
        mean_shape = new_mean
        if diff < tol:
            print(f"Procrustes converged in {it+1} iterations (diff: {diff:.2e})")
            break

    return mean_shape, aligned_shapes

# -------------------------------------------------------------
# 3. Canvas Target Coordinates & Boundary Anchors
# -------------------------------------------------------------
def get_target_landmarks(mean_shape, canvas_w=800, canvas_h=1000):
    """
    Scale and translate the normalized Procrustes mean shape into
    canonical canvas coordinates (centered portrait framing).
    """
    # Landmarks 36:42 are left eye (subject right eye), 42:48 are right eye
    # Average eye centers in 68-point model:
    # Right eye (landmarks 36 to 41), Left eye (landmarks 42 to 47)
    re_center = np.mean(mean_shape[36:42], axis=0)
    le_center = np.mean(mean_shape[42:48], axis=0)

    # Target eye positions in the canvas
    target_eye_y = canvas_h * 0.40  # 40% from top
    target_eye_dist = canvas_w * 0.28  # Interocular distance ~28% of canvas width
    target_re = np.array([canvas_w * 0.5 - target_eye_dist / 2.0, target_eye_y])
    target_le = np.array([canvas_w * 0.5 + target_eye_dist / 2.0, target_eye_y])

    # Find similarity transform from mean_shape eye centers to target eye positions
    curr_eye_dist = np.linalg.norm(le_center - re_center)
    scale = target_eye_dist / curr_eye_dist

    # Rotation angle
    curr_dx = le_center[0] - re_center[0]
    curr_dy = le_center[1] - re_center[1]
    angle = np.arctan2(curr_dy, curr_dx)

    # Center between eyes
    curr_center = (re_center + le_center) / 2.0
    target_center = (target_re + target_le) / 2.0

    # Transform all points:
    # 1. Translate so eye center is at (0, 0)
    # 2. Scale
    # 3. Rotate by -angle (to level eyes)
    # 4. Translate to target_center
    transformed = (mean_shape - curr_center) * scale
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    transformed = np.dot(transformed, R.T) + target_center

    return transformed

def add_boundary_points(landmarks, w, h):
    """
    Add boundary points around the canvas perimeter to ensure
    Delaunay triangulation covers the entire canvas.
    """
    boundary_pts = np.array([
        [0, 0], [w // 2, 0], [w - 1, 0],
        [0, h // 4], [w - 1, h // 4],
        [0, h // 2], [w - 1, h // 2],
        [0, 3 * h // 4], [w - 1, 3 * h // 4],
        [0, h - 1], [w // 2, h - 1], [w - 1, h - 1]
    ], dtype=np.float64)

    return np.vstack([landmarks, boundary_pts])

# -------------------------------------------------------------
# 4. Piecewise Affine Warping
# -------------------------------------------------------------
def warp_triangle(img_src, img_dst, t_src, t_dst):
    """
    Warps pixels inside triangle t_src from img_src into t_dst in img_dst.
    """
    # Find bounding box for each triangle
    r_src = cv2.boundingRect(np.float32([t_src]))
    r_dst = cv2.boundingRect(np.float32([t_dst]))

    # Offset points by top-left coordinates of bounding box
    t_src_rect = []
    t_dst_rect = []
    for i in range(3):
        t_src_rect.append(((t_src[i][0] - r_src[0]), (t_src[i][1] - r_src[1])))
        t_dst_rect.append(((t_dst[i][0] - r_dst[0]), (t_dst[i][1] - r_dst[1])))

    # Get mask by filling triangle
    mask = np.zeros((r_dst[3], r_dst[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_dst_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warp to small rectangular patch
    img_src_rect = img_src[r_src[1]:r_src[1] + r_src[3], r_src[0]:r_src[0] + r_src[2]]

    size = (r_dst[2], r_dst[3])
    warp_mat = cv2.getAffineTransform(np.float32(t_src_rect), np.float32(t_dst_rect))
    img_dst_rect = cv2.warpAffine(
        img_src_rect, warp_mat, size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )

    # Copy triangular region of the rectangular patch to the destination image
    img_dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]] = (
        img_dst[r_dst[1]:r_dst[1] + r_dst[3], r_dst[0]:r_dst[0] + r_dst[2]] * (1 - mask) +
        img_dst_rect * mask
    )

def warp_image_to_target(img, src_points, target_points, triangles, canvas_w, canvas_h):
    """
    Warps the entire image using piecewise affine mapping over Delaunay triangles.
    """
    warped = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)

    for tri_indices in triangles:
        t_src = src_points[tri_indices]
        t_dst = target_points[tri_indices]
        warp_triangle(img, warped, t_src, t_dst)

    return warped

def get_corresponding_source_boundaries(src_landmarks, target_landmarks_with_bounds, target_shape_no_bounds):
    """
    Maps canonical target boundary points back into the source image coordinate space
    using the global similarity transform between target face and source face.
    """
    # Compute similarity transform from target face to source face
    M, _ = cv2.estimateAffinePartial2D(target_shape_no_bounds, src_landmarks)
    boundary_target = target_landmarks_with_bounds[68:] # the 12 boundary points
    ones = np.ones((boundary_target.shape[0], 1))
    homo = np.hstack([boundary_target, ones])
    src_boundaries = np.dot(homo, M.T)
    return np.vstack([src_landmarks, src_boundaries])

print("Module defined successfully.")
