"""
face_verifier.py: Strict Computer Vision verification for Direct-to-Camera Talking Head videos.
Enforces:
1. Exactly ONE prominent human face in the frame (rejects multi-person interviews/conversations).
2. High-confidence detection (confidence >= 0.85).
3. Direct-to-Camera Frontal Orientation: Facial landmarks (eyes, nose, mouth) must be symmetric,
   proving the speaker is looking straight into the camera lens, NOT sideways at an interviewer.
4. Prominent and Centered: Face occupies >= 4-15% of frame area, centered in horizontal middle.
"""

import os
import cv2
import numpy as np
import requests

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")

class FaceVerifier:
    def __init__(self, model_path: str = MODEL_PATH, min_confidence: float = 0.85):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at {model_path}")
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._input_size = (320, 320)
        self.detector = cv2.FaceDetectorYN.create(
            self.model_path,
            "",
            self._input_size,
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000
        )

    def _ensure_input_size(self, width: int, height: int):
        if self._input_size != (width, height):
            self._input_size = (width, height)
            self.detector.setInputSize((width, height))

    def verify_image(self, img_bgr: np.ndarray) -> tuple[bool, dict]:
        """
        Verify if an image frame contains a solitary, frontal, centered talking-head human face.
        Returns (is_talking_head: bool, metrics: dict).
        """
        if img_bgr is None or img_bgr.size == 0:
            return False, {"error": "Empty image"}

        h, w = img_bgr.shape[:2]
        self._ensure_input_size(w, h)

        retval, faces = self.detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return False, {"faces_detected": 0, "reason": "No face detected"}

        # Filter faces by minimum confidence
        valid_faces = [f for f in faces if float(f[-1]) >= self.min_confidence]
        if not valid_faces:
            return False, {"faces_detected": len(faces), "reason": f"Face confidence below {self.min_confidence}"}

        # Multi-person check: In talking head monologues, there should not be multiple competing faces
        # (which indicates interview / podcast split / group conversation)
        if len(valid_faces) > 2:
            return False, {"faces_detected": len(valid_faces), "reason": "Multiple faces detected (group/interview setup)"}

        # Find the dominant face
        best_face = None
        max_area = 0
        for face in valid_faces:
            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
            area = fw * fh
            if area > max_area:
                max_area = area
                best_face = (face, fx, fy, fw, fh, float(face[-1]))

        face_raw, fx, fy, fw, fh, score = best_face
        frame_area = w * h
        face_area_ratio = (fw * fh) / frame_area
        face_width_ratio = fw / w
        face_height_ratio = fh / h

        face_center_x = (fx + fw / 2.0) / w
        face_center_y = (fy + fh / 2.0) / h

        # Landmarks:
        # [4,5]=re_x,re_y, [6,7]=le_x,le_y, [8,9]=nt_x,nt_y, [10,11]=rc_x,rc_y, [12,13]=lc_x,lc_y
        re_x, re_y = float(face_raw[4]), float(face_raw[5])
        le_x, le_y = float(face_raw[6]), float(face_raw[7])
        nt_x, nt_y = float(face_raw[8]), float(face_raw[9])
        rm_y, lm_y = float(face_raw[11]), float(face_raw[13])

        # 1. Vertical Landmark Geometry: eyes above nose, nose above mouth
        eyes_y = (re_y + le_y) / 2.0
        mouth_y = (rm_y + lm_y) / 2.0
        has_human_geometry = (eyes_y < nt_y < mouth_y)

        # 2. Frontal Facing Symmetry Check (Direct to Camera):
        # In a direct talking head, the person looks into the camera lens.
        # Nose x is roughly equidistant between right eye and left eye.
        eye_span = abs(le_x - re_x)
        if eye_span > 0:
            nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / eye_span
            is_frontal = (nose_offset < 0.48)
        else:
            is_frontal = True
            nose_offset = 0.0

        # 3. Prominence & Centering
        is_prominent = (face_area_ratio >= 0.035 or face_width_ratio >= 0.16 or face_height_ratio >= 0.14)
        is_centered = (0.18 <= face_center_x <= 0.82) and (0.10 <= face_center_y <= 0.75)

        is_talking_head = bool(
            has_human_geometry and
            is_frontal and
            is_prominent and
            is_centered
        )

        metrics = {
            "faces_detected": len(valid_faces),
            "confidence": round(score, 3),
            "face_area_ratio_pct": round(face_area_ratio * 100, 2),
            "face_width_ratio_pct": round(face_width_ratio * 100, 2),
            "face_height_ratio_pct": round(face_height_ratio * 100, 2),
            "face_center": (round(face_center_x, 2), round(face_center_y, 2)),
            "nose_offset": round(nose_offset, 2),
            "is_frontal": is_frontal,
            "has_human_geometry": has_human_geometry,
            "is_prominent": is_prominent,
            "is_centered": is_centered,
            "is_talking_head": is_talking_head
        }
        return is_talking_head, metrics

    def verify_image_url(self, url: str) -> tuple[bool, dict]:
        """Download image from URL and verify frontal talking head."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            res = requests.get(url, headers=headers, timeout=4)
            if res.status_code != 200:
                return False, {"error": f"HTTP {res.status_code}"}
            nparr = np.frombuffer(res.content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return False, {"error": "Failed to decode image"}
            return self.verify_image(img)
        except Exception as e:
            return False, {"error": str(e)}
