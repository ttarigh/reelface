import cv2
import os
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_detection_yunet_2023mar.onnx")

video_url = 'https://scontent-lga3-3.cdninstagram.com/o1/v/t2/f2/m86/AQOW6kgj9ZCJm_qu0mAY2U8BIITBxpRmMBkdfePR51KXcT6VTTEfaLahkCBUusuSk-OtIbTQbmhKtO_1TctGAUY4A5cSzQ1QAllNrsw.mp4?_nc_cat=104&_nc_sid=5e9851&_nc_ht=scontent-lga3-3.cdninstagram.com&_nc_ohc=FvJKJ_A7Jp0Q7kNvwGN-Kc9&efg=eyJ2ZW5jb2RlX3RhZyI6Inhwdl9wcm9ncmVzc2l2ZS5JTlNUQUdSQU0uQ0xJUFMuQzMuNzIwLmRhc2hfYmFzZWxpbmVfMV92MSIsInhwdl9hc3NldF9pZCI6MTc5NTM3NDgzNjQwOTg0NzksImFzc2V0X2FnZV9kYXlzIjoxOTIsInZpX3VzZWNhc2VfaWQiOjEwMDk5LCJkdXJhdGlvbl9zIjo0MywidXJsZ2VuX3NvdXJjZSI6Ind3dyJ9&ccb=17-1&vs=d98ca1025b578e33&_nc_vs=HBksFQIYUmlnX3hwdl9yZWVsc19wZXJtYW5lbnRfc3JfcHJvZC8wMjQwMTdCQTZCRjNENkY3NjA5MDVGN0RGQjVDQjRBRV92aWRlb19kYXNoaW5pdC5tcDQVAALIARIAFQIYUWlnX3hwdl9wbGFjZW1lbnRfcGVybWFuZW50X3YyL0VENEFCRTJBQTlBOUY2MzIxQTVBNkEzQzg1RUFGQkIzX2F1ZGlvX2Rhc2hpbml0Lm1wNBUCAsgBEgAoABgAGwKIB3VzZV9vaWwBMRJwcm9ncmVzc2l2ZV9yZWNpcGUBMRUAACbe7oea2bXkPxUCKAJDMywXQEWIcrAgxJwYEmRhc2hfYmFzZWxpbmVfMV92MREAdf4HZeadAQA&_nc_gid=mqkt-1ysbjc5hBchA47LKg&_nc_ss=7f689&_nc_zt=28&oh=00_AQIiCrns7XQRghiFBtO8m2MpgXhiB0u5e1WMcbrOtqckeQ&oe=6AB3C6F5'

cap = cv2.VideoCapture(video_url)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Total frames: {total_frames}, FPS: {fps}")

detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.7)

frames_checked = 0
best_frame = None
best_score = -1.0
best_face_data = None

# Sample 15 frames evenly across the video
step = max(1, total_frames // 15) if total_frames > 0 else 30

for f_idx in range(0, min(total_frames, 300), step):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    retval, faces = detector.detect(frame)
    if faces is None or len(faces) == 0:
        continue

    # Evaluate faces
    for face in faces:
        conf = float(face[-1])
        fx, fy, fw, fh = face[0:4]
        if conf > best_score:
            best_score = conf
            best_frame = frame.copy()
            best_face_data = face

cap.release()

print(f"Best face score: {best_score}")
if best_frame is not None:
    os.makedirs("scratch/test_out", exist_ok=True)
    cv2.imwrite("scratch/test_out/full_best.jpg", best_frame)
    fx, fy, fw, fh = [int(v) for v in best_face_data[0:4]]
    # Crop headshot with margin
    h, w = best_frame.shape[:2]
    pad_x = int(fw * 0.4)
    pad_y_top = int(fh * 0.6)
    pad_y_bot = int(fh * 0.4)
    x1 = max(0, fx - pad_x)
    y1 = max(0, fy - pad_y_top)
    x2 = min(w, fx + fw + pad_x)
    y2 = min(h, fy + fh + pad_y_bot)
    crop = best_frame[y1:y2, x1:x2]
    cv2.imwrite("scratch/test_out/face_crop.jpg", crop)
    print(f"Saved test output to scratch/test_out, crop size: {crop.shape}")
