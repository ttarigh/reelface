#!/usr/bin/env python3
import os
import sys
import json
import shutil
import glob
import time
import fcntl
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "talking_head_photos_100")
OLD_UNVERIFIED_DIR = os.path.join(BASE_DIR, "old_unverified")
MANIFEST_PATH = os.path.join(BASE_DIR, "assets", "verified_50k_manifest.json")
CREATORS_PATH = os.path.join(BASE_DIR, "assets", "creators.json")
LOCK_PATH = os.path.join(BASE_DIR, "assets", "manifest.lock")

class FileLock:
    def __enter__(self):
        self.fp = open(LOCK_PATH, "w")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
        except: pass

class CurationHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-cache, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == '/api/creators':
            try:
                with open(MANIFEST_PATH) as f:
                    manifest = json.load(f)

                bl_path = os.path.join(BASE_DIR, "assets", "rejected_blacklist.json")
                blacklisted_handles = set()
                if os.path.exists(bl_path):
                    with open(bl_path) as bf:
                        bl_data = json.load(bf)
                        blacklisted_handles = {h.lower() for h in bl_data.get("handles", [])}

                # Only return creators whose photo exists on disk and is NOT blacklisted
                valid_creators = []
                for c in manifest.values():
                    p_name = c.get('photo', '')
                    h = c.get('handle', '').lower()
                    if h in blacklisted_handles:
                        continue
                    if os.path.exists(os.path.join(OUTPUT_DIR, p_name)):
                        valid_creators.append(c)

                data = json.dumps({'success': True, 'creators': valid_creators, 'count': len(valid_creators)}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        super().do_GET()

    def do_POST(self):
        if self.path == '/api/approve':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                payload = json.loads(body)
                photo_name = payload.get('photo')
                rank = payload.get('rank')

                with FileLock():
                    with open(MANIFEST_PATH) as f:
                        manifest = json.load(f)

                    found = False
                    for k, v in manifest.items():
                        if v.get('photo') == photo_name or k == rank:
                            v['approved'] = True
                            found = True
                            break

                    if found:
                        tmp_mf = f"{MANIFEST_PATH}.srv_{time.time()}.tmp"
                        with open(tmp_mf, 'w') as f:
                            json.dump(manifest, f, indent=2)
                        os.replace(tmp_mf, MANIFEST_PATH)

                resp = json.dumps({'success': True, 'photo': photo_name}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return
            except Exception as e:
                self.send_error(500, str(e))
                return

        elif self.path == '/api/reject':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                payload = json.loads(body)
                photo_name = payload.get('photo')
                rank = payload.get('rank')

                if not photo_name:
                    self.send_error(400, "Missing photo parameter")
                    return

                src_photo = os.path.join(OUTPUT_DIR, photo_name)
                dst_photo = os.path.join(OLD_UNVERIFIED_DIR, photo_name)

                os.makedirs(OLD_UNVERIFIED_DIR, exist_ok=True)
                if os.path.exists(src_photo):
                    shutil.move(src_photo, dst_photo)

                with FileLock():
                    # Remove from manifest and capture metadata for blacklist
                    with open(MANIFEST_PATH) as f:
                        manifest = json.load(f)

                    rejected_vid = None
                    rejected_handle = photo_name.replace('.jpg', '').split('_', 1)[-1].lower()

                    keys_to_del = [k for k, v in manifest.items() if v.get('photo') == photo_name or k == rank]
                    for k in keys_to_del:
                        rejected_vid = manifest[k].get('video_id')
                        del manifest[k]

                    tmp_mf = f"{MANIFEST_PATH}.srv_{time.time()}.tmp"
                    with open(tmp_mf, 'w') as f:
                        json.dump(manifest, f, indent=2)
                    os.replace(tmp_mf, MANIFEST_PATH)

                # Append to permanent blacklist
                bl_path = os.path.join(BASE_DIR, "assets", "rejected_blacklist.json")
                if os.path.exists(bl_path):
                    with open(bl_path) as bf:
                        bl_data = json.load(bf)
                else:
                    bl_data = {"video_ids": [], "handles": []}

                bl_handles = set(bl_data.get("handles", []))
                bl_vids = set(bl_data.get("video_ids", []))
                bl_handles.add(rejected_handle)
                if rejected_vid: bl_vids.add(rejected_vid)

                bl_data["handles"] = sorted(list(bl_handles))
                bl_data["video_ids"] = sorted(list(bl_vids))
                with open(bl_path, 'w') as bf:
                    json.dump(bl_data, bf, indent=2)

                creators_list = list(manifest.values())
                tmp_cf = CREATORS_PATH + '.tmp'
                with open(tmp_cf, 'w') as f:
                    json.dump(creators_list, f, indent=2)
                os.replace(tmp_cf, CREATORS_PATH)

                web_creators = os.path.join(BASE_DIR, 'web', 'assets', 'creators.json')
                if os.path.exists(os.path.dirname(web_creators)):
                    with open(web_creators, 'w') as f:
                        json.dump(creators_list, f, indent=2)

                print(f"[CURATION API] Permanently Blacklisted & Rejected: @{rejected_handle} (Remaining in DB: {len(manifest)})", flush=True)

                resp = json.dumps({'success': True, 'photo': photo_name, 'remaining': len(manifest)}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return

            except Exception as e:
                self.send_error(500, str(e))
                return

        elif self.path == '/api/recomposite':
            try:
                import cv2
                import numpy as np

                MODEL_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
                CANVAS_W, CANVAS_H = 800, 1000
                detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)

                canonical_eye_r = np.array([285.0, 400.0], dtype=np.float32)
                canonical_eye_l = np.array([513.0, 397.0], dtype=np.float32)
                dst_eyes = np.array([canonical_eye_r, canonical_eye_l], dtype=np.float32)

                with open(MANIFEST_PATH) as f:
                    manifest = json.load(f)
                approved_photos = {v.get('photo') for v in manifest.values() if v.get('approved')}
                photos = [os.path.join(OUTPUT_DIR, fn) for fn in approved_photos if fn and os.path.exists(os.path.join(OUTPUT_DIR, fn))]
                aligned = []

                for p in photos:
                    img = cv2.imread(p)
                    if img is None: continue
                    h, w = img.shape[:2]
                    detector.setInputSize((w, h))
                    _, faces = detector.detect(img)
                    if faces is not None and len(faces) > 0:
                        f = faces[0]
                        re_pt = np.array([f[4], f[5]], dtype=np.float32)
                        le_pt = np.array([f[6], f[7]], dtype=np.float32)
                        src_eyes = np.array([re_pt, le_pt], dtype=np.float32)
                        M = cv2.estimateAffinePartial2D(src_eyes.reshape(1, 2, 2), dst_eyes.reshape(1, 2, 2))[0]
                        if M is not None:
                            warped = cv2.warpAffine(img, M, (CANVAS_W, CANVAS_H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
                            aligned.append(warped.astype(np.float32))

                if aligned:
                    avg = np.mean(aligned, axis=0)
                    avg_bgr = np.clip(avg, 0, 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(BASE_DIR, 'composited_1000_faces.jpg'), avg_bgr)

                    # Update mask
                    h, w = avg_bgr.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.ellipse(mask, (w//2, int(h*0.48)), (int(w*0.36), int(h*0.44)), 0, 0, 360, 255, -1)
                    mask = cv2.GaussianBlur(mask, (65, 65), 0)
                    b, g, r = cv2.split(avg_bgr)
                    rgba = cv2.merge([b, g, r, mask])
                    mask_out = os.path.join(BASE_DIR, 'assets', 'mask_1000_cutout.png')
                    cv2.imwrite(mask_out, rgba)
                    web_mask = os.path.join(BASE_DIR, 'web', 'assets', 'mask_1000_cutout.png')
                    if os.path.exists(os.path.dirname(web_mask)):
                        cv2.imwrite(web_mask, rgba)

                resp = json.dumps({'success': True, 'count': len(aligned)}).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
                return

            except Exception as e:
                self.send_error(500, str(e))
                return

        super().do_POST()

def run(port=8080):
    os.chdir(BASE_DIR)
    server_address = ('0.0.0.0', port)
    httpd = ThreadingHTTPServer(server_address, CurationHandler)
    print(f"Curation Server running on http://localhost:{port}/ (curate at http://localhost:{port}/curate.html)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.", flush=True)
        httpd.server_close()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run(port)
