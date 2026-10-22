import json, os, subprocess, time
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(BASE_DIR, "assets", "creators.json")) as f:
    creators = json.load(f)

font_path = "/System/Library/Fonts/HelveticaNeue.ttc"

def render_video(w, h, duration_sec, fps, out_filename, font_size=52, line_h=80):
    total_frames = int(duration_sec * fps)
    
    title_font = ImageFont.truetype(font_path, int(font_size * 1.35))
    sub_font = ImageFont.truetype(font_path, int(font_size * 0.75))
    handle_font = ImageFont.truetype(font_path, font_size)
    
    # Layout Y coordinates
    items = []
    # Header
    items.append({"type": "title", "text": "THE MOST VIRAL FACE", "y": h * 0.45, "font": title_font})
    items.append({"type": "sub", "text": "1,000 CREATORS", "y": h * 0.45 + font_size * 1.8, "font": sub_font})
    
    start_y = h * 0.45 + font_size * 4.5
    for i, c in enumerate(creators):
        items.append({
            "type": "handle",
            "text": "@" + c["handle"],
            "y": start_y + i * line_h,
            "font": handle_font
        })
    
    end_y = start_y + len(creators) * line_h + 300
    items.append({"type": "sub", "text": "SYNTHESIZED BY", "y": end_y, "font": sub_font})
    items.append({"type": "title", "text": "PROCRUSTES ANALYSIS", "y": end_y + font_size * 1.5, "font": title_font})
    
    total_scroll = end_y + 600
    scroll_speed = total_scroll / (total_frames - 1)
    
    out_path = os.path.join(BASE_DIR, out_filename)
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "fast",
        "-crf", "19",
        out_path
    ]
    
    print(f"Rendering {out_filename} ({w}x{h}, {total_frames} frames)...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    frame = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(frame)
    
    for f_idx in range(total_frames):
        scrollY = f_idx * scroll_speed
        
        # Clear frame
        draw.rectangle([0, 0, w, h], fill=(0, 0, 0))
        
        # Draw items currently visible on screen
        for item in items:
            sy = item["y"] - scrollY
            if -120 <= sy <= h + 80:
                text = item["text"]
                bbox = item["font"].getbbox(text)
                tw = bbox[2] - bbox[0]
                tx = (w - tw) // 2
                
                # Soft vertical fade at top & bottom 12% of screen
                edge_margin = h * 0.12
                if sy < edge_margin:
                    alpha_factor = max(0.0, sy / edge_margin)
                elif sy > h - edge_margin:
                    alpha_factor = max(0.0, (h - sy) / edge_margin)
                else:
                    alpha_factor = 1.0
                
                val = int(240 * alpha_factor)
                if val > 5:
                    draw.text((tx, sy), text, fill=(val, val, val), font=item["font"])
        
        proc.stdin.write(frame.tobytes())
    
    proc.stdin.close()
    proc.wait()
    print(f"Finished {out_filename} in {time.time()-t0:.1f}s!", flush=True)

# 1. 9x16 vertical video (1080x1920, 16s fast scroll)
render_video(1080, 1920, duration_sec=16, fps=30, out_filename="credits_9x16_fast.mp4", font_size=58, line_h=90)

# 2. 16x9 horizontal video (1920x1080, 16s fast scroll)
render_video(1920, 1080, duration_sec=16, fps=30, out_filename="credits_16x9_fast.mp4", font_size=54, line_h=84)

print("Both credit videos successfully rendered!")
