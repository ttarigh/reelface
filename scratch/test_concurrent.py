import asyncio
import cv2
import numpy as np
import time
from playwright.async_api import async_playwright

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

async def test_concurrent_rescue():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    
    test_reels = [
        ("creator___", "DTy5H8UjDhV"),
        ("sunny_.vloger", "DdRq03yNf9k"),
        ("lauren.welds", "DcOprJPuiaY"),
        ("startupseekho", "DX9kXhDp46C"),
    ]

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(
            user_data_dir='.browser_session',
            headless=True,
            args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
        )
        
        async def rescue_one(name, video_id):
            page = await ctx.new_page()
            url = f"https://www.instagram.com/reel/{video_id}/embed/"
            try:
                t0 = time.time()
                await page.goto(url, wait_until='networkidle', timeout=12000)
                video_src = await page.evaluate("() => document.querySelector('video')?.src || null")
                dt = time.time() - t0
                print(f"[{name}] video src fetched in {dt:.1f}s: {bool(video_src)}")
                return name, video_src
            except Exception as e:
                print(f"[{name}] error: {e}")
                return name, None
            finally:
                await page.close()

        tasks = [rescue_one(name, vid) for name, vid in test_reels]
        results = await asyncio.gather(*tasks)
        await ctx.close()

    print("Concurrent results:", results)

asyncio.run(test_concurrent_rescue())
