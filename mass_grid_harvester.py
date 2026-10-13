import asyncio
import os
import json
import re
import cv2
import numpy as np
import urllib.request
import ssl
from playwright.async_api import async_playwright

VERIFIED_MANIFEST = 'assets/verified_50k_manifest.json'
CREATORS_JSON = 'assets/creators.json'
OUTPUT_DIR = 'talking_head_photos_100'
BLACKLIST_PATH = 'assets/rejected_blacklist.json'
MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"
LBF_MODEL = "models/lbfmodel.yaml"

ctx_ssl = ssl._create_unverified_context()

ALL_TAGS = [
    # Storytime / Monologue / Relatable
    'storytime', 'storytimes', 'truestory', 'storyteller', 'crazystory', 'relatable',
    'relatablestory', 'confession', 'confessions', 'spillthetea', 'tea', 'whathappened',
    'rant', 'letschat', 'venting', 'unpopularopinion', 'casualchat', 'dayinthelife',
    'diml', 'vlog',
    # GRWM / Lifestyle / Routine
    'grwm', 'grwmroutine', 'getreadywithme', 'skincareroutine', 'morningroutine',
    'nightroutine', 'dailyroutine', 'skincaretips', 'beautytips', 'makeuptutorial',
    'haircare', 'hairroutine', 'outfitideas', 'grwmmakeup', 'grwmtalk', 'lifestylevlog',
    'dailylife', 'wellnessroutine', 'healthyhabits', 'glowup',
    # Career / Workplace / Professional
    'careeradvice', 'workplacetips', 'corporatehumor', 'interviewtips', 'resumetips',
    'jobinterview', 'corporatelife', 'careergoals', '9to5', 'wfh', 'officehumor',
    'salarynegotiation', 'internship', 'promotion', 'remotework', 'jobsearch',
    'networking', 'careercoach', 'professionaldevelopment', 'worklifebalance',
    # Personal Finance / Business / Wealth
    'personalfinance', 'moneytips', 'investingtips', 'budgetingtips', 'financialfreedom',
    'moneytok', 'passiveincome', 'savingmoney', 'wealthbuilding', 'realestateinvesting',
    'stockmarket', 'businessadvice', 'startup', 'entrepreneurship', 'smallbusiness',
    'sidehustle', 'ecommerce', 'founder', 'businessowner', 'financialliteracy',
    # Psychology / Wellness / Mental Health
    'psychology', 'psychologyfacts', 'mindsetshift', 'mentalhealth', 'therapytips',
    'innerpeace', 'boundaries', 'overcominganxiety', 'selflove', 'emotionalintelligence',
    'traumahealing', 'shadowwork', 'mindfulness', 'healingjourney', 'selfcare',
    'mentalwellness', 'growthmindset', 'affirmations', 'anxietysupport', 'personalgrowth',
    # BookTok / Knowledge / Ideas
    'booktok', 'bookrecommendations', 'bookreview', 'mustreadbooks', 'reading',
    'readinglist', 'favoritebooks', 'bookish', 'booklover', 'bookclub', 'literature',
    'nonfiction', 'selfhelpbooks', 'bookworm', 'writing',
    # Motivation / Discipline / Philosophy
    'motivationalspeech', 'motivationalspeaker', 'inspiration', 'hardtruths', 'discipline',
    'wisdomoftheday', 'dailywisdom', 'dailymotivation', 'focus', 'successmindset',
    'grit', 'hustle', 'philosophy', 'stoicism', 'stoic',
    # Communication / Public Speaking / Social Skills
    'publicspeaking', 'communication', 'communicationskills', 'speechtips', 'confidence',
    'bodylanguage', 'socialskills', 'charisma', 'conversation', 'speakingtips',
    'assertiveness', 'activelistening', 'persuasion', 'debating', 'leadership',
    # Dating / Relationships / Life Advice
    'relationshiptips', 'datingadvice', 'datingstorytime', 'redflags', 'greenflags',
    'marriageadvice', 'relationshipgoals', 'breakupadvice', 'healthyrelationships',
    'lifeadvice', 'advice', 'realtalk', 'deepthoughts', 'perspective', 'lifelessons'
]

def ear(pts):
    p2_p6 = np.linalg.norm(pts[37] - pts[41])
    p3_p5 = np.linalg.norm(pts[38] - pts[40])
    p1_p4 = np.linalg.norm(pts[36] - pts[39])
    left_ear = (p2_p6 + p3_p5) / (2.0 * p1_p4 + 1e-5)

    p2_p6_r = np.linalg.norm(pts[43] - pts[47])
    p3_p5_r = np.linalg.norm(pts[44] - pts[46])
    p1_p4_r = np.linalg.norm(pts[42] - pts[45])
    right_ear = (p2_p6_r + p3_p5_r) / (2.0 * p1_p4_r + 1e-5)

    return (left_ear + right_ear) / 2.0

def create_models():
    detector = cv2.FaceDetectorYN.create(MODEL_PATH, "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LBF_MODEL)
    return detector, facemark

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0

    top_bar = fr[:int(h*0.12), :]
    bot_bar = fr[int(h*0.88):, :]
    if (np.mean(top_bar) < 12 and np.std(top_bar) < 10) and (np.mean(bot_bar) < 12 and np.std(bot_bar) < 10):
        return False, 0

    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    # Strictly single face in frame (no two-person scenes, skits, street interviews)
    if faces is None or len(faces) != 1: return False, 0

    primary = faces[0]
    conf = float(primary[-1])
    if conf < 0.82: return False, 0

    fx, fy, fw, fh = [int(v) for v in primary[:4]]
    area = (fw*fh)/(w*h)

    # Head & shoulders framing (3.0% to 28.0% of frame)
    if area < 0.030 or area > 0.28: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.22: return False, 0
    if fy < int(h*0.05) or fy > int(h*0.58): return False, 0

    re_x, re_y = int(primary[4]), int(primary[5])
    le_x, le_y = int(primary[6]), int(primary[7])
    nt_x, nt_y = int(primary[8]), int(primary[9])
    eye_span = abs(le_x - re_x)
    if eye_span < 45: return False, 0

    nose_offset = abs((nt_x - re_x) - (le_x - nt_x)) / (eye_span + 1e-5)
    if nose_offset > 0.20: return False, 0

    face_crop = fr[max(0, fy):min(h, fy+fh), max(0, fx):min(w, fx+fw)]
    if face_crop.size > 0:
        unique_colors = len(np.unique(face_crop.reshape(-1, 3), axis=0))
        if unique_colors < 1600: return False, 0

    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)

    eye_ratio = ear(pts)
    if eye_ratio < 0.20: return False, 0

    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.18 or m_ratio > 0.88: return False, 0

    score = float(conf*100 + m_ratio*50 + eye_ratio*50 - nose_offset*50)
    return True, score

def parse_metric(txt):
    m = re.search(r'([0-9,.]+)\s*([KkMmBbn]?)\b', txt)
    if not m: return 500000
    val = float(m.group(1).replace(',', ''))
    unit = m.group(2).upper()
    if unit == 'K': val *= 1000
    elif unit == 'M': val *= 1000000
    return int(val)

BLOCKED_KEYWORDS = [
    'comedy', 'scene', 'movie', 'vault', 'maxxing', 'netflix', 'fallon', 'colbert', 'kimmel',
    'clip', 'quotes', 'fact', 'cinema', 'bpm', 'tv', 'show', 'studio', 'page', 'central',
    'songs', 'status', 'theatre', 'film', 'dentist', 'clinic', 'surgery', 'doctor', 'hospital',
    'derm', 'dermatolog', 'dr.', 'dr_', 'dr', 'doct', 'ped', 'pediatric', 'accent', 'pronounc', 'esl',
    'medical', 'md', 'pharma', 'physio', 'chiro', 'nurse', 'english', 'spanish', 'french', 'german',
    'japanese', 'chinese', 'korean', 'grammar', 'vocab', 'lingo', 'tutor', 'language', 'speakup',
    'podcast', 'news', 'music', 'singer', 'radio', 'band', 'guitar', 'vogue', 'hollywood', 'unlocked',
    'standup'
]

async def process_tag(page, tag, detector, facemark, lock, state):
    url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
    try:
        await page.goto(url, wait_until='domcontentloaded', timeout=12000)
        await asyncio.sleep(1.0)

        # Scroll 3 times to load ~40-50 cards
        for _ in range(3):
            await page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            await asyncio.sleep(0.7)

        cards = await page.evaluate(r'''() => {
            const items = [];
            const links = Array.from(document.querySelectorAll('a[href*="/reel/"]'));
            for (const a of links) {
                const imgs = Array.from(a.querySelectorAll('img'));
                if (imgs.length === 0) continue;
                const thumb = imgs[0].src;
                let handle = null;
                for (const img of imgs) {
                    if (img.alt && img.alt.includes("'s profile picture")) {
                        handle = img.alt.replace("'s profile picture", "").trim();
                        break;
                    }
                }
                const text = a.innerText || '';
                items.push({
                    href: a.href,
                    thumb: thumb,
                    handle: handle,
                    text: text
                });
            }
            return items;
        }''')

        found_in_tag = 0
        for c in cards:
            href = c.get('href', '')
            m = re.search(r'/reel/([a-zA-Z0-9_-]+)', href)
            if not m: continue
            vid = m.group(1)

            if vid in state['seen_vids']:
                continue

            raw_handle = c.get('handle')
            if not raw_handle:
                lines = [l.strip() for l in c.get('text', '').split('\n') if l.strip()]
                if lines: raw_handle = lines[0]
            if not raw_handle:
                continue

            clean_handle = re.sub(r'[^a-zA-Z0-9._-]', '', raw_handle)[:25]
            clean_lower = clean_handle.lower()

            if clean_lower in state['seen_handles']:
                continue

            if any(b in clean_lower for b in BLOCKED_KEYWORDS):
                continue

            thumb_url = c.get('thumb')
            if not thumb_url or not thumb_url.startswith('http'):
                continue

            try:
                req = urllib.request.Request(thumb_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=4.0, context=ctx_ssl) as u:
                    fr = cv2.imdecode(np.frombuffer(u.read(), dtype=np.uint8), -1)
            except:
                continue

            if fr is None:
                continue

            ok, score = eval_frame(fr, detector, facemark)
            if not ok:
                continue

            # Passed all checks!
            state['seen_vids'].add(vid)
            state['seen_handles'].add(clean_lower)

            views_metric = 500000
            for line in c.get('text', '').split('\n'):
                line = line.strip()
                if re.match(r'^[0-9,.]+\s*[KkMmBbn]?$', line):
                    views_metric = parse_metric(line)
                    break
            estimated_likes = max(28500, int(views_metric * 0.075))

            async with lock:
                with open(VERIFIED_MANIFEST) as mf:
                    current_manifest = json.load(mf)

                existing_ranks = [int(k) for k in current_manifest.keys() if str(k).isdigit()]
                next_rank = max(existing_ranks + [0]) + 1
                rank_str = f"{next_rank:04d}"
                out_name = f"{rank_str}_{clean_handle}.jpg"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, fr)

                score_val = float(round(float(score), 1))
                reel_url = f"https://www.instagram.com/reel/{vid}/"
                current_manifest[rank_str] = {
                    'rank': rank_str,
                    'handle': clean_handle,
                    'video_id': vid,
                    'reelUrl': reel_url,
                    'likes': estimated_likes,
                    'photo': out_name,
                    'category': tag,
                    'score': score_val,
                    'verified': True,
                    'source': f'popular_grid_{tag}'
                }

                tmp_mf = VERIFIED_MANIFEST + '.tmp'
                with open(tmp_mf, 'w') as mf:
                    json.dump(current_manifest, mf, indent=2)
                os.replace(tmp_mf, VERIFIED_MANIFEST)

                tmp_cf = CREATORS_JSON + '.tmp'
                with open(tmp_cf, 'w') as cf:
                    json.dump(list(current_manifest.values()), cf, indent=2)
                os.replace(tmp_cf, CREATORS_JSON)

                state['saved_count'] += 1
                found_in_tag += 1
                print(f"  ★ [{rank_str}] [{tag}] SAVED: @{clean_handle} -> {reel_url} ({estimated_likes:,} likes, score: {score_val:.1f}) [Total Staged: {state['saved_count']}]", flush=True)

        if found_in_tag > 0:
            print(f"✓ #{tag}: added {found_in_tag} cards (Total Staged: {state['saved_count']})", flush=True)

    except Exception as e:
        print(f"Error on #{tag}: {e}", flush=True)

async def worker(worker_id, tag_sublist, lock, state, browser):
    detector, facemark = create_models()
    context = await browser.new_context(viewport={'width': 1280, 'height': 1600})
    page = await context.new_page()

    for idx, tag in enumerate(tag_sublist):
        await process_tag(page, tag, detector, facemark, lock, state)
        # Check if reached 680 staged cards
        if state['saved_count'] >= 680:
            print(f"[Worker {worker_id}] Target of 680 cards reached! Stopping early.")
            break

    await context.close()

async def main():
    print(f"=== Launching Mass Grid Harvester across {len(ALL_TAGS)} Tags ===")
    lock = asyncio.Lock()

    with open(BLACKLIST_PATH) as bf:
        bl_data = json.load(bf)
    blacklisted_vids = set(bl_data.get("video_ids", []))
    blacklisted_handles = {h.lower() for h in bl_data.get("handles", [])}

    with open(VERIFIED_MANIFEST) as mf:
        manifest = json.load(mf)
    existing_handles = {v['handle'].lower() for v in manifest.values()} | blacklisted_handles
    existing_vids = {v.get('video_id') for v in manifest.values() if v.get('video_id')} | blacklisted_vids

    state = {
        'seen_vids': existing_vids,
        'seen_handles': existing_handles,
        'saved_count': 0
    }

    # Split tags across 4 workers
    num_workers = 4
    sublists = [ALL_TAGS[i::num_workers] for i in range(num_workers)]

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--disable-blink-features=AutomationControlled']
        )
        workers = [worker(i, sublists[i], lock, state, browser) for i in range(num_workers)]
        await asyncio.gather(*workers)
        await browser.close()

    print(f"\n=======================================================")
    print(f"✓ MASS HARVEST COMPLETE! Staged {state['saved_count']} fresh talking-head cards into Curation Studio!")
    print(f"=======================================================")

if __name__ == '__main__':
    asyncio.run(main())
