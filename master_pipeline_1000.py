import asyncio
import os
import re
import json
import urllib.request
import ssl
import cv2
import numpy as np
from skimage.transform import PiecewiseAffineTransform, warp
from scipy.spatial import Delaunay
from playwright.async_api import async_playwright

OUTPUT_DIR = "talking_head_photos_100"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("scratch", exist_ok=True)
CANDIDATES_FILE = "candidates_1000.json"

ctx_ssl = ssl._create_unverified_context()

WAVE_1 = [
    # High-density interview, speech & keynote tags
    "interview", "interviews", "podcastclip", "podcasts", "keynotespeaker", "publicspeaker",
    "tedtalk", "tedx", "leadership", "speaking", "creator", "storyteller", "filmmaker", "director",
    # TEDx, Keynote, Speech & Leadership
    "tedtalks", "tedxspeaker", "confspeaker", "guestspeaker", "commencementspeech", "valedictorian",
    "keynotepresentation", "speechexpert", "communicationcoach", "storytellingtips", "publicspeakingcoach",
    "speechtraining", "executivecoach", "leadershipskills", "speakerlife", "keynotes",
    # Creative Writing, Authors & Poets
    "creativewriting", "authorinterview", "authorlife", "writerslife", "writerscorner", "writersunite",
    "poetsofig", "spokenwordartist", "spokenwordpoetry", "instapoetry", "indieauthors", "booktoker",
    "bookreviewers", "fictionwriters", "yaauthors", "fantasywriters", "novelwriting",
    # Theatre, Film Directors & Masterclasses
    "actingmasterclass", "actingworkshop", "actingexercises", "theatredirector", "stageacting",
    "filmscoring", "cinematographers", "shortfilmmaking", "documentaryfilmmaker", "actorslife",
    "castingcall", "filmfestival", "theatrecompany", "stageplay", "broadwayshow",
    # History Buffs, Archaeology & Academia
    "historybuffs", "ancienthistoryfacts", "medievaltimes", "worldhistoryfacts", "archaeologicalfinds",
    "anthropologystudent", "museumcurator", "museumvisit", "historicalfacts", "philosophygeek",
    "academicresearch", "scholars", "professorslife", "historians", "archaeologylovers",
    # Science Communicators & Medical Residents
    "astronomylover", "astronomynerd", "physicsmajor", "quantumphysicsfacts", "biologyfacts",
    "microscopic", "sciencenerd", "sciencecommunicator", "stemwomen", "womeninstem", "medicalstudents",
    "futuredoctor", "doctorsofinstagram", "residency", "scrublife", "medstudentlife",
    # Comedy Clubs & International Standup
    "standupcomedyclub", "standupcomedyclip", "standupcomic", "comedyclubsf", "comedyclubnyc",
    "comedyclubla", "standupcomedyuk", "standupcomedyindia", "irishcomedian", "scotcomedy",
    "canadianstandup", "aussiestandup", "standupcomedyshowcase", "livecomedy", "comedyroast"
]

WAVE_2 = [
    # Stand-up Comedy, Crowd Work, Improv & Satire
    "standupcomedyshow", "standupcomediansofinstagram", "standupclip", "comedynight", "comedyvideo",
    "standuplive", "chicagocomedy", "bostoncomedy", "austincomedy", "seattlecomedy", "denvercomedy",
    "atlantacomedy", "dryhumor", "darkhumour", "cleanjokes", "dadjokes", "comedysketch", "comedyskit",
    "improvtheatre", "secondcity", "groundlings", "ucbtheatre", "comedycellar", "thecomedystore",
    "laughfactory", "carolinesonbroadway", "edinburghfringe", "fringecomedy", "melbournecomedyfestival",
    "montrealjustforlaughs", "jfl", "roastbattle", "standuproutine", "crowdworkclip", "openmicnight",
    "comedycentralstandup", "bbc_comedy", "standupuk", "aussiecomedian", "canadiancomedian"
]

WAVE_3 = [
    # University Professors, Historians & Philosophers
    "historylesson", "historybuff", "ancientcivilizations", "worldhistory", "medievalhistory",
    "militaryhistory", "artcurator", "museumguide", "librarians", "bookclub", "literaturelover",
    "classics", "philosophyquotes", "philosophical", "existentialism", "ethics", "epistemology",
    "logic", "criticalthinking", "politicalphilosophy", "classicalstudies", "renaissanceart",
    "byzantinehistory", "romanempire", "ancientegypt", "archaeologist", "paleontologist",
    "anthropologist", "sociologist", "universitylecturer", "academiclife", "highereducation",
    "phdstudent", "professoroftheday", "historyprofessor", "philosophyclass", "artcuration"
]

WAVE_4 = [
    # Scientists, Medical Experts, Neuroscientists & Psychologists
    "cellularbiology", "microbiology", "genetics", "biochemistry", "organicchemistry",
    "experimentalphysics", "theoreticalphysics", "climatology", "oceanography", "marinebiologist",
    "wildlifebiologist", "neuroscientist", "neurology", "clinicalpsychology", "cognitivepsychology",
    "childpsychology", "psychiatrist", "therapisttalk", "mentalhealthprofessional", "traumarecovery",
    "publichealtheducation", "epidemiologist", "immunologist", "dermatologist", "optometrist",
    "orthopedic", "cardiology", "physiotherapist", "nutritionscience", "registereddietitian",
    "sportsmedicine", "exercisephysiologist", "astrobiology", "cosmology", "quantummechanics"
]

WAVE_5 = [
    # Actors, Theatre Performers, Voiceover & Filmmakers
    "methodacting", "meisnertechnique", "stanislavski", "auditionadvice", "auditionroom",
    "castingdirectors", "hollywoodactor", "indiefilm", "shortfilm", "screenplay", "screenwriterlife",
    "filmmakinglife", "behindthescenesfilm", "cinematographylife", "theatreactor", "theatrelife",
    "musicaltheatreperformer", "voiceactingtips", "voiceoverwork", "voicecoach", "dialectcoach",
    "accentcoach", "stageplay", "offbroadway", "westendtheatre", "dramagraduate", "actingstudent",
    "auditiontape", "self-tape", "monologuepractice", "filmdirecting", "directorsguild"
]

WAVE_6 = [
    # Keynote Speakers, Authors, Artisans & Thought Leaders
    "debutauthor", "publishedauthor", "yaauthor", "fantasyauthor", "romanceauthor", "thrillerauthor",
    "poetrycommunity", "slamquotes", "tedxtalks", "keynotespeaking", "publicspeakingtips",
    "publicspeakingtraining", "executivepresence", "corporatecommunications", "communicationskills",
    "presentationtips", "leadershipcoach", "storytellingcoach", "speechwriter", "keynotespeakerlife",
    "masterclass", "sommelier", "wineeducator", "culinaryinstructor", "masterchocolatier",
    "artisanbread", "coffeecup", "woodworkingcommunity", "customwoodwork", "luthier", "bespoketailor"
]

ALL_WAVES = [WAVE_1, WAVE_2, WAVE_3, WAVE_4, WAVE_5, WAVE_6]

def load_models():
    detector = cv2.FaceDetectorYN.create("models/face_detection_yunet_2023mar.onnx", "", (320, 320), score_threshold=0.6)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("models/lbfmodel.yaml")
    return detector, facemark

def ear(pts):
    r_h1 = np.linalg.norm(pts[37] - pts[41])
    r_h2 = np.linalg.norm(pts[38] - pts[40])
    r_w  = np.linalg.norm(pts[36] - pts[39])
    r_ear = (r_h1 + r_h2) / (2.0 * r_w + 1e-5)
    
    l_h1 = np.linalg.norm(pts[43] - pts[47])
    l_h2 = np.linalg.norm(pts[44] - pts[46])
    l_w  = np.linalg.norm(pts[42] - pts[45])
    l_ear = (l_h1 + l_h2) / (2.0 * l_w + 1e-5)
    return (r_ear + l_ear) / 2.0

def eval_frame(fr, detector, facemark):
    h, w = fr.shape[:2]
    if h <= w: return False, 0
    
    # 1. Color check (reject archival / B&W)
    b, g, r = cv2.split(fr)
    if np.mean(np.abs(b - g)) < 5 and np.mean(np.abs(g - r)) < 5:
        return False, 0
        
    # Letterbox check
    top = fr[:int(h*0.06), :]
    bot = fr[int(h*0.94):, :]
    if (np.mean(top) < 15 and np.std(top) < 10) or (np.mean(bot) < 15 and np.std(bot) < 10):
        return False, 0
        
    detector.setInputSize((w, h))
    _, faces = detector.detect(fr)
    if faces is None or len(faces) != 1: return False, 0
    face = faces[0]
    conf = float(face[-1])
    if conf < 0.88: return False, 0
    fx, fy, fw, fh = [int(v) for v in face[:4]]
    area = (fw*fh)/(w*h)
    if area < 0.040 or area > 0.17: return False, 0
    if abs((fx+fw/2) - w/2) > w*0.17: return False, 0
    if fy < int(h*0.10) or fy > int(h*0.50): return False, 0
    
    re_x, re_y = int(face[4]), int(face[5])
    le_x, le_y = int(face[6]), int(face[7])
    
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    
    # 2. Sunglasses check
    r_patch = gray[max(0, re_y-15):min(h, re_y+15), max(0, re_x-20):min(w, re_x+20)]
    l_patch = gray[max(0, le_y-15):min(h, le_y+15), max(0, le_x-20):min(w, le_x+20)]
    if np.mean(r_patch) < 55 or np.mean(l_patch) < 55: return False, 0
    
    ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
    if not ok or len(lms) == 0: return False, 0
    pts = lms[0].reshape(68, 2)
    
    # 3. Direct camera gaze (reject downward laptop/book gaze)
    eye_ratio = ear(pts)
    if eye_ratio < 0.30: return False, 0
    
    # 4. Speech articulation
    mw = np.linalg.norm(pts[48] - pts[54])
    mh = np.linalg.norm(pts[51] - pts[57])
    m_ratio = mh / (mw + 1e-5)
    if m_ratio < 0.30 or m_ratio > 0.80: return False, 0
    
    # 5. Sclera / Cartoon check (reject 3D Pixar / CGI doll eyes)
    y1, y2 = int(min(pts[37,1], pts[38,1])), int(max(pts[40,1], pts[41,1]))
    x1, x2 = int(pts[36,0]), int(pts[39,0])
    if y2 > y1 and x2 > x1:
        re_box = gray[y1:y2, x1:x2]
        if re_box.size > 0:
            p95 = np.percentile(re_box, 95)
            re_h = y2 - y1
            if p95 > 195 and (re_h / max(1, fw)) > 0.08:
                return False, 0
    
    # 6. Chin / Lower mouth obstruction check (reject microphones, hands, pop filters)
    chin_y = int(pts[8, 1])
    chin_x = int(pts[8, 0])
    my = int(pts[62, 1])
    mx = int(pts[62, 0])
    if chin_y > my:
        sub_chin = gray[my:chin_y, max(0, mx-30):min(w, mx+30)]
        if sub_chin.size > 0:
            if np.mean(sub_chin) < 30 or np.std(sub_chin) < 12: return False, 0
            if np.max(np.abs(cv2.Sobel(sub_chin, cv2.CV_64F, 0, 1, ksize=3))) > 920: return False, 0
        
    # 7. Unobstructed cheeks and neck (no hands covering face)
    l_cheek = gray[fy:fy+fh, max(0, fx-40):fx]
    r_cheek = gray[fy:fy+fh, fx+fw:min(w, fx+fw+40)]
    if np.max(np.abs(cv2.Sobel(l_cheek, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
    if np.max(np.abs(cv2.Sobel(r_cheek, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
    
    neck_h = min(h - chin_y, 70)
    if neck_h > 20:
        neck = gray[chin_y:chin_y+neck_h, max(0, chin_x-40):min(w, chin_x+40)]
        if np.max(np.abs(cv2.Sobel(neck, cv2.CV_64F, 0, 1, ksize=3))) > 1050: return False, 0
        
    score = conf*100 + m_ratio*50 + eye_ratio*50
    return True, score

def align_to_canonical_canvas(img, landmarks, canvas_w=800, canvas_h=1000):
    re_center = np.mean(landmarks[36:42], axis=0)
    le_center = np.mean(landmarks[42:48], axis=0)

    target_re = np.array([canvas_w * 0.36, canvas_h * 0.40])
    target_le = np.array([canvas_w * 0.64, canvas_h * 0.40])

    src_pts = np.float32([re_center, le_center])
    dst_pts = np.float32([target_re, target_le])

    d_x = target_le[0] - target_re[0]
    d_y = target_le[1] - target_re[1]
    target_dist = np.sqrt(d_x**2 + d_y**2)

    s_x = le_center[0] - re_center[0]
    s_y = le_center[1] - re_center[1]
    src_dist = np.sqrt(s_x**2 + s_y**2)

    scale = target_dist / src_dist
    angle = np.degrees(np.arctan2(s_y, s_x))

    M = cv2.getRotationMatrix2D(tuple(re_center), angle, scale)
    M[0, 2] += target_re[0] - re_center[0]
    M[1, 2] += target_re[1] - re_center[1]

    aligned_img = cv2.warpAffine(img, M, (canvas_w, canvas_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    ones = np.ones((landmarks.shape[0], 1))
    pts_homo = np.hstack([landmarks, ones])
    aligned_landmarks = M.dot(pts_homo.T).T
    return aligned_img, aligned_landmarks

def generalized_procrustes_analysis(shapes_list, max_iter=15, tol=1e-5):
    shapes = np.array(shapes_list, dtype=np.float64)
    n_shapes, n_points, _ = shapes.shape

    centered = shapes - np.mean(shapes, axis=1, keepdims=True)
    scales = np.sqrt(np.sum(centered**2, axis=(1, 2), keepdims=True))
    normalized = centered / scales

    mean_shape = normalized[0].copy()
    for _ in range(max_iter):
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
    aimg, apts, target_full, boundary_anchors, CANVAS_W, CANVAS_H = args
    src_full = np.vstack([apts, boundary_anchors])
    rgb_float = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tform = PiecewiseAffineTransform.from_estimate(target_full, src_full)
    return warp(rgb_float, tform, output_shape=(CANVAS_H, CANVAS_W), mode='reflect').astype(np.float32)

def run_milestone_compositing(target_count):
    CANVAS_W, CANVAS_H = 800, 1000
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING PROCRUSTEAN COMPOSITING FOR MILESTONE {target_count}", flush=True)
    print(f"=======================================================", flush=True)
    
    yunet, facemark = load_models()
    all_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    target_files = all_files[:target_count]
    
    raw_images = []
    raw_landmarks = []
    valid_files = []
    
    for f in target_files:
        img = cv2.imread(os.path.join(OUTPUT_DIR, f))
        if img is None: continue
        h, w = img.shape[:2]
        yunet.setInputSize((w, h))
        _, faces = yunet.detect(img)
        if faces is None or len(faces) == 0: continue
        fx, fy, fw, fh = [int(v) for v in faces[0][:4]]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, lms = facemark.fit(gray, np.array([[fx, fy, fw, fh]], dtype=np.int32))
        if ok and len(lms) > 0:
            raw_images.append(img)
            raw_landmarks.append(lms[0].reshape(68, 2).astype(np.float64))
            valid_files.append(f)
            
    print(f"Extracted landmarks for all {len(raw_images)}/{target_count} images.", flush=True)
    
    aligned_images = []
    aligned_landmarks = []
    for img, pts in zip(raw_images, raw_landmarks):
        aimg, apts = align_to_canonical_canvas(img, pts, CANVAS_W, CANVAS_H)
        aligned_images.append(aimg)
        aligned_landmarks.append(apts)
        
    procrustes_mean = generalized_procrustes_analysis(aligned_landmarks)
    boundary_anchors = get_boundary_anchors(CANVAS_W, CANVAS_H)
    target_full = np.vstack([procrustes_mean, boundary_anchors])
    
    cum_acc = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    batch_acc = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.float32)
    start_batch = target_count - 100
    
    warp_tasks = [(aimg, apts, target_full, boundary_anchors, CANVAS_W, CANVAS_H) for aimg, apts in zip(aligned_images, aligned_landmarks)]
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for idx, warped_rgb in enumerate(executor.map(warp_worker, warp_tasks)):
            cum_acc += warped_rgb
            if idx >= start_batch:
                batch_acc += warped_rgb
        
    avg_rgb = cum_acc / len(aligned_images)
    avg_bgr = cv2.cvtColor((np.clip(avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    cum_path = f"composited_{target_count}_faces.jpg"
    cv2.imwrite(cum_path, avg_bgr)
    print(f"  ✓ Saved {cum_path}", flush=True)
    
    if start_batch >= 0 and len(aligned_images) >= target_count:
        batch_avg_rgb = batch_acc / 100.0
        batch_avg_bgr = cv2.cvtColor((np.clip(batch_avg_rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        batch_path = f"composited_batch_{start_batch+1}_to_{target_count}_only.jpg"
        cv2.imwrite(batch_path, batch_avg_bgr)
        print(f"  ✓ Saved {batch_path}", flush=True)
        
    # Grid generation
    cols = 25 if target_count >= 500 else 20
    rows = (target_count + cols - 1) // cols
    cw, ch = 80, 100
    grid = np.zeros((rows * ch, cols * cw, 3), dtype=np.uint8)
    for idx, cell_img in enumerate(aligned_images[:target_count]):
        r = idx // cols
        c = idx % cols
        grid[r*ch:(r+1)*ch, c*cw:(c+1)*cw] = cv2.resize(cell_img, (cw, ch))
    grid_path = f"procrustes_{target_count}_faces_grid.jpg"
    cv2.imwrite(grid_path, grid)
    print(f"  ✓ Saved {grid_path}", flush=True)
    
    # Save progressive catalog
    cat_path = f"talking_heads_{target_count}.md"
    with open(cat_path, "w") as out:
        out.write(f"# Viral Talking-Head Creators Catalog ({target_count} Faces)\n\n")
        out.write(f"Total Verified Viral Talking Heads: **{target_count}**\n\n")
        out.write("| Rank | Creator Handle | Reel Link | Screenshot |\n")
        out.write("| :--- | :--- | :--- | :--- |\n")
        for idx, f in enumerate(target_files):
            handle = f.split('_', 1)[1].replace('.jpg', '')
            out.write(f"| {idx+1:04d} | @{handle} | [{handle}](https://www.instagram.com/{handle}/) | [{f}](file:///Users/tarighian/most-viral-face/{OUTPUT_DIR}/{f}) |\n")
    print(f"  ✓ Generated {cat_path}", flush=True)

    # Progressive Master Evolution Banner
    milestones = [m for m in [15, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000] if m <= target_count]
    panel_w, panel_h = 240, 310
    banner_h = 390
    total_w = len(milestones) * panel_w + (len(milestones) + 1) * 15
    master = np.zeros((banner_h, total_w, 3), dtype=np.uint8)
    
    for idx, m in enumerate(milestones):
        m_path = f"composited_{m}_faces.jpg"
        if os.path.exists(m_path):
            m_img = cv2.imread(m_path)
            m_res = cv2.resize(m_img, (panel_w, panel_h))
            x = 15 + idx * (panel_w + 15)
            master[55:55+panel_h, x:x+panel_w] = m_res
            label = f"{m} Faces"
            cv2.putText(master, label, (x + panel_w//6, 40), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            
    evo_name = f"master_evolution_15_to_{target_count}.jpg"
    cv2.imwrite(evo_name, master)
    print(f"  ✓ Saved {evo_name}", flush=True)

    # Isolated Batches Comparison
    num_batches = target_count // 100
    batch_cards = []
    card_w, card_h = 320, 400
    for b_idx in range(num_batches):
        s_idx = b_idx * 100 + 1
        e_idx = (b_idx + 1) * 100
        bp = f"composited_batch_{s_idx}_to_{e_idx}_only.jpg" if b_idx > 0 else "composited_100_faces.jpg"
        if os.path.exists(bp):
            b_img = cv2.imread(bp)
            b_res = cv2.resize(b_img, (card_w, card_h))
            b_card = np.zeros((card_h + 55, card_w, 3), dtype=np.uint8)
            b_card[:card_h, :] = b_res
            cv2.putText(b_card, f"Batch {b_idx+1} ({s_idx}-{e_idx})", (card_w//2 - 110, card_h + 38), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
            batch_cards.append(b_card)
    
    if batch_cards:
        divider = np.ones((card_h + 55, 3, 3), dtype=np.uint8) * 50
        row_elems = []
        for i, c in enumerate(batch_cards):
            row_elems.append(c)
            if i < len(batch_cards) - 1:
                row_elems.append(divider)
        b_row = np.hstack(row_elems)
        title_b = np.zeros((65, b_row.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_b, f"ISOLATED 100-FACE SLICE COMPOSITES (BATCH 1 TO BATCH {num_batches})", (b_row.shape[1]//2 - 380, 44), cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 255), 2)
        b_banner = np.vstack([title_b, b_row])
        cv2.imwrite("master_isolated_batches_comparison.jpg", b_banner)
        print("  ✓ Saved master_isolated_batches_comparison.jpg", flush=True)

def process_video_stream(v_src, detector, facemark):
    try:
        req = urllib.request.Request(v_src, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=3.5, context=ctx_ssl) as resp:
            data = resp.read(2500000)
        if len(data) < 20000:
            return None, -1
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=True) as tmp:
            tmp.write(data)
            tmp.flush()
            cap = cv2.VideoCapture(tmp.name)
            best_f = None
            best_s = -1
            for i in range(120):
                ret, fr = cap.read()
                if not ret or fr is None: break
                if i < 8 or i % 3 != 0: continue
                ok, sc = eval_frame(fr, detector, facemark)
                if ok and sc > best_s:
                    best_s = sc
                    best_f = fr.copy()
                    if sc > 145: break
            cap.release()
            return best_f, best_s
    except Exception:
        return None, -1

async def worker(worker_id, queue, saved_counter, seen, lock):
    detector, facemark = load_models()
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        while not queue.empty():
            cand = await queue.get()
            vid = cand.get('video_id')
            author = cand.get('author', '').split('\n')[0].strip()
            clean = re.sub(r'[^a-zA-Z0-9._-]', '', author)[:25]
            
            async with lock:
                if saved_counter['count'] >= 1000:
                    break
                if clean and clean != 'creator' and clean[:6] in seen:
                    continue

            try:
                await page.goto(f'https://www.instagram.com/reel/{vid}/embed/', wait_until='domcontentloaded', timeout=3000)
                el = await page.wait_for_selector('video', timeout=1800)
                v_src = await el.get_attribute('src')
                if not v_src:
                    continue

                header_author = await page.evaluate(r'''() => {
                    const el = document.querySelector('.EmbedHeaderUsername, header a, a[href*=\"/p/\"], .UsernameText');
                    return el ? el.innerText.trim() : '';
                }''')
                if header_author:
                    clean = re.sub(r'[^a-zA-Z0-9._-]', '', header_author)[:25]
                    async with lock:
                        if clean[:6] in seen:
                            continue

                best_f, best_s = await asyncio.to_thread(process_video_stream, v_src, detector, facemark)

                if best_f is not None:
                    async with lock:
                        if not clean or clean == 'creator':
                            clean = f"creator_{vid[:8]}"
                        if saved_counter['count'] < 1000 and clean[:6] not in seen:
                            saved_counter['count'] += 1
                            count = saved_counter['count']
                            out_name = f"{count:04d}_{clean}.jpg"
                            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), best_f)
                            if not clean.startswith('creator_'):
                                seen.add(clean[:6])
                            print(f"  ✓ [{count:04d}/1000] SAVED (#{cand.get('tag')}): @{clean} -> {out_name} (score: {best_s:.1f})", flush=True)
                            
                            # Check milestone triggers
                            if count in [600, 700, 800, 900, 1000]:
                                await asyncio.to_thread(run_milestone_compositing, count)
            except Exception:
                pass
            finally:
                queue.task_done()
        await browser.close()

async def collect_batch(tags):
    print(f"\nScraping candidate reels across {len(tags)} explore tags...", flush=True)
    cands = []
    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(user_data_dir='.browser_session', headless=True)
        page = await ctx.new_page()
        for tag in tags:
            try:
                url = f"https://www.instagram.com/popular/{tag}/?utm_source=explore_tag"
                await page.goto(url, wait_until='domcontentloaded', timeout=3500)
                await asyncio.sleep(0.1)
                for _ in range(2):
                    await page.evaluate("window.scrollBy(0, 1500)")
                    await asyncio.sleep(0.15)
                found = await page.evaluate(r'''() => {
                    return Array.from(document.querySelectorAll('a[href*=\"/reel/\"]')).map(a => ({
                        href: a.href.split('?')[0],
                        author: (a.innerText || '').split('\\n')[0].trim()
                    }));
                }''')
                for r in found:
                    m = re.search(r'/reel/([a-zA-Z0-9_-]+)', r['href'])
                    if m:
                        r['video_id'] = m.group(1)
                        r['tag'] = tag
                        if not r.get('author'):
                            r['author'] = 'creator'
                        cands.append(r)
                print(f"  #{tag:24}: {len(found)} reels", flush=True)
            except Exception:
                pass
        await ctx.close()
    return cands

async def main():
    existing = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    current_count = len(existing)
    seen = {f.split('_', 1)[1].replace('.jpg', '')[:6] for f in existing}
    saved_counter = {'count': current_count}
    lock = asyncio.Lock()
    
    print(f"=== Starting Master Pipeline to 1,000 Faces: Count = {current_count}/1000 ===", flush=True)
    
    wave_idx = 0
    while saved_counter['count'] < 1000:
        current_tags = ALL_WAVES[wave_idx % len(ALL_WAVES)]
        wave_idx += 1
        
        candidates = await collect_batch(current_tags)
        queue = asyncio.Queue()
        for c in candidates:
            auth = c.get('author', '')
            if auth == 'creator' or (auth and auth[:6] not in seen):
                queue.put_nowait(c)
                
        print(f"\n--> Queued {queue.qsize()} fresh candidate reels across 8 parallel workers (Target: 1000, Current: {saved_counter['count']})...", flush=True)
        if queue.empty():
            await asyncio.sleep(1)
            continue
            
        workers = [asyncio.create_task(worker(i, queue, saved_counter, seen, lock)) for i in range(8)]
        await asyncio.gather(*workers)
        print(f"Wave finished. Current count: {saved_counter['count']}/1000", flush=True)
        
    final_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg') and not f.startswith(('composited', 'procrustes', 'milestone', 'master'))])
    print(f"\n=======================================================")
    print(f"MASTER PIPELINE FINISHED! Total creators: {len(final_files)}/1000")
    print(f"=======================================================", flush=True)

if __name__ == '__main__':
    asyncio.run(main())
