"""
编钟体感演奏器
用手掌在摄像头前"敲击"弧形排列的编钟，触发对应音调。
依赖: mediapipe, opencv-python, sounddevice, scipy, numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import math
import os
from PIL import Image, ImageDraw, ImageFont

# ─── 音频合成 ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100
pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)

def synth_bell(freq: float, duration: float = 1.2, volume: float = 0.6) -> np.ndarray:
    """合成编钟音色：基频 + 泛音叠加 + 指数衰减"""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # 基频 + 2/3/4 次谐波，模拟金属钟声
    wave = (
        1.0 * np.sin(2 * np.pi * freq * t) +
        0.5 * np.sin(2 * np.pi * freq * 2.76 * t) +  # 略失谐的三次泛音（钟声特征）
        0.25 * np.sin(2 * np.pi * freq * 5.4 * t) +
        0.1 * np.sin(2 * np.pi * freq * 8.93 * t)
    )
    # 快攻慢衰 envelope
    attack = int(SAMPLE_RATE * 0.005)
    env = np.exp(-3.5 * t / duration)
    env[:attack] = np.linspace(0, 1, attack)
    wave = wave * env * volume
    # 立体声
    stereo = np.stack([wave, wave], axis=1).astype(np.float32)
    return stereo

# ─── 编钟定义（C大调五声音阶 + 扩展，8个钟）────────────────────────────────

BELLS = [
    {"note": "C4", "cn_note": "宫", "freq": 261.63,  "scale": 1.2},
    {"note": "D4", "cn_note": "商", "freq": 293.66,  "scale": 1.15},
    {"note": "E4", "cn_note": "角", "freq": 329.63,  "scale": 1.1},
    {"note": "G4", "cn_note": "徵", "freq": 392.00,  "scale": 1.05},
    {"note": "A4", "cn_note": "羽", "freq": 440.00,  "scale": 1.0},
    {"note": "C5", "cn_note": "高宫", "freq": 523.25,  "scale": 0.95},
    {"note": "D5", "cn_note": "高商", "freq": 587.33,  "scale": 0.9},
    {"note": "E5", "cn_note": "高角", "freq": 659.25,  "scale": 0.85},
]

N_BELLS = len(BELLS)

# ─── 预先合成并缓存音频 ────────────────────────────────────────────────────
BELL_SOUNDS = {}
for i, bell in enumerate(BELLS):
    arr = synth_bell(bell["freq"], duration=1.2, volume=0.6)
    arr_int16 = np.int16(arr * 32767)
    # 转换为 PyGame 支持的 Sound 对象
    BELL_SOUNDS[i] = pygame.sndarray.make_sound(arr_int16)

def play_bell(b_idx: int):
    """如果当前声音已播放完毕，则播放声音"""
    # get_num_channels() 返回当前有几个声道在播放这个声音
    if BELL_SOUNDS[b_idx].get_num_channels() == 0:
        BELL_SOUNDS[b_idx].play()

# ─── 图像资源加载 ────────────────────────────────────────────────────────────
def load_image_safe(filename):
    """安全加载图片，支持中文路径，自动搜索常见位置"""
    search_paths = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", filename),
        os.path.join("d:/其他项目/MotionMusic/", filename)
    ]
    
    found_path = None
    for p in search_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        print(f"[WARN] Image {filename} not found!")
        return None
    
    try:
        img_data = np.fromfile(found_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"[ERROR] Loading {found_path}: {e}")
        return None

# ─── 编钟位置与绘制 ──────────────────────────────────────────────────────────

def compute_bell_positions(frame_w: int, frame_h: int):
    """计算每个编钟在画面中的中心位置（顶部横排）"""
    # 顶部留出空间 (为了避开 Logo，下移一些)
    top_y = int(frame_h * 0.32) 
    # 左右留出边距
    margin_x = int(frame_w * 0.1)
    available_w = frame_w - 2 * margin_x
    
    step_x = available_w / (N_BELLS - 1) if N_BELLS > 1 else 0
    
    positions = []
    for i in range(N_BELLS):
        x = int(margin_x + i * step_x)
        y = top_y
        positions.append((x, y))
    return positions

def overlay_image(background, overlay, x, y):
    """将带有 alpha 通道的 overlay 叠加到 background 上"""
    h, w = overlay.shape[:2]
    
    # 边界检查
    if x < 0: x = 0
    if y < 0: y = 0
    if x + w > background.shape[1]: w = background.shape[1] - x
    if y + h > background.shape[0]: h = background.shape[0] - y
    
    if w <= 0 or h <= 0: return

    overlay_crop = overlay[:h, :w]
    bg_crop = background[y:y+h, x:x+w]

    # 分离通道
    if overlay_crop.shape[2] == 4:
        alpha = overlay_crop[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha
        
        for c in range(3):
            bg_crop[:, :, c] = (alpha * overlay_crop[:, :, c] + alpha_inv * bg_crop[:, :, c])
    else:
        background[y:y+h, x:x+w] = overlay_crop

# ─── 碰撞状态追踪 ────────────────────────────────────────────────────────────

class HitTracker:
    """防止同一次触碰连续重复触发"""
    def __init__(self, cooldown: float = 0.35):
        self.last_hit = {}   # bell_idx -> timestamp
        self.cooldown = cooldown

    def try_hit(self, bell_idx: int) -> bool:
        now = time.time()
        last = self.last_hit.get(bell_idx, 0)
        if now - last > self.cooldown:
            self.last_hit[bell_idx] = now
            return True
        return False

    def recently_hit(self, bell_idx: int, window: float = 0.18) -> bool:
        return time.time() - self.last_hit.get(bell_idx, 0) < window

# ─── 绘制函数 ────────────────────────────────────────────────────────────────

def draw_bell(frame, pos, bell, img_bell, idx, hit: bool, glow: bool):
    x, y = pos
    scale = bell["scale"]
    
    if img_bell is None:
        # Fallback: draw circle
        cv2.circle(frame, (x, y), int(30 * scale), (100, 100, 200), -1)
        return

    # 根据音调调整大小
    base_h = 120
    target_h = int(base_h * scale)
    aspect_ratio = img_bell.shape[1] / img_bell.shape[0]
    target_w = int(target_h * aspect_ratio)
    
    resized_bell = cv2.resize(img_bell, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    # 如果被敲击或发光，增加亮度
    if hit or glow:
        # 简单增加亮度: 转换为 HSV 增加 V 分量，再转回 BGR (注意 img 是 BGRA)
        bgr = resized_bell[:, :, :3]
        alpha = resized_bell[:, :, 3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 50) # 增加亮度
        final_hsv = cv2.merge((h, s, v))
        final_bgr = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        resized_bell = cv2.merge((final_bgr, alpha))

    # 计算左上角坐标，使 (x,y) 为中心
    top_left_x = x - target_w // 2
    top_left_y = y - target_h // 2
    
    overlay_image(frame, resized_bell, top_left_x, top_left_y)

    # 保存碰撞区域大小供 main 使用 (简单做法: 存入 bell 字典)
    bell["w"] = target_w
    bell["h"] = target_h
    # 保存中心坐标供绘制文字
    bell["x"] = x
    bell["y"] = y

def draw_arch_beam(frame, positions):
    """画横梁 (这里改为直线)"""
    if not positions: return
    # 取第一个和最后一个点的上方
    start_pos = (positions[0][0], positions[0][1] - 80)
    end_pos = (positions[-1][0], positions[-1][1] - 80)
    cv2.line(frame, start_pos, end_pos, (160, 130, 80), 6)
    
    # 画每根挂绳
    for x, y in positions:
        cv2.line(frame, (x, y - 80), (x, y - 20), (180, 160, 130), 2)

def draw_hand_landmarks(frame, hand_landmarks, frame_w, frame_h, fingertip_indices):
    """画手部骨骼，高亮指尖"""
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    mp_drawing.draw_landmarks(
        frame, hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 200, 120), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(80, 200, 120), thickness=2)
    )
    # 高亮指尖
    for idx in fingertip_indices:
        lm = hand_landmarks.landmark[idx]
        cx = int(lm.x * frame_w)
        cy = int(lm.y * frame_h)
        cv2.circle(frame, (cx, cy), 10, (0, 255, 200), -1)
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)

def put_chinese_text(pil_draw, text, pos, color, font_size=20):
    """辅助函数：在 PIL ImageDraw 对象上绘制中文"""
    try:
        # 尝试加载 Windows 常见中文字体
        font = ImageFont.truetype("msyh.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("simhei.ttf", font_size)
        except:
            # 最后的 fallback，可能不支持中文
            font = ImageFont.load_default()
    
    pil_draw.text(pos, text, font=font, fill=color)

# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查设备连接")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    tracker = HitTracker(cooldown=0.35)
    
    # 加载编钟图片
    bell_img = load_image_safe("bell_transparent.png")
    if bell_img is None:
        print("[WARN] 使用默认圆形绘制")
        
    # 加载logo图片
    logo_img = load_image_safe("logo.png")
    if logo_img is not None:
        # 缩放logo，原图1280x462
        target_w = 280
        target_h = int(logo_img.shape[0] * (target_w / logo_img.shape[1]))
        logo_img = cv2.resize(logo_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        print("[WARN] Logo图片未找到")

    # 指尖关键点索引（拇指+四指）
    FINGERTIPS = [4, 8, 12, 16, 20]
    
    print("[INFO] 编钟体感演奏器启动！")
    print("   伸出手掌，用指尖敲击画面中的编钟")
    print("   按 Q 退出")

    previous_hit_bells = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像，更直观
        h, w = frame.shape[:2]

        # 计算钟的位置（横排）
        bell_positions = compute_bell_positions(w, h)

        # 暗色背景叠加，增加舞台感
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # 手势识别
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        hit_bells = set()

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_lm, w, h, FINGERTIPS)

                # 检测每根指尖
                for tip_idx in FINGERTIPS:
                    lm = hand_lm.landmark[tip_idx]
                    tip_x = int(lm.x * w)
                    tip_y = int(lm.y * h)

                    for b_idx, (bx, by) in enumerate(bell_positions):
                        # 获取编钟大小 (从字典中获取，如果未绘制过则给默认值)
                        bw = BELLS[b_idx].get("w", 50)
                        bh = BELLS[b_idx].get("h", 80)
                        
                        # 碰撞检测：矩形区域
                        in_rect_x = abs(tip_x - bx) < (bw / 2)
                        in_rect_y = abs(tip_y - by) < (bh / 2)
                        
                        if in_rect_x and in_rect_y:
                            hit_bells.add(b_idx)
                            # 只有这一帧是新触碰（不在上一帧中）才尝试播放
                            if b_idx not in previous_hit_bells:
                                if tracker.try_hit(b_idx):
                                    play_bell(b_idx)
        
        previous_hit_bells = hit_bells

        # ── 绘制编钟图像（OpenCV部分）──
        
        # 0. 绘制 Logo (左上角)
        if logo_img is not None:
            # Logo 已调整大小，直接绘制在 (20, 20)
            overlay_image(frame, logo_img, 20, 20)
            
        draw_arch_beam(frame, bell_positions)
        for i, (pos, bell) in enumerate(zip(bell_positions, BELLS)):
            is_hit = i in hit_bells
            is_glow = tracker.recently_hit(i, window=0.25)
            draw_bell(frame, pos, bell, bell_img, i, is_hit, is_glow)

        # ── 中文文字绘制（Pillow部分）──
        # 将 OpenCV 图像转换为 PIL 图像
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 1. 绘制右下角标题和提示
        # 智韵编钟 | 伸手体验
        title_text = "智韵编钟 | 伸手体验"
        info_text = "按 Q 退出"
        if results.multi_hand_landmarks:
            n = len(results.multi_hand_landmarks)
            hand_info = f"检测到 {n} 只手"
        else:
            hand_info = "请伸出双手演奏..."
            
        # 计算位置 (右对齐)
        # 假设字体大小 24 和 18
        # 需要获取图像宽高
        img_w, img_h = img_pil.size
        
        # 简单估算宽度
        title_w = len(title_text) * 24
        put_chinese_text(draw, title_text, (img_w - title_w - 40, img_h - 80), (200, 200, 200), font_size=24)
        
        info_full = f"{hand_info}  {info_text}"
        info_w = len(info_full) * 18
        put_chinese_text(draw, info_full, (img_w - info_w - 40, img_h - 45), (120, 220, 150), font_size=18)

        # 2. 绘制每个编钟的音名（全部在编钟下方）
        for bell in BELLS:
            if "x" in bell and "y" in bell:
                x, y = bell["x"], bell["y"]
                th = bell["h"] # 编钟高度
                
                # 中文音名（宫、商...）在编钟下方
                cn_text = bell["cn_note"]
                text_w = len(cn_text) * 20
                # 放在编钟底部下方 10 像素
                put_chinese_text(draw, cn_text, (x - text_w // 2, y + th // 2 + 10), (255, 255, 200), font_size=22)
                
                # 英文音名（C4...）在中文音名下方
                en_text = bell["note"]
                text_w_en = len(en_text) * 12
                # 放在中文音名下方 25 像素
                put_chinese_text(draw, en_text, (x - text_w_en // 2, y + th // 2 + 40), (200, 200, 200), font_size=16)

        # 将 PIL 图像转换回 OpenCV 图像
        frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Smart Bianzhong", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("[INFO] 再见！")

if __name__ == "__main__":
    main()