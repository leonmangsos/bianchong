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

import sys

# ─── 图像资源加载 ────────────────────────────────────────────────────────────
def load_image_safe(filename):
    """安全加载图片，支持中文路径，自动搜索常见位置，包括 PyInstaller 打包后的路径"""
    search_paths = [
        filename,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../", filename),
        os.path.join("d:/其他项目/MotionMusic/", filename)
    ]
    
    # 支持 PyInstaller 运行时解压路径
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        search_paths.insert(0, os.path.join(base_path, filename))
    
    found_path = None
    for p in search_paths:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        # print(f"[WARN] Image {filename} not found!") # 可选屏蔽警告
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
    if overlay is None: return
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

def put_chinese_text(pil_draw, text, pos, color, font_size=20, font_path=None):
    """辅助函数：在 PIL ImageDraw 对象上绘制中文"""
    font = None
    if font_path:
        try:
             font = ImageFont.truetype(font_path, font_size)
        except:
             pass
    
    if font is None:
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

# ─── 场景管理与交互 ──────────────────────────────────────────────────────────

SCENE_MENU = 0
SCENE_GAME = 1
SCENE_LEARN = 2

class AppState:
    def __init__(self):
        self.current_scene = SCENE_MENU
        self.width = 1280
        self.height = 720
        self.running = True
        
        # 资源
        self.bg_main = load_image_safe("bg_main.jpg")
        self.icon_option = load_image_safe("icon_option_circle.png")
        self.bell_img = load_image_safe("bell_transparent.png")
        self.logo_img = load_image_safe("logo.png")
        
        # 处理 Logo
        if self.logo_img is not None:
            target_w = 280
            target_h = int(self.logo_img.shape[0] * (target_w / self.logo_img.shape[1]))
            self.logo_img = cv2.resize(self.logo_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # 菜单状态
        self.menu_options = ["听音弹曲", "自由练习"]
        self.menu_index = 1  # 默认选中自由练习
        
        # 鼠标状态
        self.mouse_pos = (0, 0)
        self.mouse_clicked = False

def on_mouse(event, x, y, flags, param):
    app = param
    app.mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        app.mouse_clicked = True
    elif event == cv2.EVENT_LBUTTONUP:
        app.mouse_clicked = False

def run_menu_scene(app):
    cv2.setMouseCallback("Smart Bianzhong", on_mouse, app)
    
    # 准备背景
    bg_frame = np.zeros((app.height, app.width, 3), dtype=np.uint8)
    if app.bg_main is not None:
        # 缩放背景填满屏幕而不压缩比例的话，可以考虑裁剪，但用户提到不要压缩，我们可以直接保持比例缩放并裁剪，或者简单直接Resize(因为之前是直接Resize)
        # “现在封面的图片不要压缩” -> 改为等比例缩放并居中裁剪
        h, w = app.bg_main.shape[:2]
        scale = max(app.width / w, app.height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        bg_scaled = cv2.resize(app.bg_main, (new_w, new_h))
        # 居中裁剪
        start_x = (new_w - app.width) // 2
        start_y = (new_h - app.height) // 2
        bg_cropped = bg_scaled[start_y:start_y+app.height, start_x:start_x+app.width]
        
        # 底部模糊处理打底
        # 用户反馈图片模糊不完整，移除模糊处理，直接显示裁剪后的背景
        # blur_start_y = int(app.height * 0.45)
        # bottom_part = bg_cropped[blur_start_y:, :]
        # blurred_bottom = cv2.GaussianBlur(bottom_part, (51, 51), 0)
        # 叠加一点黑色，变暗以突出文字
        # black_overlay = np.zeros_like(blurred_bottom)
        # blurred_bottom = cv2.addWeighted(blurred_bottom, 0.6, black_overlay, 0.4, 0)
        # bg_cropped[blur_start_y:, :] = blurred_bottom
        
        bg_frame = bg_cropped
    else:
        # 默认淡黄色古风背景
        bg_frame[:] = (230, 240, 250) 
        
    while app.running and app.current_scene == SCENE_MENU:
        frame = bg_frame.copy()
        
        # 转 PIL 绘制文字和 UI
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 绘制标题 (可以用 Logo 代替，或者再画一次 Logo)
        # 如果有 Logo，画在中间偏上
        if app.logo_img is not None:
            # 转回 cv2 绘制 Logo，因为 overlay_image 是 cv2 操作
            # 这里为了性能，先转回 cv2，画完 Logo 再转 PIL 画字，或者直接全部 PIL
            # 为简单起见，我们分层：背景 -> Logo -> PIL文字
            pass
        
        # 我们先处理 PIL 文字，最后转回 cv2 再叠 Logo 和 图标
        
        # 绘制菜单选项
        center_x = app.width // 2
        start_y = int(app.height * 0.5)
        gap_y = 100
        
        option_rects = [] # (x, y, w, h, index)
        
        for i, text in enumerate(app.menu_options):
            # 选中状态
            is_selected = (i == app.menu_index)
            
            # 文字大小
            font_size = 40 if is_selected else 32
            color = (100, 50, 20) if is_selected else (150, 100, 80)
            if is_selected:
                 color = (255, 100, 50) # 选中亮橙色
            
            # 简单估算文字宽高
            text_w = len(text) * font_size
            text_h = font_size + 20
            
            x = center_x - text_w // 2 + 30 # 稍微右偏给图标留位
            y = start_y + i * gap_y
            
            # 记录区域用于鼠标检测
            # 整个按钮区域（包括图标）
            btn_w = text_w + 100
            btn_h = 80
            btn_x = center_x - btn_w // 2
            btn_y = y - btn_h // 2
            option_rects.append((btn_x, btn_y, btn_w, btn_h, i))
            
            # 绘制文字
            # put_chinese_text(draw, text, (x, y - font_size//2), color, font_size=font_size)
        
        # PIL 绘制结束，转回 CV2
        frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        
        # 绘制 Logo (上方居中)
        if app.logo_img is not None:
            lx = (app.width - app.logo_img.shape[1]) // 2
            ly = 80
            overlay_image(frame, app.logo_img, lx, ly)
            
        # 绘制选项 UI (图标 + 文字)
        # 重新再用 PIL 画文字可能比较清晰，但这里为了层级关系，我们在 CV2 之后画文字也行
        # 或者直接在 CV2 上画图标，文字刚才已经画在 PIL 层了？不对，刚才只是计算了位置，没画
        
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        mx, my = app.mouse_pos
        
        for rx, ry, rw, rh, idx in option_rects:
            # 鼠标悬停检测
            if rx <= mx <= rx+rw and ry <= my <= ry+rh:
                app.menu_index = idx
                if app.mouse_clicked:
                    # 点击确认
                    if idx == 0:
                        print("选择：听音弹曲")
                        app.current_scene = SCENE_LEARN
                    elif idx == 1:
                        print("选择：自由练习")
                        app.current_scene = SCENE_GAME
                    app.mouse_clicked = False # 重置
            
            is_selected = (idx == app.menu_index)
            
            # 绘制选项框背景 (可选，半透明圆角矩形很难画，先略过)
            # if is_selected:
            #     draw.rectangle([rx, ry, rx+rw, ry+rh], fill=(255, 255, 255, 100), outline=None)
            
            # 绘制图标
            icon_size = 64 if is_selected else 48
            icon_x = rx + 10
            icon_y = ry + (rh - icon_size) // 2
            
            # 绘制文字
            font_size = 36 if is_selected else 30
            text = app.menu_options[idx]
            text_x = icon_x + icon_size + 20
            text_y = ry + (rh - font_size) // 2 - 5
            
            color = (255, 250, 240) if is_selected else (220, 220, 220)
            # 描边效果
            stroke_color = (139, 69, 19) # 棕色描边
            
            # 绘制文字（模拟描边）
            put_chinese_text(draw, text, (text_x+2, text_y+2), stroke_color, font_size=font_size)
            put_chinese_text(draw, text, (text_x, text_y), color, font_size=font_size)

        frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        
        # 叠加图标 (CV2)
        for rx, ry, rw, rh, idx in option_rects:
            is_selected = (idx == app.menu_index)
            icon_size = 64 if is_selected else 48
            icon_x = rx + 10
            icon_y = ry + (rh - icon_size) // 2
            
            if app.icon_option is not None:
                icon_resized = cv2.resize(app.icon_option, (icon_size, icon_size), interpolation=cv2.INTER_AREA)
                overlay_image(frame, icon_resized, icon_x, icon_y)

        cv2.imshow("Smart Bianzhong", frame)
        
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') or key == 27:
            app.running = False
            break
        elif key == 13: # Enter
            if app.menu_index == 0: # 听音弹曲
                app.current_scene = SCENE_LEARN
            elif app.menu_index == 1: # 自由练习
                app.current_scene = SCENE_GAME
        elif key == ord('w') or key == 0x26: # Up
            app.menu_index = (app.menu_index - 1) % len(app.menu_options)
        elif key == ord('s') or key == 0x28: # Down
            app.menu_index = (app.menu_index + 1) % len(app.menu_options)

def run_game_scene(app):
    # 清除鼠标回调
    cv2.setMouseCallback("Smart Bianzhong", lambda *args: None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 无法打开摄像头，请检查设备连接")
        app.current_scene = SCENE_MENU # 返回菜单
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, app.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app.height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    )

    tracker = HitTracker(cooldown=0.35)
    FINGERTIPS = [4, 8, 12, 16, 20]
    previous_hit_bells = set()
    
    # 游戏主循环
    while app.running and app.current_scene == SCENE_GAME:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像
        h, w = frame.shape[:2]

        bell_positions = compute_bell_positions(w, h)

        # 舞台背景
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

                for tip_idx in FINGERTIPS:
                    lm = hand_lm.landmark[tip_idx]
                    tip_x = int(lm.x * w)
                    tip_y = int(lm.y * h)

                    for b_idx, (bx, by) in enumerate(bell_positions):
                        bw = BELLS[b_idx].get("w", 50)
                        bh = BELLS[b_idx].get("h", 80)
                        
                        if abs(tip_x - bx) < (bw / 2) and abs(tip_y - by) < (bh / 2):
                            hit_bells.add(b_idx)
                            if b_idx not in previous_hit_bells:
                                if tracker.try_hit(b_idx):
                                    play_bell(b_idx)
        
        previous_hit_bells = hit_bells

        # 绘制 UI
        if app.logo_img is not None:
            overlay_image(frame, app.logo_img, 20, 20)
            
        draw_arch_beam(frame, bell_positions)
        for i, (pos, bell) in enumerate(zip(bell_positions, BELLS)):
            is_hit = i in hit_bells
            is_glow = tracker.recently_hit(i, window=0.25)
            draw_bell(frame, pos, bell, app.bell_img, i, is_hit, is_glow)

        # 文字绘制
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        title_text = "智韵编钟 | 伸手体验"
        info_text = "按 Q 返回菜单" # 修改提示
        
        if results.multi_hand_landmarks:
            n = len(results.multi_hand_landmarks)
            hand_info = f"检测到 {n} 只手"
        else:
            hand_info = "请伸出双手演奏..."
            
        img_w, img_h = img_pil.size
        title_w = len(title_text) * 24
        put_chinese_text(draw, title_text, (img_w - title_w - 40, img_h - 80), (200, 200, 200), font_size=24)
        
        info_full = f"{hand_info}  {info_text}"
        info_w = len(info_full) * 18
        put_chinese_text(draw, info_full, (img_w - info_w - 40, img_h - 45), (120, 220, 150), font_size=18)

        for bell in BELLS:
            if "x" in bell and "y" in bell:
                x, y = bell["x"], bell["y"]
                th = bell["h"]
                cn_text = bell["cn_note"]
                text_w = len(cn_text) * 20
                put_chinese_text(draw, cn_text, (x - text_w // 2, y + th // 2 + 10), (255, 255, 200), font_size=22)
                en_text = bell["note"]
                text_w_en = len(en_text) * 12
                put_chinese_text(draw, en_text, (x - text_w_en // 2, y + th // 2 + 40), (200, 200, 200), font_size=16)

        frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)
        cv2.imshow("Smart Bianzhong", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            app.current_scene = SCENE_MENU # 返回菜单而不是直接退出
            break

    cap.release()
    hands.close()

# ─── 主入口 ──────────────────────────────────────────────────────────────────

def main():
    app = AppState()
    print("[INFO] 编钟体感演奏器启动！")
    
    cv2.namedWindow("Smart Bianzhong", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Smart Bianzhong", app.width, app.height)
    
    while app.running:
        if app.current_scene == SCENE_MENU:
            run_menu_scene(app)
        elif app.current_scene == SCENE_GAME:
            run_game_scene(app)
        elif app.current_scene == SCENE_LEARN:
            run_learn_scene(app)

    cv2.destroyAllWindows()
    print("[INFO] 再见！")

def run_learn_scene(app):
    """听音弹曲模式：
    1. 示例：从右到左弹一遍
    2. 挑战：随机播放一段曲子
    3. 玩家尝试并验证
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, app.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app.height)
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
    tracker = HitTracker(cooldown=0.35)
    FINGERTIPS = [4, 8, 12, 16, 20]
    
    # ── 1. 示例演示：从右到左 ──
    demo_sequence = list(range(N_BELLS-1, -1, -1))
    
    # 状态变量用于视觉反馈
    feedback_timer = 0
    feedback_msg = ""
    feedback_color = (255, 255, 255)
    
    def draw_base_scene(frame_in, highlight_idx=-1, status_text="", show_hand=True):
        frame_out = frame_in.copy()
        h, w = frame_out.shape[:2]
        bell_positions = compute_bell_positions(w, h)
        
        # 舞台背景
        overlay = frame_out.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, frame_out, 0.65, 0, frame_out)
        
        if app.logo_img is not None:
            overlay_image(frame_out, app.logo_img, 20, 20)
            
        draw_arch_beam(frame_out, bell_positions)
        for i, (pos, bell) in enumerate(zip(bell_positions, BELLS)):
            # 过关时全亮，或者演示时高亮，或者最近被敲击
            is_glow = (status_text == "恭喜过关！") or (i == highlight_idx) or tracker.recently_hit(i, 0.25)
            draw_bell(frame_out, pos, bell, app.bell_img, i, False, is_glow)
            
        # 文字绘制
        img_pil = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        put_chinese_text(draw, "模式：听音弹曲", (app.width - 200, 30), (200, 200, 200), font_size=20)
        
        # 居中显示大标题/状态
        status_font_size = 40
        status_w = len(status_text) * status_font_size
        put_chinese_text(draw, status_text, (app.width//2 - status_w // 2, app.height - 120), (255, 255, 200), font_size=status_font_size)
        
        # 弹错反馈
        nonlocal feedback_timer, feedback_msg, feedback_color
        if time.time() < feedback_timer:
            msg_w = len(feedback_msg) * 50
            put_chinese_text(draw, feedback_msg, (app.width//2 - msg_w//2, app.height//2), feedback_color, font_size=50)

        for bell in BELLS:
            if "x" in bell and "y" in bell:
                x, y = bell["x"], bell["y"]
                th = bell["h"]
                cn_text = bell["cn_note"]
                text_w = len(cn_text) * 20
                put_chinese_text(draw, cn_text, (x - text_w // 2, y + th // 2 + 10), (255, 255, 200), font_size=22)
                en_text = bell["note"]
                text_w_en = len(en_text) * 12
                put_chinese_text(draw, en_text, (x - text_w_en // 2, y + th // 2 + 40), (200, 200, 200), font_size=16)
        
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

    # 演示循环
    for idx in demo_sequence:
        start_t = time.time()
        play_bell(idx)
        tracker.try_hit(idx)
        while time.time() - start_t < 0.5:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = draw_base_scene(frame, highlight_idx=idx, status_text="系统演示中...")
            cv2.imshow("Smart Bianzhong", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): return

    # ── 2. 随机生成挑战 ──
    import random
    challenge_len = 4
    challenge_sequence = [random.randint(0, N_BELLS-1) for _ in range(challenge_len)]
    
    time.sleep(0.5)
    for idx in challenge_sequence:
        start_t = time.time()
        play_bell(idx)
        tracker.try_hit(idx)
        while time.time() - start_t < 0.7:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = draw_base_scene(frame, highlight_idx=idx, status_text="请记住这段旋律！")
            cv2.imshow("Smart Bianzhong", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): return
            
    # ── 准备开始倒计时 ──
    for count in ["3", "2", "1", "开始！"]:
        start_t = time.time()
        while time.time() - start_t < 0.8:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            display_frame = draw_base_scene(frame, status_text=f"准备：{count}")
            cv2.imshow("Smart Bianzhong", display_frame)
            cv2.waitKey(1)

    # ── 3. 玩家尝试 ──
    user_sequence = []
    is_success = False
    previous_hit_bells = set()
    
    # 重新播放按钮区域 (左下方)
    replay_btn_rect = (20, app.height - 100, 160, 80) # x, y, w, h
    
    # 注册鼠标事件
    mouse_click_pos = None
    def on_click(event, x, y, flags, param):
        nonlocal mouse_click_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_click_pos = (x, y)
    cv2.setMouseCallback("Smart Bianzhong", on_click)
    
    while app.running and app.current_scene == SCENE_LEARN:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        bell_positions = compute_bell_positions(w, h)
        
        # 处理重新播放逻辑
        if mouse_click_pos:
            mx, my = mouse_click_pos
            rx, ry, rw, rh = replay_btn_rect
            if rx <= mx <= rx+rw and ry <= my <= ry+rh:
                # 点击了重新播放
                print("重新播放挑战旋律...")
                mouse_click_pos = None # 重置
                # 播放旋律 (不发光)
                for idx in challenge_sequence:
                    start_t = time.time()
                    play_bell(idx)
                    while time.time() - start_t < 0.7:
                        ret, frame = cap.read()
                        if not ret: break
                        frame = cv2.flip(frame, 1)
                        # highlight_idx=-1 表示不强制发光
                        display_frame = draw_base_scene(frame, highlight_idx=-1, status_text="请仔细听...")
                        cv2.imshow("Smart Bianzhong", display_frame)
                        cv2.waitKey(1)
                
                # 重置用户输入
                user_sequence = []
                previous_hit_bells = set()
                continue # 跳过本帧后续逻辑
            
            mouse_click_pos = None # 未点中也重置

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        hit_bells = set()
        if not is_success and results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                draw_hand_landmarks(frame, hand_lm, w, h, FINGERTIPS)
                for tip_idx in FINGERTIPS:
                    lm = hand_lm.landmark[tip_idx]
                    tx, ty = int(lm.x * w), int(lm.y * h)
                    
                    for b_idx, (bx, by) in enumerate(bell_positions):
                        bw, bh = BELLS[b_idx].get("w", 50), BELLS[b_idx].get("h", 80)
                        if abs(tx - bx) < (bw / 2) and abs(ty - by) < (bh / 2):
                            hit_bells.add(b_idx)
                            if b_idx not in previous_hit_bells:
                                if tracker.try_hit(b_idx):
                                    play_bell(b_idx)
                                    user_sequence.append(b_idx)
                                    # 检查是否按错了
                                    curr_idx = len(user_sequence) - 1
                                    if user_sequence[curr_idx] != challenge_sequence[curr_idx]:
                                        user_sequence = [] 
                                        feedback_msg = "弹错了，再来一次吧！"
                                        feedback_color = (100, 100, 255) # BGR: 红色
                                        feedback_timer = time.time() + 1.2
        
        previous_hit_bells = hit_bells
        
        if not is_success and len(user_sequence) == len(challenge_sequence):
            is_success = True
            for i in range(N_BELLS): play_bell(i)
            
        status = f"请复现旋律: {len(user_sequence)} / {len(challenge_sequence)}"
        if is_success: status = "恭喜过关！"
        
        display_frame = draw_base_scene(frame, status_text=status)
        
        # 绘制“重新播放”按钮
        if not is_success:
            rx, ry, rw, rh = replay_btn_rect
            
            # 1. 绘制半透明背景
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (40, 40, 40), -1)
            display_frame = cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0)
            
            # 2. 绘制边框 (亮青色)
            cv2.rectangle(display_frame, (rx, ry), (rx+rw, ry+rh), (200, 200, 100), 2)
            
            # 3. 绘制文字 (PIL)
            img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            # 居中计算
            btn_text = "重播" 
            text_size = 32
            # 简单估算文字宽度 (2个汉字)
            text_w = text_size * 2
            text_x = rx + (rw - text_w) // 2
            text_y = ry + (rh - text_size) // 2 - 5 # 微调垂直位置
            
            put_chinese_text(draw, btn_text, (text_x, text_y), (255, 255, 255), font_size=text_size)
            display_frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        # 过关后增加返回提示
        if is_success:
            img_pil = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            put_chinese_text(draw, "按 Q 返回菜单", (app.width//2 - 80, app.height - 60), (150, 255, 150), font_size=24)
            display_frame = cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Smart Bianzhong", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            app.current_scene = SCENE_MENU
            break
            
    cap.release()
    hands.close()

if __name__ == "__main__":
    main()
