"""
编钟体感演奏器
用手掌在摄像头前"敲击"弧形排列的编钟，触发对应音调。
依赖: mediapipe, opencv-python, sounddevice, scipy, numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import threading
import time
import math

# ─── 音频合成 ────────────────────────────────────────────────────────────────

SAMPLE_RATE = 44100

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

def play_bell(freq: float):
    """在独立线程中播放，不阻塞主循环"""
    def _play():
        audio = synth_bell(freq)
        sd.play(audio, samplerate=SAMPLE_RATE)
    threading.Thread(target=_play, daemon=True).start()

# ─── 编钟定义（C大调五声音阶 + 扩展，8个钟）────────────────────────────────

BELLS = [
    {"note": "宫 C4", "freq": 261.63,  "color": (180, 80,  220)},
    {"note": "商 D4", "freq": 293.66,  "color": (80,  120, 220)},
    {"note": "角 E4", "freq": 329.63,  "color": (60,  190, 120)},
    {"note": "徵 G4", "freq": 392.00,  "color": (200, 180, 40)},
    {"note": "羽 A4", "freq": 440.00,  "color": (200, 100, 40)},
    {"note": "宫 C5", "freq": 523.25,  "color": (180, 50,  180)},
    {"note": "商 D5", "freq": 587.33,  "color": (60,  160, 210)},
    {"note": "角 E5", "freq": 659.25,  "color": (50,  200, 100)},
]

N_BELLS = len(BELLS)

# ─── 编钟几何（弧形排列）────────────────────────────────────────────────────

def compute_bell_positions(frame_w: int, frame_h: int):
    """计算每个编钟在画面中的中心位置（弧形）"""
    cx = frame_w // 2
    arc_radius = int(frame_h * 0.55)
    arc_bottom_y = int(frame_h * 0.92)
    # 弧形范围：从 -60° 到 +60°（相对竖直方向）
    angle_start = -60
    angle_end   =  60
    positions = []
    for i in range(N_BELLS):
        t = i / (N_BELLS - 1)  # 0 ~ 1
        angle_deg = angle_start + t * (angle_end - angle_start)
        angle_rad = math.radians(angle_deg)
        x = int(cx + arc_radius * math.sin(angle_rad))
        y = int(arc_bottom_y - arc_radius * (1 - math.cos(angle_rad)))
        positions.append((x, y))
    return positions

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

def draw_bell(frame, pos, bell, idx, hit: bool, glow: bool):
    x, y = pos
    color = bell["color"]
    bell_w = 46
    bell_h = 70

    # 发光光晕
    if glow:
        for r in range(5, 0, -1):
            alpha_layer = frame.copy()
            cv2.ellipse(alpha_layer, (x, y), (bell_w + r*6, bell_h + r*6),
                        0, 0, 360, color, -1)
            cv2.addWeighted(alpha_layer, 0.06, frame, 0.94, 0, frame)

    # 悬挂绳
    cv2.line(frame, (x, y - bell_h - 10), (x, y - bell_h + 5), (180, 160, 130), 2)

    # 钟体（梯形近似：上窄下宽）
    pts = np.array([
        [x - bell_w//3, y - bell_h],
        [x + bell_w//3, y - bell_h],
        [x + bell_w//2, y],
        [x - bell_w//2, y],
    ], dtype=np.int32)

    bright = tuple(min(255, int(c * 1.5)) for c in color) if hit else color
    cv2.fillPoly(frame, [pts], bright)
    cv2.polylines(frame, [pts], True, (255, 255, 255), 1)

    # 钟口弧线（装饰）
    cv2.ellipse(frame, (x, y), (bell_w//2, 10), 0, 0, 180, (255, 255, 255), 1)

    # 音名标签
    label = bell["note"]
    font_scale = 0.42
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    cv2.putText(frame, label, (x - tw//2, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

def draw_arch_beam(frame, positions):
    """画弧形横梁连接各钟顶"""
    tops = [(x, y - 70) for x, y in positions]
    for i in range(len(tops) - 1):
        cv2.line(frame, tops[i], tops[i+1], (160, 130, 80), 4)

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

# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头，请检查设备连接")
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

    # 指尖关键点索引（拇指+四指）
    FINGERTIPS = [4, 8, 12, 16, 20]
    HIT_RADIUS = 40  # 碰撞检测半径（像素）

    print("🔔 编钟体感演奏器启动！")
    print("   伸出手掌，用指尖敲击画面中的编钟")
    print("   按 Q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像，更直观
        h, w = frame.shape[:2]

        # 计算钟的位置（每帧动态适配分辨率）
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
                        # 碰撞：指尖在钟体范围内
                        bell_w2, bell_h2 = 46, 70
                        in_x = abs(tip_x - bx) < bell_w2
                        in_y = (by - bell_h2) < tip_y < (by + 15)
                        dist = math.hypot(tip_x - bx, tip_y - (by - bell_h2 // 2))

                        if (in_x and in_y) or dist < HIT_RADIUS:
                            hit_bells.add(b_idx)
                            if tracker.try_hit(b_idx):
                                play_bell(BELLS[b_idx]["freq"])

        # ── 绘制编钟 ──
        draw_arch_beam(frame, bell_positions)
        for i, (pos, bell) in enumerate(zip(bell_positions, BELLS)):
            is_hit = i in hit_bells
            is_glow = tracker.recently_hit(i, window=0.25)
            draw_bell(frame, pos, bell, i, is_hit, is_glow)

        # ── UI 提示 ──
        cv2.putText(frame, "编钟体感演奏器  |  Q 退出",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # 手部检测状态
        if results.multi_hand_landmarks:
            n = len(results.multi_hand_landmarks)
            cv2.putText(frame, f"检测到 {n} 只手",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (80, 220, 120), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "伸出双手开始演奏...",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (120, 120, 200), 1, cv2.LINE_AA)

        cv2.imshow("🔔 编钟体感演奏器", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("👋 再见！")

if __name__ == "__main__":
    main()
