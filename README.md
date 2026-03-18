# 🔔 编钟体感演奏器

> 用手掌在摄像头前"敲击"弧形排列的虚拟编钟，触发对应的古风音调。

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)

---

## 🎬 效果预览

- 打开程序后，摄像头实时捕捉手部动作
- 画面中显示 **8 个弧形排列的编钟**，对应 C 大调五声音阶（宫商角徵羽）
- 用任意手指**伸向并触碰**编钟，触发对应音调
- 触碰时钟体**变亮 + 光晕发光**，视觉反馈清晰
- 支持**双手同时演奏**

## 🎵 音阶对照

| 钟序 | 音名 | 频率 | 中文音名 |
|------|------|------|--------|
| 1 | C4 | 261.63 Hz | 宫 |
| 2 | D4 | 293.66 Hz | 商 |
| 3 | E4 | 329.63 Hz | 角 |
| 4 | G4 | 392.00 Hz | 徵 |
| 5 | A4 | 440.00 Hz | 羽 |
| 6 | C5 | 523.25 Hz | 高宫 |
| 7 | D5 | 587.33 Hz | 高商 |
| 8 | E5 | 659.25 Hz | 高角 |

---

## 🚀 快速开始

### 环境要求

- Python 3.9 ~ 3.13（**注意：Python 3.14 暂不支持 pygame，请用 3.9-3.13**）
- 摄像头（USB 或内置均可）
- Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行

```bash
python bianchong.py
```

按 **Q** 或 **ESC** 退出。

---

## 🛠 技术栈

| 模块 | 用途 |
|------|------|
| `mediapipe` | 手部 21 关键点实时识别 |
| `opencv-python` | 摄像头采集 + 画面渲染 |
| `sounddevice` | 实时音频输出 |
| `scipy / numpy` | 合成钟声音色（基频 + 泛音 + 衰减） |

### 音色合成原理

```
基频 × 1.0
+ 2.76次谐波 × 0.5   ← 金属钟声失谐特征
+ 5.4次谐波  × 0.25
+ 8.93次谐波 × 0.1
× 指数衰减 envelope（快攻慢衰）
```

---

## 📁 项目结构

```
bianchong/
├── bianchong.py      # 主程序
├── requirements.txt  # 依赖列表
└── README.md
```

---

## 🎮 操作说明

- **伸出手掌**：MediaPipe 自动检测，绿色骨骼线显示
- **触碰编钟**：指尖进入钟体范围触发音效（0.35s 冷却防抖）
- **双手演奏**：同时支持 2 只手，可和弦
- **退出**：按 `Q` 或 `ESC`

---

## 💡 扩展思路

- [ ] 加入录制功能，保存演奏片段
- [ ] 支持自定义音阶（如古琴调、编磬调）
- [ ] 添加演奏模式（跟谱练习）
- [ ] 接入更多打击乐器（磬、鼓、木鱼）

---

Made with ❤️ by leonmangsos × OpenClaw AI
