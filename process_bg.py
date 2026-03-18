import cv2
import numpy as np

# 读取原图 (支持中文路径)
img_path = 'd:/其他项目/MotionMusic/image.png'
try:
    img_data = np.fromfile(img_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
except Exception as e:
    print(f"Error reading image: {e}")
    exit(1)

if img is None:
    print(f"Failed to decode image from {img_path}")
    exit(1)

# 转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 二值化提取背景（白色背景），设定阈值，大于240认为是白色背景
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

# 形态学操作去噪
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# 寻找最大轮廓裁剪出编钟主体
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No contours found!")
    exit(1)

c = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(c)

# 裁剪
img_cropped = img[y:y+h, x:x+w]
mask_cropped = mask[y:y+h, x:x+w]

# 转换为 BGRA 并应用掩膜作为透明通道
bgra = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2BGRA)
bgra[:, :, 3] = mask_cropped

# 保存结果
output_path = 'd:/其他项目/MotionMusic/bianchong_repo/bianchong-main/bell_transparent.png'
cv2.imencode('.png', bgra)[1].tofile(output_path)
print(f"成功生成去背图片: {output_path}")
