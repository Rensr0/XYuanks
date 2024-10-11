import os
import time
import logging
from threading import Thread
from PIL import Image
import numpy as np
import pygetwindow as gw
import pyautogui
import cv2

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
WINDOW_TITLE = "MuMu模拟器12"
PIC_DIRECTORY = "pic"
SLEEP_INTERVAL = 1
TEMPLATE_PATH = './pic/finish.png'
START_TEMPLATE_PATHS = ['./pic/start.png', './pic/start1.png']
MATCH_THRESHOLD = 0.8

def create_pic_directory():
    """创建存储截图的目录"""
    os.makedirs(PIC_DIRECTORY, exist_ok=True)
    logging.info(f"确保目录存在: {PIC_DIRECTORY}")

def is_image_matched(image_np, template_path, match_threshold=MATCH_THRESHOLD):
    """判断图像是否与模板匹配"""
    template = cv2.imread(template_path)
    if template is None:
        logging.error(f"未能读取模板图像: {template_path}")
        return False

    if image_np.shape[0] < template.shape[0] or image_np.shape[1] < template.shape[1]:
        logging.error("输入图像尺寸小于模板图像尺寸，无法进行匹配")
        return False

    result = cv2.matchTemplate(image_np, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= match_threshold

def take_screenshot(window):
    """截取窗口的截图"""
    if window is None:
        logging.warning("无法截取截图，因为窗口未激活。")
        return None

    left, top, width, height = window.left, window.top, window.width, window.height
    screenshot = pyautogui.screenshot(region=(left, top + height // 5, width, height // 6))
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def run_ocr(image):
    """运行OCR处理"""
    screenshot_path = os.path.join(PIC_DIRECTORY, 'screenshot.png')
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(screenshot_path)

    try:
        os.system(f'python ocr.py {screenshot_path}')  # 调用外部OCR脚本
        logging.info("OCR处理完成")
    except Exception as e:
        logging.error(f"OCR处理失败: {e}")

def find_window_by_title(title):
    """根据窗口标题查找窗口"""
    windows = gw.getWindowsWithTitle(title)
    if windows:
        return windows[0]
    logging.warning(f"未找到窗口: {title}")
    return None

def activate_window():
    """激活指定标题的窗口"""
    window = find_window_by_title(WINDOW_TITLE)
    if window:
        logging.info(f"激活窗口: {WINDOW_TITLE}")
        window.activate()
    return window

def main():
    create_pic_directory()
    last_image = None

    # 等待匹配到 start.png
    while not (window := activate_window()) or (current_image := take_screenshot(window)) is None:
        time.sleep(SLEEP_INTERVAL)

    while not is_image_matched(current_image, START_TEMPLATE_PATHS[0]) and not is_image_matched(current_image, START_TEMPLATE_PATHS[1]):
        time.sleep(SLEEP_INTERVAL)
        current_image = take_screenshot(window)

    logging.info("检测到启动图像，开始运行")
    time.sleep(0.01)

    # 开始主循环
    while True:
        window = activate_window()
        current_image = take_screenshot(window)

        if current_image is None:
            time.sleep(SLEEP_INTERVAL)
            continue

        if is_image_matched(current_image, TEMPLATE_PATH):
            logging.info("检测到完成图像，停止程序")
            break

        if last_image is None or not np.array_equal(current_image, last_image):
            logging.info("窗口内容已改变，进行OCR处理")
            Thread(target=run_ocr, args=(current_image,)).start()

        last_image = current_image
        time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
