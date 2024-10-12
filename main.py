import os
import time
import logging
import random
import numpy as np
import pygetwindow as gw
import pyautogui
import cv2
import pytesseract
from concurrent.futures import ThreadPoolExecutor

# 设置 Tesseract OCR 路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置参数
WINDOW_TITLE = "MuMu模拟器12"
PIC_DIRECTORY = "pic"
SLEEP_INTERVAL = 0.3  # 减少睡眠间隔
TEMPLATE_PATH = './pic/finish.png'
MATCH_THRESHOLD = 0.8

# 符号绘制参数
SYMBOL_SIZE = 5

def create_pic_directory():
    os.makedirs(PIC_DIRECTORY, exist_ok=True)
    logging.info(f"确保目录存在: {PIC_DIRECTORY}")

def is_image_matched(image_np, template_path, match_threshold=MATCH_THRESHOLD):
    template = cv2.imread(template_path)
    if template is None:
        logging.error(f"未能读取模板图像: {template_path}")
        return False

    result = cv2.matchTemplate(image_np, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= match_threshold

def take_screenshot(window, top_left=(32, 314), bottom_right=(731, 431)):  # pk选项
# def take_screenshot(window, top_left=(14, 214), bottom_right=(739, 335)):  # 练习选项
    if window is None:
        logging.warning("无法截取截图，因为窗口未激活。")
        return None

    left, top = window.left, window.top
    screenshot_region = (left + top_left[0], top + top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    screenshot = pyautogui.screenshot(region=screenshot_region)
    return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

def run_ocr(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)

    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(binary_image, output_type=pytesseract.Output.DICT, config=custom_config)

    numbers = [int(data['text'][i]) for i in range(len(data['text']))
               if int(data['conf'][i]) > 60 and data['text'][i].isdigit()]

    return numbers

def find_window_by_title(title):
    windows = gw.getWindowsWithTitle(title)
    if windows:
        return windows[0]
    logging.warning(f"未找到窗口: {title}")
    return None

def draw_symbol(symbol, start_x, start_y):
    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.mouseDown()

    if symbol == '>':
        pyautogui.moveRel(SYMBOL_SIZE, -SYMBOL_SIZE, duration=0.1)
        pyautogui.moveRel(-SYMBOL_SIZE, -SYMBOL_SIZE, duration=0.1)
        pyautogui.moveRel(0, SYMBOL_SIZE, duration=0.1)
    elif symbol == '<':
        pyautogui.moveRel(-SYMBOL_SIZE, -SYMBOL_SIZE, duration=0.1)
        pyautogui.moveRel(SYMBOL_SIZE, -SYMBOL_SIZE, duration=0.1)
        pyautogui.moveRel(0, SYMBOL_SIZE, duration=0.1)

    pyautogui.mouseUp()
    time.sleep(0.1)

def process_numbers(numbers, window, previous_numbers):
    start_x = window.left + window.width // 2
    start_y = window.top + window.height // 1.5 + 50

    if len(numbers) == 0:
        symbol = random.choice(['>', '<'])
        logging.info(f"未识别到任何数字，随机选择符号: {symbol}")
    elif len(numbers) == 1:
        symbol = random.choice(['>', '<'])
        logging.info(f"只识别到一个数字，随机选择符号: {symbol}")
    else:
        num1, num2 = numbers[0], numbers[1]
        symbol = '>' if num1 > num2 else '<'
        logging.info(f"识别到多个数字，比较结果: {num1} {symbol} {num2}")

    # 如果数字没有变化，选择下一个随机符号
    if previous_numbers is not None and numbers == previous_numbers:
        symbol = random.choice(['>', '<'])
        logging.info(f"数字未变化，随机选择新的符号: {symbol}")

    draw_symbol(symbol, start_x, start_y)

def process_image(image, window, previous_numbers):
    numbers = run_ocr(image)
    logging.info(f"识别到的数字: {numbers}")
    process_numbers(numbers, window, previous_numbers)
    return numbers

def main():
    create_pic_directory()
    last_image = None
    previous_numbers = None

    with ThreadPoolExecutor() as executor:
        while True:
            try:
                window = find_window_by_title(WINDOW_TITLE)
                current_image = take_screenshot(window)

                if current_image is None:
                    time.sleep(SLEEP_INTERVAL)
                    continue

                if is_image_matched(current_image, TEMPLATE_PATH):
                    logging.info("检测到完成图像，停止程序")
                    break

                if last_image is None or not np.array_equal(current_image, last_image):
                    logging.info("窗口内容已改变，进行OCR处理")
                    previous_numbers = executor.submit(process_image, current_image, window, previous_numbers).result()

                last_image = current_image
                time.sleep(SLEEP_INTERVAL)

            except Exception as e:
                logging.error(f"发生异常: {e}")
                time.sleep(SLEEP_INTERVAL)

if __name__ == "__main__":
    main()
