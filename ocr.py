import cv2
import pytesseract
import re
import pyautogui
import pygetwindow as gw
import random
import time  # 导入time模块以便使用sleep

# 设置Tesseract OCR路径
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"图像未能加载，请检查路径：{image_path}")
    return image

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

def extract_numbers(binary_image):
    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(binary_image, output_type=pytesseract.Output.DICT, config=custom_config)

    numbers = [int(data['text'][i]) for i in range(len(data['text']))
               if int(data['conf'][i]) > 60 and re.match(r'^\d+$', data['text'][i])]
    return numbers

def get_window_position(title):
    window = gw.getWindowsWithTitle(title)
    if window:
        return window[0].left, window[0].top, window[0].width, window[0].height
    raise Exception(f"未找到窗口: {title}")

def draw_symbol(symbol, start_x, start_y, symbol_size):
    pyautogui.moveTo(start_x, start_y)
    pyautogui.mouseDown()
    if symbol == '>':
        pyautogui.moveRel(symbol_size, -symbol_size)
        pyautogui.moveRel(-symbol_size, -symbol_size)
        pyautogui.moveRel(0, symbol_size)
    elif symbol == '<':
        pyautogui.moveRel(-symbol_size, -symbol_size)
        pyautogui.moveRel(symbol_size, -symbol_size)
        pyautogui.moveRel(0, symbol_size)
    pyautogui.mouseUp()
    time.sleep(0.010)  # 等待10毫秒

def main():
    image_path = './pic/screenshot.png'
    image = load_image(image_path)
    binary_image = preprocess_image(image)
    numbers = extract_numbers(binary_image)

    window_title = 'MuMu模拟器12'
    left, top, width, height = get_window_position(window_title)
    start_x = left + width // 2
    start_y = top + height // 1.5 + 50
    symbol_size = 5

    if len(numbers) == 0:
        # 如果没有识别到数字，随机选择符号，不包括等于号
        symbol = random.choice(['>', '<'])
        print("未识别到任何数字，随机选择符号:", symbol)
    elif len(numbers) == 1:
        # 如果只识别到一个数字，随机选择符号，不包括等于号
        symbol = random.choice(['>', '<'])
        print("只识别到一个数字，随机选择符号:", symbol)
    else:
        num1, num2 = numbers[0], numbers[1]
        symbol = '>' if num1 > num2 else '<'  # 不包括等于号
        print(f"识别到多个数字，比较结果: {num1} {symbol} {num2}")

    draw_symbol(symbol, start_x, start_y, symbol_size)
    print("所有识别到的数字:", numbers)

if __name__ == "__main__":
    main()
