import pyautogui
from random import randint
import time

def mouse_move():
    while True:
        # pyautogui.moveTo(100, 100, duration = 1) # move the mouse
        pyautogui.press('volumedown')
        time.sleep(60) # Every 1 min
        time.sleep(randint(10, 20))
        # pyautogui.moveTo(50, 100, duration = 1) # move the mouse
        # pyautogui.press("down", presses=2)
        # pyautogui.keyDown('w')
        # pyautogui.keyUp('w')
        time.sleep(1)
        pyautogui.press('volumeup')

pyautogui.FAILSAFE = False
mouse_move()