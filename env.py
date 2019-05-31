import cv2
import time
import numpy as np
from io import BytesIO
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pickle as pkl

class Env:
    def __init__(self):
        self.driver = webdriver.Chrome("/home/kmc/Desktop/chromedriver")
        self.driver.get("http://www.trex-game.skipser.com/")
        self.body = self.driver.find_element_by_tag_name("body")
        self.canvas = self.driver.find_element_by_tag_name("canvas")

        with open('gameover.pkl', 'rb') as f:
            self.game_over_array = pkl.load(f)
        
    def __del__(self):
        self.driver.close()

    @property
    def state_size(self):
        h, w = self._get_screen().shape

        return [4, h, w]
    
    @property
    def action_size(self):
        return 2
    
    def reset(self):
        # gameover 직후 바로 스페이스바를 눌러도 시작하지 않음
        # 약 1초정도 기다렸다가 해야함.
        time.sleep(1)
        self.body.send_keys(Keys.SPACE)

        self.history = np.stack([self._get_screen()] * 4)

        return self.history

    def step(self, action):
        
        # JUMP
        if action == 1:
            self.body.send_keys(Keys.SPACE)

        frame = self._get_screen()

        # Game Over
        if np.all(frame[20:27, 72:80] == self.game_over_array):
            reward = -100
            done = True
        else:
            reward = 1
            done = False

        next_state = np.expand_dims(frame, axis=0)

        self.history = np.concatenate((self.history[1:, :, :], next_state), axis=0)
        
        return self.history, reward, done

    def _get_screen(self):
        canvas_png = self.canvas.screenshot_as_png
        image = Image.open(BytesIO(canvas_png))
        img = np.array(image)
        img = cv2.cvtColor(img[::4, ::4, :], cv2.COLOR_BGR2GRAY)

        return img

if __name__ == '__main__':
    import cv2

    env = Env()

    state = env.reset()
    
    while True:
        s,r,d = env.step(0)

        print(d)
            

    # 
    # import pickle as pkl
    # while True:
    #     s,r,d = env.step(0)
        
    #     with open("gameover.pkl", "wb") as f:
    #         pkl.dump(s[0, 20:27, 72:80], f)
    #     cv2.imshow("Img", s[0, 20:27, 72:80])
        
    #     cv2.waitKey(1)

            
            