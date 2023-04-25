import time
import numpy as np
import pygame
from pygame.locals import *

class JoyController:
    def __init__(self, id):
        pygame.joystick.init()
        self.joy = pygame.joystick.Joystick(id)
        self.debug = False
        # self.debug = True

        self.l_hand_x = 0.0
        self.l_hand_y = 0.0
        self.r_hand_x = 0.0
        self.r_hand_y = 0.0

    def update(self, event):
        for eventlist in event:
            # Read stick
            if eventlist.type == JOYAXISMOTION:
                self.l_hand_x = self.joy.get_axis(0)
                self.l_hand_y = self.joy.get_axis(1)
                self.r_hand_x = self.joy.get_axis(2)
                self.r_hand_y = self.joy.get_axis(3)
            self.button_A = self.joy.get_button(0)
            self.button_B = self.joy.get_button(1)
            self.button_X = self.joy.get_button(2)
            self.button_Y = self.joy.get_button(3)

            if self.debug is True:
                print(
                    self.l_hand_x,
                    self.l_hand_y,
                    self.r_hand_x,
                    self.r_hand_y,
                    self.button_A,
                    self.button_B,
                    self.button_X,
                    self.button_Y,
                )

    def get_joy(self):
        action = np.array(
            [
                self.l_hand_y,
                self.l_hand_x,
                self.r_hand_y,
                self.r_hand_x,
            ]
        )
        return action


if __name__ == "__main__":
    pygame.init()
    controller = JoyController(3)
    while True:
        eventlist = pygame.event.get()
        controller.update(eventlist)
        action = controller.get_joy()
        time.sleep(0.01)
