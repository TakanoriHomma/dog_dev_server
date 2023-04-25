import time
import numpy as np
import pygame
from util.control_joy import JoyController
from isaac_env import Isaac_Env


class Test_Isaac_Env:
    def __init__(self):
        np.set_printoptions(precision=3, suppress=True)
        pygame.init()
        self.joy = JoyController(0)

        self.config_task()
        self.count = 0

    def config_task(self):
        self.max_episodes = 5
        self.max_steps = 100000

    def get_joy_action(self):
        coef = 1.0
        joy_deadzone = 0.1
        eventlist = pygame.event.get()
        self.joy.update(eventlist)
        action = self.joy.get_joy()
        for i in range(len(action)):
            if abs(action[i]) < joy_deadzone:
                action[i] = 0.0
        action = -coef * action
        action = [action[0]] * 12
        self.count +=1
        #action[1] = 0.1
        #action[4] = -0.3*np.sin(self.count/100)
        ##action[7] = -0.2
        ##action[10] = -0.1
        
        action[0] = 0.0
        #action[1] = 0.0
        action[2] = 0.0
        action[3] = 0.0
        action[4] = 0.0
        action[5] = 0.0
        action[6] = 0.0
        action[7] = 0.0
        action[8] = 0.0
        action[9] = 0.0
        action[10] = 0.0
        action[11] = 0.0
        
        return action

    def main(self):

        for epi in range(self.max_episodes):

            self.env = Isaac_Env(headless=False)
            self.env.create_env()
            self.env.reset()

            done = False
            for stp in range(self.max_steps):
                action = self.get_joy_action()
                # action = self.env.random_sample()

                observation, reward, done, _ = self.env.step(action)

                self.env.render(render_collision=False)

                print("--------------")
                print("episode: ", epi)
                print("steps: ", stp)
                print("observation: ", observation)
                print("action: ", action)
                print("reward: ", reward)
                

                # reset command
                if self.joy.button_A == 1:
                    done = True
                    time.sleep(1)

                if done:
                    print("----- Success -----")
                    break

            self.env.close()


if __name__ == "__main__":
    test_isaac_env = Test_Isaac_Env()
    test_isaac_env.main()