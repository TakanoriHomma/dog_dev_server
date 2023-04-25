import datetime
import argparse
from imageio import RETURN_BYTES
import numpy as np
import itertools

from isaac_env import Isaac_Env

import torch
from torch.utils.tensorboard import SummaryWriter

from sac import SAC
from util.replay_memory import ReplayMemory


class Test_SAC:
    def __init__(self):
        self.config_parser()

        # seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

        # Tesnorboard
        self.writer = SummaryWriter(
            "data/sac/log/{}_SAC".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
        )

    def create_env(self):
        env = Isaac_Env(headless=False)
        env.create_env()
        agent = SAC(
            env.observation_space.shape[0],
            env.action_space,
            self.args,
        )
        memory = ReplayMemory(self.args.replay_size, self.args.seed)
        return env, agent, memory

    def return_state(self, observation):
        state = []
        state.extend(observation["leg_position"])
        state.extend(observation["body_pose"])
        state = np.array(state, dtype=np.float64)
        return state

    def main(self):
        env, agent, memory = self.create_env()

        # Training Loop
        # -------------------------------------------
        total_steps = 0
        updates = 0

        # Infinite loop
        for epi in itertools.count(1):
            episode_reward = 0
            steps = 0
            done = False

            observation = env.reset()
            state = self.return_state(observation)

            while not done:

                # sample action by random or policy until start_steps
                if self.args.start_steps_with_random > total_steps:
                    action = env.random_sample()  # random
                else:
                    action = agent.select_action(state)  # policy

                # Update parameters of all the networks
                if len(memory) > self.args.batch_size:
                    # Number of updates per step in environment
                    for i in range(self.args.updates_per_step):
                        (
                            critic_1_loss,
                            critic_2_loss,
                            policy_loss,
                            ent_loss,
                            alpha,
                        ) = agent.update_parameters(
                            memory, self.args.batch_size, updates
                        )

                        self.writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                        self.writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                        self.writer.add_scalar("loss/policy", policy_loss, updates)
                        self.writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                        self.writer.add_scalar(
                            "entropy_temprature/alpha", alpha, updates
                        )
                        updates += 1

                next_observation, reward, done, _ = env.step(action)
                next_state = self.return_state(next_observation)

                steps += 1
                total_steps += 1
                episode_reward += reward

                if steps == int(self.args.max_episode_steps):
                    done = 1

                # Ignore the "done" signal if it comes from hitting the time horizon.
                mask = (
                    1 if steps == int(self.args.max_episode_steps) else float(not done)
                )

                # Update memory
                memory.push(state, action, reward, next_state, mask)

                state = next_state

            # Finish: steps over
            if total_steps > self.args.total_maximum_steps:
                break

            self.writer.add_scalar("reward/train", episode_reward, epi)
            print(
                "Episode: {}, total steps: {}, steps: {}, reward: {}".format(
                    epi, total_steps, steps, round(episode_reward, 2)
                )
            )

            # Test loop per 10 episode
            # -------------------------------------------
            if epi % 10 == 0 and self.args.eval is True:
                print("---------------------------------------------")
                print("Testing phase")
                avg_reward = 0.0
                num_test_episodes = int(self.args.test_maximum_episodes)

                for i in range(num_test_episodes):
                    observation = env.reset()
                    state = self.return_state(observation)

                    steps_test = 0
                    episode_reward = 0
                    done = False
                    while not done:
                        action = agent.select_action(state, evaluate=True)

                        next_observation, reward, done, _ = env.step(action)
                        state = self.return_state(next_observation)

                        episode_reward += reward

                        steps_test += 1

                        if steps_test == int(self.args.max_episode_steps):
                            done = 1

                    print(
                        "Episode: {}, steps: {}, reward: {}".format(
                            i, steps_test, round(episode_reward, 2)
                        )
                    )

                    avg_reward += episode_reward
                avg_reward /= num_test_episodes

                print("----------------------------------------")
                print("Avg. Reward: ", round(avg_reward, 2))
                print("----------------------------------------")
                self.writer.add_scalar("avg_reward/test", avg_reward, epi)

                agent.save_checkpoint("issac_gym_env", str(epi))

        env.close()

    def config_parser(self):
        parser = argparse.ArgumentParser(description="SAC Args")

        parser.add_argument(
            "--policy",
            default="Gaussian",
            help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
        )
        parser.add_argument(
            "--eval",
            type=bool,
            default=True,
            help="Evaluates a policy a policy every 10 episode (default: True)",
        )
        parser.add_argument(
            "--gamma",
            type=float,
            default=0.99,
            metavar="G",
            help="discount factor for reward (default: 0.99)",
        )
        parser.add_argument(
            "--tau",
            type=float,
            default=0.005,
            metavar="G",
            help="target smoothing coefficient(τ) (default: 0.005)",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.0003,
            metavar="G",
            help="learning rate (default: 0.0003)",
        )
        parser.add_argument(
            "--alpha",
            type=float,
            default=0.2,
            metavar="G",
            help="Temperature parameter α determines the relative importance of the entropy\
                                    term against the reward (default: 0.2)",
        )
        parser.add_argument(
            "--automatic_entropy_tuning",
            type=bool,
            default=False,
            metavar="G",
            help="Automaically adjust α (default: False)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=123456,
            metavar="N",
            help="random seed (default: 123456)",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=256,
            metavar="N",
            help="batch size (default: 256)",
        )
        parser.add_argument(
            "--total_maximum_steps",
            type=int,
            default=1000000,
            metavar="N",
            help="maximum number of steps (default: 1000000)",
        )
        parser.add_argument(
            "--hidden_size",
            type=int,
            default=256,
            metavar="N",
            help="hidden size (default: 256)",
        )
        parser.add_argument(
            "--updates_per_step",
            type=int,
            default=1,
            metavar="N",
            help="model updates per simulator step (default: 1)",
        )
        parser.add_argument(
            "--start_steps_with_random",
            type=int,
            default=10000,
            metavar="N",
            help="Steps sampling random actions (default: 10000)",
        )
        parser.add_argument(
            "--target_update_interval",
            type=int,
            default=1,
            metavar="N",
            help="Value target update per no. of updates per step (default: 1)",
        )
        parser.add_argument(
            "--replay_size",
            type=int,
            default=1000000,
            metavar="N",
            help="size of replay buffer (default: 10000000)",
        )
        parser.add_argument(
            "--cuda", action="store_true", help="run on CUDA (default: False)"
        )
        parser.add_argument(
            "--max_episode_steps",
            type=int,
            default=500,
            metavar="N",
        )
        parser.add_argument(
            "--test_maximum_episodes",
            type=int,
            default=10,
            metavar="N",
        )

        self.args = parser.parse_args()

        print("-----------------------------")
        print(self.args)
        print("-----------------------------")


if __name__ == "__main__":
    test_sac = Test_SAC()
    test_sac.main()