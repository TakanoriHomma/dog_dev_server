# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.go1 import Go1
from omniisaacgymenvs.robots.articulations.views.go1_view import Go1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.quadruped.utils.go1_sys_model import Go1SysModel

from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.rotations import *

import numpy as np
import torch
import math


class Go1Task(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self.delta_q = 0.

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["cosmetic"] = self._task_cfg["env"]["learn"]["cosmeticRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = 1 / 60
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._go1_translation = torch.tensor([0.0, 0.0, 0.35])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 48
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        
        self.go1_sys = Go1SysModel()
        
        
        return

    def set_up_scene(self, scene) -> None:
        self.get_go1()
        super().set_up_scene(scene)
        self._go1s = Go1View(prim_paths_expr="/World/envs/.*/Go1", name="Go1view")
        scene.add(self._go1s)
        scene.add(self._go1s._knees)
        scene.add(self._go1s._base)
        scene.add(self._go1s._foots)
        scene
        
        init_torso_position = self._go1s.get_world_poses(clone=False)
        self.init_torso_x =init_torso_position[0][0,0]
        

        
        # self.get_contact()
        # print(self._go1s.get_contact_view())
        
        # self._go1_foots = ArticulationView(prim_paths_expr="/World/envs/.*/go1/.*", name="foot_view", reset_xform_properties=False)
        return

    # def get_contact(self) -> None:
    #     # robots = Go1View(prim_paths_expr="/World/envs/.*/go1/.*_foot", enable_dof_force_sensors = True)     
    #     sensor_readings = self._go1s.get_contact_view()
    #     print(sensor_readings)
        

        

    def get_go1(self):
        go1 = Go1(prim_path=self.default_zero_env_path + "/Go1", name="Go1", translation=self._go1_translation)
        self._sim_config.apply_articulation_settings("Go1", get_prim_at_path(go1.prim_path), self._sim_config.parse_actor_config("Go1"))
        
        
        # Configure joint properties
        joint_paths = []
        joint_paths.append(f"trunk/FR_hip_joint")
        joint_paths.append(f"FR_hip/FR_thigh_joint")
        joint_paths.append(f"FR_thigh/FR_calf_joint")
        joint_paths.append(f"trunk/FL_hip_joint")
        joint_paths.append(f"FL_hip/FL_thigh_joint")
        joint_paths.append(f"FL_thigh/FL_calf_joint")
        joint_paths.append(f"trunk/RR_hip_joint")
        joint_paths.append(f"RR_hip/RR_thigh_joint")
        joint_paths.append(f"RR_thigh/RR_calf_joint")
        joint_paths.append(f"trunk/RL_hip_joint")
        joint_paths.append(f"RL_hip/RL_thigh_joint")
        joint_paths.append(f"RL_thigh/RL_calf_joint")
        for joint_path in joint_paths:
            set_drive(f"{go1.prim_path}/{joint_path}", "angular", "position", 0, 45, 3, 1000)
            # set_drive(f"{go1.prim_path}/{joint_path}", "angular", "position", 0, 400, 40, 1000)

        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        dof_names = go1.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            print(name)
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
            

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._go1s.get_world_poses(clone=False)
        root_velocities = self._go1s.get_velocities(clone=False)
        dof_pos = self._go1s.get_joint_positions(clone=False)
        dof_vel = self._go1s.get_joint_velocities(clone=False)
        # print("body_posiotion = " ,torso_position[:,:])
        # print("foot_heights = " ,self._go1s._foots_heights[0][0,2:])
        # print("foot_heights = " ,self._go1s._foots_heights[0][1,2:])
        # print("foot_heights = " ,self._go1s._foots_heights[0][2,2:])
        # print("foot_heights = " ,self._go1s._foots_heights[0][3,2:])
        self.FR = self._go1s._foots_heights[0][0,2:]
        self.FL = self._go1s._foots_heights[0][1,2:]
        self.RR = self._go1s._foots_heights[0][2,2:]
        self.RL = self._go1s._foots_heights[0][3,2:]
        
        
        self.torso_x =torso_position[:,0]
        self.torso_y =torso_position[:,1]

        # print(self.q)
        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]
        
        
        # self._go1_foots = ArticulationView(prim_paths_expr="/World/envs/.*/go1/.*_foot", name="foot_view", reset_xform_properties=False)
        # self.get_contact()
        
        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )
        
        # print(dof_pos)
        
        # sensor_force_torques = self._go1_foots._physics_view.get_force_sensor_forces()
        # print(sensor_force_torques)

        obs = torch.cat(
            (
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                commands_scaled,
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf[:] = obs

        observations = {
            self._go1s.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self._go1s.count, dtype=torch.int32, device=self._device)
        # self.actions[:] = actions.clone().to(self._device)
        
        
        
        dof_pos = self._go1s.get_joint_positions(clone=False)
        
        
        # self.actions[:] = actions.clone().to(self._device)
        # self.actions[:,1] = torch.tensor(0)
        # self.actions[:,5] = torch.tensor(-0.4)
        # self.actions[:,9] = torch.tensor(0.1)
        
        self.dof_pos_FR = dof_pos[0,:3]
        self.delta_q = self.calculate_jacobian()
        self.delta_q[0] = (self.delta_q[0]//3.14)/5
        self.delta_q[1] = (self.delta_q[1]//3.14)/5
        self.delta_q[2] = (self.delta_q[2]//3.14)/5
        
        self.actions[:] = actions.clone().to(self._device)
        self.actions[:,1] = torch.tensor(self.delta_q[0])
        self.actions[:,5] = torch.tensor(self.delta_q[1])
        self.actions[:,9] = torch.tensor(self.delta_q[2])
        # print(self.actions)
        
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt 
        self.current_targets[:] = tensor_clamp(current_targets, self.go1_dof_lower_limits, self.go1_dof_upper_limits)
        self._go1s.set_joint_position_targets(self.current_targets, indices)


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._go1s.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities

        self.current_targets[env_ids] = dof_pos[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._go1s.set_joint_positions(dof_pos, indices)
        self._go1s.set_joint_velocities(dof_vel, indices)

        self._go1s.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices)
        self._go1s.set_velocities(root_vel, indices)

        self.commands_x[env_ids] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_y[env_ids] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (num_resets, 1), device=self._device
        ).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (num_resets, 1), device=self._device
        ).squeeze()

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.

    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._go1s.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._go1s.get_dof_limits()
        self.go1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.go1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)

        self.commands = torch.zeros(self._num_envs, 3, dtype=torch.float, device=self._device, requires_grad=False)
        self.commands_y = self.commands.view(self._num_envs, 3)[..., 1]
        self.commands_x = self.commands.view(self._num_envs, 3)[..., 0]
        self.commands_yaw = self.commands.view(self._num_envs, 3)[..., 2]

        # initialize some data used later on
        self.extras = {}
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self._device).repeat(
            (self._num_envs, 1)
        )
        self.actions = torch.zeros(
            self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False
        )
        self.last_dof_vel = torch.zeros((self._num_envs, 12), dtype=torch.float, device=self._device, requires_grad=False)
        self.last_actions = torch.zeros(self._num_envs, self.num_actions, dtype=torch.float, device=self._device, requires_grad=False)

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(self._go1s.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._go1s.get_world_poses(clone=False)
        root_velocities = self._go1s.get_velocities(clone=False)
        dof_pos = self._go1s.get_joint_positions(clone=False)
        dof_vel = self._go1s.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        rew_lin_vel_z = torch.square(base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_joint_acc = torch.sum(torch.square(self.last_dof_vel - dof_vel), dim=1) * self.rew_scales["joint_acc"]
        rew_action_rate = torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        rew_cosmetic = torch.sum(torch.abs(dof_pos[:, 0:4] - self.default_dof_pos[:, 0:4]), dim=1) * self.rew_scales["cosmetic"] 
        
        
        total_reward = (0.0200 - self.FR)**2 - (self.torso_x)**4 - (0.0200 - self.torso_y)**4

        # total_reward = (0.0200 - self.FR)**4 - (0.0200 - self.FL)**5 - (0.0200 - self.RR)**5 - (0.0200 - self.RL)**5
        # print(total_reward)
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = self._go1s.is_base_below_threshold(threshold=0.2, ground_heights=0.0)
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()


    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
        
    
    def calculate_jacobian(self) -> np.array:
        idx = 0
        # current_angles = np.array([-0.1, 0.8, -1.5])
        current_angles = self.dof_pos_FR
        current_angles = current_angles.cpu().numpy()
        # self.delta_p = np.array([0.2, 0.0, 0.3])
        self._env_pos_x = self._env_pos[0,0]
        self.target_p = torch.tensor([self._env_pos_x + 0.355, 0.0, 0.32]).to(self._device)
        # self.target_p = torch.tensor([0.335, 0.0, 0.32]).to(self._device)
        self.FR_all = self._go1s._foots_heights[0][0,:]
        self.FR_all[1] = 0.0
        delta_p = self.target_p - self.FR_all
        self.delta_p = delta_p.cpu().numpy()
        self.J = self.go1_sys.jacobian(idx, current_angles)
        self.delta_q = np.dot(np.linalg.inv(self.J),self.delta_p) 
        # self.q = current_angles + self.delta_q

        
        return self.delta_q

