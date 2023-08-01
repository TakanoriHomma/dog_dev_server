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
from omniisaacgymenvs.robots.articulations.a1_bipedal import A1_bipedal
from omniisaacgymenvs.robots.articulations.views.a1_view import A1View
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive


from omni.isaac.quadruped.utils.a1_sys_model import A1SysModel

from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.prims import get_prim_at_path

from omni.isaac.core.utils.torch.rotations import *

from omni.isaac.quadruped.utils.a1_classes import A1State, A1Measurement, A1Command

import numpy as np
import torch
import math
import matplotlib.pyplot as plt

import quaternion


class A1_bipedal_Task(RLTask):
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
        self._a1_translation = torch.tensor([0.0, 0.0, 0.5])
        self._a1_quat = torch.tensor([0.5, 0.5, -0.5, 0.5]) # standing
        # self._a1_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]) # lying
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._num_observations = 42
        self._num_actions = 12

        RLTask.__init__(self, name, env)
        
        self.a1_sys = A1SysModel()
        
        # self.potentials = 0
        # self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self._num_envs)
        # self.prev_potentials = self.potentials.clone()
        
        return

    def set_up_scene(self, scene) -> None:
        self.get_a1()
        super().set_up_scene(scene)
        self._a1s = A1View(prim_paths_expr="/World/envs/.*/A1", name="A1view")
        scene.add(self._a1s)
        scene.add(self._a1s._knees)
        scene.add(self._a1s._base)
        scene.add(self._a1s._foots)
        scene
        
        init_torso_position = self._a1s.get_world_poses(clone=False)
        self.init_torso_x =init_torso_position[0][:,0]
        self.init_torso_y =init_torso_position[0][:,1]

        return
        

    def get_a1(self):
        self._a1 = A1_bipedal(prim_path=self.default_zero_env_path + "/A1", name="A1", translation=self._a1_translation, orientation=self._a1_quat)
        self._sim_config.apply_articulation_settings("A1", get_prim_at_path(self._a1.prim_path), self._sim_config.parse_actor_config("A1"))
        
        
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
            # set_drive(f"{self._a1.prim_path}/{joint_path}", "angular", "position", 0, 45, 3, 1000)
            set_drive(f"{self._a1.prim_path}/{joint_path}", "angular", "position", 0, 10000, 1000, 1000)
            #PDの値設定
            # set_drive(f"{self._a1.prim_path}/{joint_path}", "angular", "position", 0, 10000, 100, 1000)
            # set_drive(f"{self._a1.prim_path}/{joint_path}", "angular", "position", 0, 400, 40, 1000)
            # set_drive(f"{self._a1.prim_path}/{joint_path}", "angular", "position", 0, 5, 1, 1000)

        self.default_dof_pos = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)
        dof_names = self._a1.dof_names
        for i in range(self.num_actions):
            name = dof_names[i]
            print(name)
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle
        
        
            

    def get_observations(self) -> dict:
        torso_position, torso_rotation = self._a1s.get_world_poses(clone=False)
        root_velocities = self._a1s.get_velocities(clone=False)
        dof_pos = self._a1s.get_joint_positions(clone=False)
        dof_vel = self._a1s.get_joint_velocities(clone=False)
        
        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]

        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity) * self.lin_vel_scale
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity) * self.ang_vel_scale
        projected_gravity = quat_rotate(torso_rotation, self.gravity_vec)
        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale
    
        
        # print(projected_gravity)
        # print(base_lin_vel)
        print(base_ang_vel)

        commands_scaled = self.commands * torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            requires_grad=False,
            device=self.commands.device,
        )
        
        # get foot heights
        self.FR = self._a1s._foots_heights[0][0,2:]
        self.FL = self._a1s._foots_heights[0][1,2:]
        self.RR = self._a1s._foots_heights[0][2,2:]
        self.RL = self._a1s._foots_heights[0][3,2:]
        
        self.torso_x =torso_position[:,0]
        self.torso_y =torso_position[:,1]
        self.torso_z =torso_position[:,2]


        dof_pos_scaled = (dof_pos - self.default_dof_pos) * self.dof_pos_scale


        
        #get foot force
        self.foot_contact = self._a1s._foots.get_net_contact_forces(clone=False).view(self._num_envs, 4, 3)
        self.foot_contact_z = self.foot_contact[:,:,2]
        self.foot_contact_z_FL = self.foot_contact_z[:,0]
        self.foot_contact_z_FR = self.foot_contact_z[:,1]
        self.foot_contact_z_RL = self.foot_contact_z[:,2]
        self.foot_contact_z_RR = self.foot_contact_z[:,3]
        self.foot_contact_z_FL_output = torch.where(self.foot_contact_z_FL > 0, torch.tensor(2.0, device='cuda:0'), torch.tensor(-1.0, device='cuda:0'))
        self.foot_contact_z_FR_output = torch.where(self.foot_contact_z_FR > 0, torch.tensor(2.0, device='cuda:0'), torch.tensor(-1.0, device='cuda:0'))
        self.foot_contact_z_RL_output = torch.where(self.foot_contact_z_RL > 0, torch.tensor(2.0, device='cuda:0'), torch.tensor(-1.0, device='cuda:0'))
        self.foot_contact_z_RR_output = torch.where(self.foot_contact_z_RR > 0, torch.tensor(2.0, device='cuda:0'), torch.tensor(-1.0, device='cuda:0'))

        # self.prev_potentials = potentials.clone()
        # self.potentials = -torch.linalg.norm(to_target, p=2, dim=-1) / self.dt
        
        obs = torch.cat(
            (
                base_ang_vel,
                commands_scaled,
                dof_pos_scaled,
                dof_vel * self.dof_vel_scale,
                self.actions,
            ),
            dim=-1,
        )
        self.obs_buf[:] = obs

        observations = {
            self._a1s.name: {
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

        indices = torch.arange(self._a1s.count, dtype=torch.int32, device=self._device)
        # ランダムインプット？
        self.actions[:] = actions.clone().to(self._device)
        
        
        # 各関節のアクションを0
        # self.actions[:,0] = torch.tensor(0) # FL_hip
        # self.actions[:,1] = torch.tensor(0) # FR_hip
        # self.actions[:,2] = torch.tensor(0) # RL_hip
        # self.actions[:,3] = torch.tensor(0) # RR_hip
        # self.actions[:,4] = torch.tensor(-np.pi/12) # FL_thigh
        # self.actions[:,5] = torch.tensor(-np.pi/12) # FR_thigh
        # self.actions[:,6] = torch.tensor(0) # RL_thigh
        # self.actions[:,7] = torch.tensor(0) # RR_thigh
        # self.actions[:,8] = torch.tensor(np.pi/12) # FL_calf
        # self.actions[:,9] = torch.tensor(np.pi/12) # FR_calf
        # self.actions[:,10] = torch.tensor(0) # RL_calf
        # self.actions[:,11] = torch.tensor(0) # RR_calf
        
        
        # 40step待つ(着地後安定化させるため)
        # self.actions = torch.where(self.progress_buf.repeat_interleave(self._num_actions) > 40, self.actions.flatten(), 0)
        # self.actions = self.actions.view(self._num_envs, -1)
        
        current_targets = self.current_targets + self.action_scale * self.actions * self.dt 
        self.current_targets[:] = tensor_clamp(current_targets, self.a1_dof_lower_limits, self.a1_dof_upper_limits)
        self._a1s.set_joint_position_targets(self.current_targets, indices)
        

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # randomize DOF velocities
        velocities = torch_rand_float(-0.1, 0.1, (num_resets, self._a1s.num_dof), device=self._device)
        dof_pos = self.default_dof_pos[env_ids]
        dof_vel = velocities

        self.current_targets[env_ids] = dof_pos[:]

        root_vel = torch.zeros((num_resets, 6), device=self._device)

        # apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._a1s.set_joint_positions(dof_pos, indices)
        self._a1s.set_joint_velocities(dof_vel, indices)

        self._a1s.set_world_poses(self.initial_root_pos[env_ids].clone(), self.initial_root_rot[env_ids].clone(), indices)
        self._a1s.set_velocities(root_vel, indices)


        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        

    def post_reset(self):
        self.initial_root_pos, self.initial_root_rot = self._a1s.get_world_poses()
        self.current_targets = self.default_dof_pos.clone()

        dof_limits = self._a1s.get_dof_limits()
        self.a1_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.a1_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        
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
        
        dof_limits = self._a1s.get_dof_limits()
        self.dof_limits_lower = dof_limits[0, :, 0].to(self._device)
        self.dof_limits_upper = dof_limits[0, :, 1].to(self._device)
        
        print(self.dof_limits_upper,"a")

        # randomize all envs
        indices = torch.arange(self._a1s.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
        
    def calculate_metrics(self) -> None:
        torso_position, torso_rotation = self._a1s.get_world_poses(clone=False)
        root_velocities = self._a1s.get_velocities(clone=False)
        dof_pos = self._a1s.get_joint_positions(clone=False)
        dof_vel = self._a1s.get_joint_velocities(clone=False)

        velocity = root_velocities[:, 0:3]
        ang_velocity = root_velocities[:, 3:6]
        
        base_lin_vel = quat_rotate_inverse(torso_rotation, velocity)
        base_ang_vel = quat_rotate_inverse(torso_rotation, ang_velocity)

        # velocity tracking reward
        
        dx = abs(self.torso_x - self.init_torso_x)
        
        dy = abs(self.torso_y - self.init_torso_y)
        
        dz = abs(0.29 - self.torso_z)
        
        rew_torsodistance = 1 - torch.tanh(10.0*(dx + dy + dz)/3)
        
        x_range = (0.06 > dx)
        y_range = (0.02 > dy)
        z_range = (0.01 > dz)
        
        rew_center = x_range & y_range & z_range

        # 前脚を離したい
        # self.foot_contact_z_FR_outputは，ついてたら2，ついてなかったら-1を格納
        # print(self.foot_contact_z_FR_output)
        rew_foot_contact = -(self.foot_contact_z_FR_output \
                            + self.foot_contact_z_FL_output)
        
        # 後脚を接地させたい
        # rew_foot_contact += (self.foot_contact_z_RR_output \
        #                     + self.foot_contact_z_RL_output) * 0.1
                            
        # print(rew_foot_contact)

        # total_reward = rew_torsodistance*3 + rew_center*10 + rew_foot_contact
        # fr = self.foot_contact_z_FR_output.to('cpu').detach().numpy().copy()
        # fl = self.foot_contact_z_FL_output.to('cpu').detach().numpy().copy()
        # rr = self.foot_contact_z_RR_output.to('cpu').detach().numpy().copy()
        # rl = self.foot_contact_z_RL_output.to('cpu').detach().numpy().copy()

        # やりたいこと
        # 胴体のz座標に応じて報酬
        # 0.45に近いほど報酬
        # print(fr[0], fl[0], rr[0], rl[0])
        # print(0.45 - self.torso_z)
        # rew_z = 1.0 - torch.abs(0.45 - self.torso_z)
        # print(rew_z)
        rew_alive = torch.ones_like(self.torso_x) * 2.0
        # print(rew_alive)
        z_low = 0.45
        z_up = 0.5
        isZgood = (self.torso_z.mean() > z_low) and (self.torso_z.mean() < z_up)
        # rew_z = isZgood * 10
        rew_z = self.torso_z*0 + isZgood*10
        # print(rew_z)
        
        # print(rew_z)
        
        # print(torso_rotation.to('cpu').detach().numpy().copy())
        # print(quaternion.as_euler_angles(torso_rotation.to('cpu').detach().numpy().copy()))
        a = quaternion.as_quat_array(torso_rotation.to('cpu').detach().numpy().copy())
        b = quaternion.as_euler_angles(a)
        c = np.rad2deg(b)
        d = c.T
        e_1 = d[:1][0]
        e_2 = d[1:2][0]
        f_1 = torch.from_numpy(e_1.astype(np.float32)).clone().to(self.device)
        f_2 = torch.from_numpy(e_2.astype(np.float32)).clone().to(self.device)
        # print(f_1+90)
        # print(f_2) 
        roll_element = -(torch.abs(f_1 + 90)**2)
        pitch_element = -(torch.abs(f_2 - 90)**2)
        rew_tilt = (- torch.abs(f_1 + 90) - torch.abs(f_2 - 90))*0.001
        # print(rew_tilt)
        # print(rew_z)
        
        # total_reward = rew_z\
        #                 + rew_tilt
        
        # print(root_velocities)
        # rew_vel = .0
    
        # total_reward = rew_z
        total_reward = rew_tilt
        # total_reward = rew_z + rew_foot_contact        
        # total_reward = rew_tilt + rew_foot_contact + rew_z

        # print(total_reward)
        total_reward = torch.clip(total_reward, 0.0, None)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = dof_vel[:]

        self.fallen_over = self._a1s.is_base_below_threshold(threshold=0.4, ground_heights=0.0)
        total_reward[torch.nonzero(self.fallen_over)] = -1
        self.rew_buf[:] = total_reward.detach()
        
        # print(total_reward)


    def is_done(self) -> None:
        # reset agents
        
        time_out = self.progress_buf >= self.max_episode_length - 1
        self.reset_buf[:] = time_out | self.fallen_over
    

    
