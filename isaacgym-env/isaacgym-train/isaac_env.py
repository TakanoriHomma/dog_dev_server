import math
import time
import numpy as np
from math import sqrt

import gym
from isaacgym import gymapi
from isaacgym import gymtorch


class Isaac_Env:
    def __init__(self, headless="True"):
        np.random.seed(seed=int(time.time()))
        self.assets_path = "assets/"
        self.headless = headless
        self.config_sim_params()
        self.config_env_params()
        self.config_task()

    def config_env_params(self):
        self.env_num = 1

    def config_task(self):
        max = 100.0
        self.observation_space = np.array([0] * (12 + 7))
        self.action_space = gym.spaces.Box(
            low=np.array([-max] * 12),
            high=np.array([max] * 12),
            dtype=np.float64,
        )
        print("observation space: ", self.observation_space.shape[0])
        print("action space: ", self.action_space.shape[0])

        self.initial_robot_position = gymapi.Vec3(0.0, 0.0, 0.5)

    def config_sim_params(self):
        # common parameter
        self.sim_params = gymapi.SimParams()
        self.sim_params.dt = 0.10  # sim step
        self.sim_params.substeps = 5  #  higher value = stable simulation, 3 ~ 5
        self.sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        # PhysX-specific parameters
        self.sim_params.physx.use_gpu = True  # PhysX GPU
        self.sim_params.use_gpu_pipeline = False
        self.sim_params.physx.num_threads = 32  # number of CPU threads
        self.sim_params.physx.solver_type = 1  # 0 : Iterative sequential impulse solve, 1 : Non-linear iterative solver (more robust, but slightly more expensive)
        self.sim_params.physx.num_position_iterations = (
            4  # PhysX solver position iterations count
        )
        self.sim_params.physx.num_velocity_iterations = (
            1  # PhysX solver velocity iterations count
        )
        self.sim_params.physx.rest_offset = 0.0
        self.sim_params.physx.contact_offset = (
            0.01  # Consider contacts less than this value
        )
        #self.sim_params.physx.friction_offset_threshold = 0.001
        #self.sim_params.physx.friction_correlation_distance = 0.0005

    # ------------------------------------------------------------
    # gym-like functions
    # ------------------------------------------------------------

    def step(self, action):
        self.move_robot(action)
        self.render(render_collision=False)
        next_observation = self.get_observation()
        reward = self.get_reward(next_observation)
        done = self.check_termination(next_observation)
        return next_observation, reward, done, None

    def get_observation(self):
        # update states
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)  ## added
        
        dof_pos = self.dof_pos_
        dof_vel = self.dof_vel_

        observation = {
            "leg_position": np.array(dof_pos),
            "leg_velocity": np.array(dof_vel),
            "body_pose": np.array(self.dog_body_pose),
            "contact": np.array(self.dog_leg_contact),  ##
        }
        return observation

    def random_sample(self):
        max = 1
        random_action = np.random.uniform(low=-max, high=max, size=(12,))
        return random_action

    def get_reward(self, observation):
        ideal_pos = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]
        current_pos = observation["body_pose"][0:7]
        diff_pos = ideal_pos - current_pos
        norm_pos = np.linalg.norm(diff_pos)
        reward = np.exp(1 / (norm_pos + 0.1))
        return reward

    def check_termination(self, observation):
        if (
            observation["body_pose"][2] < 0.1
            or abs(observation["body_pose"][0]) >= 1.0
            or abs(observation["body_pose"][1]) >= 1.0
        ):
            return True
        else:
            return False

    def render(self, render_collision):
        if not self.headless:
            # update the viewer
            self.gym.draw_viewer(
                self.viewer, self.sim, render_collision=render_collision
            )

    def reset(self):
        # Initialize all
        self.gym.set_sim_rigid_body_states(
            self.sim, self.initial_state, gymapi.STATE_ALL
        )
        # Return observation
        state = self.get_observation()
        return state

    def close(self):
        if not self.headless:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        self.gym.destroy_env(self.env)

    # ------------------------------------------------------------
    # environment
    # ------------------------------------------------------------

    def create_env(self):

        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, self.sim_params)

        # set up the env grid
        num_per_row = int(sqrt(self.env_num))
        env_spacing = 1
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

        self.create_plane()
        self.create_viewer()
        self.create_a1()
        # self.create_wall()

        # observation DOF
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        num_dofs = int(self.gym.get_sim_dof_count(self.sim) / self.env_num)
        self.dof_pos_ = dof_state.view(self.env_num, num_dofs, 2)[0, :, 0]
        self.dof_vel_ = dof_state.view(self.env_num, num_dofs, 2)[0, :, 1]

        rigid_body_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        rigid_body_states = gymtorch.wrap_tensor(rigid_body_states)
        self.dog_body_pose = rigid_body_states[0][0:7]

######### observation contact   added from here　
        force_sensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        force_sensor = gymtorch.wrap_tensor(force_sensor)
        self.dog_leg_contact = force_sensor
######### added here       

        # create a local copy of initial state for initialization
        self.initial_state = np.copy(
            self.gym.get_sim_rigid_body_states(self.sim, gymapi.STATE_ALL)
        )

    def create_plane(self):
        self.plane_params = gymapi.PlaneParams()
        self.plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.plane_params.distance = 0  # distance of the plane from the origin
        self.plane_params.static_friction = 1.0
        self.plane_params.dynamic_friction = 1.0
        self.plane_params.restitution = 0.0
        self.gym.add_ground(self.sim, self.plane_params)

    def create_viewer(self):
        cam_props = gymapi.CameraProperties()
        cam_props.height = 1000
        cam_props.width = 1500
        cam_props.use_collision_geometry = False
        cam_pos = gymapi.Vec3(-1.5, +1.5, 1.5)
        cam_target = gymapi.Vec3(+1.5, -1.5, 0.0)
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, cam_props)
            self.gym.viewer_camera_look_at(
                self.viewer,
                None,
                cam_pos,
                cam_target,
            )

    def move_robot(self, action):
        action = np.array(action).astype("f")
        # position control
        dof_states = self.gym.get_actor_dof_states(
            self.env, self.a1_handle, gymapi.STATE_NONE
        )
        #dof_states["pos"] = dof_states["pos"] + action
        dof_states["pos"] = action
        # dof_states["vel"] = [0.0] * 12
        self.gym.set_actor_dof_states(
            self.env, self.a1_handle, dof_states, gymapi.STATE_POS
        )
        # step the physics
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)  # update graphics

    def create_a1(self):
        asset_file = "urdf/a1_description/robots/a1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Robot position fix
        asset_options.flip_visual_attachments = True
        asset_options.use_mesh_materials = True
        asset = self.gym.load_asset(
            self.sim, self.assets_path, asset_file, asset_options
        )
        # collision
        collision_group = 0
        collision_filter = 0
        # pose
        pose = gymapi.Transform()
        pose.p = self.initial_robot_position
        pose.r = gymapi.Quat(0, 0, 0, 1)
        # handler
        self.a1_handle = self.gym.create_actor(
            self.env, asset, pose, "a1", collision_group, collision_filter
        )
        # DOF
        self.a1_handle_fr_hip = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FR_hip_joint"
        )
        self.a1_handle_fr_thigh = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FR_thigh_joint"
        )
        self.a1_handle_fr_calf = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FR_calf_joint"
        )
        self.a1_handle_fl_hip = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FL_hip_joint"
        )
        self.a1_handle_fl_thigh = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FL_thigh_joint"
        )
        self.a1_handle_fl_calf = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "FL_calf_joint"
        )
        self.a1_handle_rr_hip = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RR_hip_joint"
        )
        self.a1_handle_rr_thigh = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RR_thigh_joint"
        )
        self.a1_handle_rr_calf = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RR_calf_joint"
        )
        self.a1_handle_rl_hip = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RL_hip_joint"
        )
        self.a1_handle_rl_thigh = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RL_thigh_joint"
        )
        self.a1_handle_rl_calf = self.gym.find_actor_dof_handle(
            self.env, self.a1_handle, "RL_calf_joint"
        )
        # Configure DOF properties
        props = self.gym.get_actor_dof_properties(self.env, self.a1_handle)
        props["driveMode"].fill(gymapi.DOF_MODE_POS)
        props["stiffness"].fill(85.0)
        props["damping"].fill(2.0)
        self.gym.set_actor_dof_properties(self.env, self.a1_handle, props)

    def create_wall(self):
        # load asset
        asset_file = "urdf/wall_description/box.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset = self.gym.load_asset(
            self.sim, self.assets_path, asset_file, asset_options
        )
        # pose
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(2.0, 0.0, 0.5)
        # collision
        collision_group = 0
        collision_filter = 0  # 0 with collide, 1 without collide
        # handler
        self.rock_handle = self.gym.create_actor(
            self.env, asset, pose, None, collision_group, collision_filter
        )