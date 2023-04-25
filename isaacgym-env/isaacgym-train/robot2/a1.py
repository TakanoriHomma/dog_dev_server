from typing import Optional
import numpy as np
import torch
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import numpy as np
import torch

from pxr import PhysxSchema

class Unitree(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "unitree_quadruped"
        physics_dt: Optional[float] = 0.0025,
        usd_path: Optional[str] = None,
        position: Optional[numpy.ndarray] = None,
        orientation: Optional[numpy.ndarray] = None,
        model: Optional[str] = 'A1',
        way_points: Optional[numpy.ndarray] = None
    ) -> None:
        """[summary]
        """        
        
        self._usd_path = usd_path
        self._name = name
    
        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find nucleus server with /Isaac folder")
            self._usd_path = assets_root_path + "/Isaac/Robots/ANYbotics/anymal_instanceable.usd"
        add_reference_to_stage(self._usd_path, prim_path)
        
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._dof_names = ["FR_hip_joint",
                           "FR_thigh_joint",
                           "FR_calf_joint",
                           "FL_hip_joint",
                           "FL_thigh_joint",
                           "FL_calf_joint",
                           "RR_hip_joint",
                           "RR_thigh_joint",
                           "RR_calf_joint",
                           "RL_hip_joint",
                           "RL_thigh_joint",
                           "RL_calf_joint"]

    @property
    def dof_names(self):
        return self._dof_names

    def set_a1_properties(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64/np.pi*180)

    def prepare_contacts(self, stage, prim):
        for link_prim in prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI): 
                if "_HIP" not in str(link_prim.GetPrimPath()):
                    rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                    rb.CreateSleepThresholdAttr().Set(0)
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)