import sapien.core as sapien
import numpy as np
import pdb
from .planner import MplibPlanner
import numpy as np
import toppra as ta
import math
import yaml
import os
import transforms3d as t3d
from copy import deepcopy
import sapien.core as sapien
import envs._GLOBAL_CONFIGS as CONFIGS
from envs.utils import transforms
from .planner import CuroboPlanner
import torch.multiprocessing as mp


class Robot:

    def __init__(self, scene, need_topp=False, **kwargs):
        super().__init__()
        ta.setup_logging("CRITICAL")  # hide logging
        self._init_robot_(scene, need_topp, **kwargs)

    def _init_robot_(self, scene, need_topp=False, **kwargs):
        # self.dual_arm = dual_arm_tag
        # self.plan_success = True
        self.scene = scene

        self.left_js = None
        self.right_js = None

        left_embodiment_args = kwargs["left_embodiment_config"]
        right_embodiment_args = kwargs["right_embodiment_config"]
        left_robot_file = kwargs["left_robot_file"]
        right_robot_file = kwargs["right_robot_file"]

        self.need_topp = need_topp

        self.left_urdf_path = os.path.join(left_robot_file, left_embodiment_args["urdf_path"])
        self.left_srdf_path = left_embodiment_args.get("srdf_path", None)
        self.left_curobo_yml_path = os.path.join(left_robot_file, "curobo.yml")
        if self.left_srdf_path is not None:
            self.left_srdf_path = os.path.join(left_robot_file, self.left_srdf_path)
        self.left_joint_stiffness = left_embodiment_args.get("joint_stiffness", 1000)
        self.left_joint_damping = left_embodiment_args.get("joint_damping", 200)
        self.left_gripper_stiffness = left_embodiment_args.get("gripper_stiffness", 1000)
        self.left_gripper_damping = left_embodiment_args.get("gripper_damping", 200)
        self.left_gripper_velocity_gain = left_embodiment_args.get("gripper_velocity_gain", 0.05)
        self.left_passive_gripper_mimic_stiffness = left_embodiment_args.get("passive_gripper_mimic_stiffness", 0)
        self.left_passive_gripper_mimic_damping = left_embodiment_args.get("passive_gripper_mimic_damping", 0)
        self.left_drive_passive_gripper_mimic_targets = left_embodiment_args.get(
            "drive_passive_gripper_mimic_targets", False)
        self.left_drive_all_passive_gripper_mimic_targets = left_embodiment_args.get(
            "drive_all_passive_gripper_mimic_targets", False)
        self.left_sync_passive_gripper_mimic_qpos = left_embodiment_args.get(
            "sync_passive_gripper_mimic_qpos", False)
        self.left_passive_gripper_mimic_target_stiffness = left_embodiment_args.get(
            "passive_gripper_mimic_target_stiffness", self.left_passive_gripper_mimic_stiffness)
        self.left_passive_gripper_mimic_target_damping = left_embodiment_args.get(
            "passive_gripper_mimic_target_damping", self.left_passive_gripper_mimic_damping)
        self.left_gripper_joint_friction = left_embodiment_args.get("gripper_joint_friction", None)
        self.left_gripper_static_friction = left_embodiment_args.get("gripper_static_friction", None)
        self.left_gripper_dynamic_friction = left_embodiment_args.get("gripper_dynamic_friction", None)
        self.left_gripper_restitution = left_embodiment_args.get("gripper_restitution", 0)
        self.left_planner_type = left_embodiment_args.get("planner", "mplib_RRT")
        self.disable_self_collision = left_embodiment_args.get("disable_self_collision", False)
        self.left_move_group = left_embodiment_args["move_group"][0]
        self.left_ee_name = left_embodiment_args["ee_joints"][0]
        self.left_arm_joints_name = left_embodiment_args["arm_joints_name"][0]
        self.left_gripper_name = left_embodiment_args["gripper_name"][0]
        self.left_gripper_bias = left_embodiment_args["gripper_bias"]
        self.left_gripper_scale = left_embodiment_args["gripper_scale"]
        self.left_homestate = left_embodiment_args.get("homestate", [[0] * len(self.left_arm_joints_name)])[0]
        self.left_fix_gripper_name = left_embodiment_args.get("fix_gripper_name", [])
        self.fixed_joint_targets = left_embodiment_args.get("fixed_joint_targets", {})
        self.left_delta_matrix = np.array(left_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.left_inv_delta_matrix = np.linalg.inv(self.left_delta_matrix)
        self.left_global_trans_matrix = np.array(
            left_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = left_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])[0]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.left_entity_origion_pose = deepcopy(_entity_origion_pose)

        self.right_urdf_path = os.path.join(right_robot_file, right_embodiment_args["urdf_path"])
        self.right_srdf_path = right_embodiment_args.get("srdf_path", None)
        if self.right_srdf_path is not None:
            self.right_srdf_path = os.path.join(right_robot_file, self.right_srdf_path)
        self.right_curobo_yml_path = os.path.join(right_robot_file, "curobo.yml")
        self.right_joint_stiffness = right_embodiment_args.get("joint_stiffness", 1000)
        self.right_joint_damping = right_embodiment_args.get("joint_damping", 200)
        self.right_gripper_stiffness = right_embodiment_args.get("gripper_stiffness", 1000)
        self.right_gripper_damping = right_embodiment_args.get("gripper_damping", 200)
        self.right_gripper_velocity_gain = right_embodiment_args.get("gripper_velocity_gain", 0.05)
        self.right_passive_gripper_mimic_stiffness = right_embodiment_args.get("passive_gripper_mimic_stiffness", 0)
        self.right_passive_gripper_mimic_damping = right_embodiment_args.get("passive_gripper_mimic_damping", 0)
        self.right_drive_passive_gripper_mimic_targets = right_embodiment_args.get(
            "drive_passive_gripper_mimic_targets", False)
        self.right_drive_all_passive_gripper_mimic_targets = right_embodiment_args.get(
            "drive_all_passive_gripper_mimic_targets", False)
        self.right_sync_passive_gripper_mimic_qpos = right_embodiment_args.get(
            "sync_passive_gripper_mimic_qpos", False)
        self.right_passive_gripper_mimic_target_stiffness = right_embodiment_args.get(
            "passive_gripper_mimic_target_stiffness", self.right_passive_gripper_mimic_stiffness)
        self.right_passive_gripper_mimic_target_damping = right_embodiment_args.get(
            "passive_gripper_mimic_target_damping", self.right_passive_gripper_mimic_damping)
        self.right_gripper_joint_friction = right_embodiment_args.get("gripper_joint_friction", None)
        self.right_gripper_static_friction = right_embodiment_args.get("gripper_static_friction", None)
        self.right_gripper_dynamic_friction = right_embodiment_args.get("gripper_dynamic_friction", None)
        self.right_gripper_restitution = right_embodiment_args.get("gripper_restitution", 0)
        self.right_planner_type = right_embodiment_args.get("planner", "mplib_RRT")
        self.right_move_group = right_embodiment_args["move_group"][1]
        self.right_ee_name = right_embodiment_args["ee_joints"][1]
        self.right_arm_joints_name = right_embodiment_args["arm_joints_name"][1]
        self.right_gripper_name = right_embodiment_args["gripper_name"][1]
        self.right_gripper_bias = right_embodiment_args["gripper_bias"]
        self.right_gripper_scale = right_embodiment_args["gripper_scale"]
        self.right_homestate = right_embodiment_args.get("homestate", [[1] * len(self.right_arm_joints_name)])[1]
        self.right_fix_gripper_name = right_embodiment_args.get("fix_gripper_name", [])
        self.right_delta_matrix = np.array(right_embodiment_args.get("delta_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.right_inv_delta_matrix = np.linalg.inv(self.right_delta_matrix)
        self.right_global_trans_matrix = np.array(
            right_embodiment_args.get("global_trans_matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

        _entity_origion_pose = right_embodiment_args.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]])
        _entity_origion_pose = _entity_origion_pose[0 if len(_entity_origion_pose) == 1 else 1]
        _entity_origion_pose = sapien.Pose(_entity_origion_pose[:3], _entity_origion_pose[-4:])
        self.right_entity_origion_pose = deepcopy(_entity_origion_pose)
        self.is_dual_arm = kwargs["dual_arm_embodied"]

        self.left_rotate_lim = left_embodiment_args.get("rotate_lim", [0, 0])
        self.right_rotate_lim = right_embodiment_args.get("rotate_lim", [0, 0])

        self.left_perfect_direction = left_embodiment_args.get("grasp_perfect_direction",
                                                               ["front_right", "front_left"])[0]
        self.right_perfect_direction = right_embodiment_args.get("grasp_perfect_direction",
                                                                 ["front_right", "front_left"])[1]

        if self.is_dual_arm:
            loader: sapien.URDFLoader = scene.create_urdf_loader()
            loader.fix_root_link = True
            self._entity = loader.load(self.left_urdf_path)
            self.left_entity = self._entity
            self.right_entity = self._entity
        else:
            arms_dis = kwargs["embodiment_dis"]
            self.left_entity_origion_pose.p += [-arms_dis / 2, 0, 0]
            self.right_entity_origion_pose.p += [arms_dis / 2, 0, 0]
            left_loader: sapien.URDFLoader = scene.create_urdf_loader()
            left_loader.fix_root_link = True
            right_loader: sapien.URDFLoader = scene.create_urdf_loader()
            right_loader.fix_root_link = True
            self.left_entity = left_loader.load(self.left_urdf_path)
            self.right_entity = right_loader.load(self.right_urdf_path)

        self.left_entity.set_root_pose(self.left_entity_origion_pose)
        self.right_entity.set_root_pose(self.right_entity_origion_pose)
        if self.disable_self_collision:
            self._disable_robot_self_collision()

    def reset(self, scene, need_topp=False, **kwargs):
        self._init_robot_(scene, need_topp, **kwargs)

        if self.communication_flag:
            if hasattr(self, "left_conn") and self.left_conn:
                self.left_conn.send({"cmd": "reset"})
                _ = self.left_conn.recv()
            if hasattr(self, "right_conn") and self.right_conn:
                self.right_conn.send({"cmd": "reset"})
                _ = self.right_conn.recv()
        else:
            if not isinstance(self.left_planner, CuroboPlanner) or not isinstance(self.right_planner, CuroboPlanner):
                self.set_planner(scene=scene)

        self.init_joints()

    def _disable_robot_self_collision(self):
        ignore_pairs = self._load_self_collision_ignore_pairs()
        if ignore_pairs:
            self._disable_robot_self_collision_pairs(ignore_pairs)
            return

        seen_entities = set()
        for entity in [self.left_entity, self.right_entity]:
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            for link in entity.get_links():
                for shape in link.get_collision_shapes():
                    groups = shape.get_collision_groups()
                    shape.set_collision_groups([groups[0], groups[1], 1, groups[3]])

    def _load_self_collision_ignore_pairs(self):
        def planner_yml_paths():
            paths = [
                os.path.join(CONFIGS.ROOT_PATH, self.left_curobo_yml_path),
                os.path.join(CONFIGS.ROOT_PATH, self.right_curobo_yml_path),
            ]
            if self.is_dual_arm:
                paths.extend([
                    paths[0].replace("curobo.yml", "curobo_left.yml"),
                    paths[1].replace("curobo.yml", "curobo_right.yml"),
                ])
            return paths

        pairs = set()
        for path in planner_yml_paths():
            if not os.path.exists(path):
                continue
            with open(path, "r") as f:
                data = yaml.safe_load(f) or {}
            ignore = (
                data.get("robot_cfg", {})
                .get("kinematics", {})
                .get("self_collision_ignore", {})
            )
            if not isinstance(ignore, dict):
                continue
            for link_name, ignored_links in ignore.items():
                if not isinstance(ignored_links, list):
                    continue
                for ignored_link in ignored_links:
                    pairs.add(tuple(sorted((link_name, ignored_link))))
        return sorted(pairs)

    def _disable_robot_self_collision_pairs(self, ignore_pairs):
        seen_entities = set()
        for entity in [self.left_entity, self.right_entity]:
            if entity in seen_entities:
                continue
            seen_entities.add(entity)
            links_by_name = {link.get_name(): link for link in entity.get_links()}
            link_masks = {}
            group_count = 0
            for link_name, ignored_link in ignore_pairs:
                if link_name not in links_by_name or ignored_link not in links_by_name:
                    continue
                group_count += 1
                if group_count == 32:
                    raise RuntimeError("Too many self-collision ignore pairs for SAPIEN collision groups")
                group = 1 << (group_count - 1)
                link_masks[link_name] = link_masks.get(link_name, 0) | group
                link_masks[ignored_link] = link_masks.get(ignored_link, 0) | group
            for link_name, group in link_masks.items():
                for shape in links_by_name[link_name].get_collision_shapes():
                    groups = shape.get_collision_groups()
                    shape.set_collision_groups([groups[0], groups[1], groups[2] | group, groups[3]])

    def get_grasp_perfect_direction(self, arm_tag):
        if arm_tag == "left":
            return self.left_perfect_direction
        elif arm_tag == "right":
            return self.right_perfect_direction

    def create_target_pose_list(self, origin_pose, center_pose, arm_tag=None):
        res_lst = []
        rotate_lim = (self.left_rotate_lim if arm_tag == "left" else self.right_rotate_lim)
        rotate_step = (rotate_lim[1] - rotate_lim[0]) / CONFIGS.ROTATE_NUM
        for i in range(CONFIGS.ROTATE_NUM):
            now_pose = transforms.rotate_along_axis(
                origin_pose,
                center_pose,
                [0, 1, 0],
                rotate_step * i + rotate_lim[0],
                axis_type="target",
                towards=[0, -1, 0],
            )
            res_lst.append(now_pose)
        return res_lst

    def get_constraint_pose(self, ori_vec: list, arm_tag=None):
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        return ori_vec[:3] + (ori_vec[-3:] @ np.linalg.inv(inv_delta_matrix)).tolist()

    def init_joints(self):
        if self.left_entity is None or self.right_entity is None:
            raise ValueError("Robote entity is None")

        self.left_active_joints = self.left_entity.get_active_joints()
        self.right_active_joints = self.right_entity.get_active_joints()

        self.left_ee = self.left_entity.find_joint_by_name(self.left_ee_name)
        self.right_ee = self.right_entity.find_joint_by_name(self.right_ee_name)

        self.left_gripper_val = 0.0
        self.right_gripper_val = 0.0

        self.left_arm_joints = [self.left_entity.find_joint_by_name(i) for i in self.left_arm_joints_name]
        self.right_arm_joints = [self.right_entity.find_joint_by_name(i) for i in self.right_arm_joints_name]

        def get_gripper_joints(find, gripper_name: str):
            if gripper_name is None:
                return []
            gripper = []
            for g in gripper_name.get("joints", []):
                gripper.append((find(g[0]), g[1], g[2]))
            if not gripper and gripper_name.get("base") is not None:
                gripper.append((find(gripper_name["base"]), 1.0, 0.0))
            if gripper_name.get("drive_mimic", True):
                for g in gripper_name.get("mimic", []):
                    gripper.append((find(g[0]), g[1], g[2]))
            return gripper

        def get_passive_mimic_joints(find, gripper_name: str):
            if gripper_name is None or gripper_name.get("drive_mimic", True):
                return []
            return [find(g[0]) for g in gripper_name.get("mimic", [])]

        self.left_gripper = get_gripper_joints(self.left_entity.find_joint_by_name, self.left_gripper_name)
        self.right_gripper = get_gripper_joints(self.right_entity.find_joint_by_name, self.right_gripper_name)
        self.left_passive_gripper_mimic = get_passive_mimic_joints(
            self.left_entity.find_joint_by_name, self.left_gripper_name)
        self.right_passive_gripper_mimic = get_passive_mimic_joints(
            self.right_entity.find_joint_by_name, self.right_gripper_name)
        self.gripper_name = deepcopy(self.left_fix_gripper_name) + deepcopy(self.right_fix_gripper_name)

        for g in self.left_gripper:
            self.gripper_name.append(g[0].child_link.get_name())
        for g in self.right_gripper:
            self.gripper_name.append(g[0].child_link.get_name())
        for joint in self.left_passive_gripper_mimic:
            self.gripper_name.append(joint.child_link.get_name())
        for joint in self.right_passive_gripper_mimic:
            self.gripper_name.append(joint.child_link.get_name())

        # camera link id
        self.left_camera = self.left_entity.find_link_by_name("left_camera")
        if self.left_camera is None:
            self.left_camera = self.left_entity.find_link_by_name("camera")
            if self.left_camera is None:
                print("No left camera link")
                self.left_camera = self.left_entity.get_links()[0]

        self.right_camera = self.right_entity.find_link_by_name("right_camera")
        if self.right_camera is None:
            self.right_camera = self.right_entity.find_link_by_name("camera")
            if self.right_camera is None:
                print("No right camera link")
                self.right_camera = self.right_entity.get_links()[0]

        left_gripper_joint_set = {g[0] for g in self.left_gripper} | set(self.left_passive_gripper_mimic)
        right_gripper_joint_set = {g[0] for g in self.right_gripper} | set(self.right_passive_gripper_mimic)
        for i, joint in enumerate(self.left_active_joints):
            if joint not in left_gripper_joint_set:
                joint.set_drive_property(stiffness=self.left_joint_stiffness, damping=self.left_joint_damping)
        for i, joint in enumerate(self.right_active_joints):
            if joint not in right_gripper_joint_set:
                joint.set_drive_property(
                    stiffness=self.right_joint_stiffness,
                    damping=self.right_joint_damping,
                )

        for joint in self.left_gripper:
            joint[0].set_drive_property(stiffness=self.left_gripper_stiffness, damping=self.left_gripper_damping)
        for joint in self.right_gripper:
            joint[0].set_drive_property(
                stiffness=self.right_gripper_stiffness,
                damping=self.right_gripper_damping,
            )
        for joint in self.left_passive_gripper_mimic + self.right_passive_gripper_mimic:
            joint.set_drive_property(stiffness=0, damping=0)

        self._apply_gripper_joint_properties()
        self._apply_gripper_physical_materials()

        self._initialize_joint_positions()

    def _apply_gripper_joint_properties(self):
        def apply(gripper, passive_joints, friction):
            joints = [joint[0] for joint in gripper] + list(passive_joints)
            for joint in joints:
                if friction is not None:
                    joint.set_friction(float(friction))

        apply(
            self.left_gripper,
            self.left_passive_gripper_mimic,
            self.left_gripper_joint_friction,
        )
        apply(
            self.right_gripper,
            self.right_passive_gripper_mimic,
            self.right_gripper_joint_friction,
        )

    def _apply_gripper_physical_materials(self):
        def apply(entity, gripper, passive_joints, static_friction, dynamic_friction, restitution):
            if static_friction is None and dynamic_friction is None:
                return
            static_friction = float(static_friction if static_friction is not None else dynamic_friction)
            dynamic_friction = float(dynamic_friction if dynamic_friction is not None else static_friction)
            material = self.scene.create_physical_material(static_friction, dynamic_friction, float(restitution))
            link_names = {joint[0].child_link.get_name() for joint in gripper}
            link_names.update(joint.child_link.get_name() for joint in passive_joints)
            for link in entity.get_links():
                if link.get_name() not in link_names:
                    continue
                for shape in link.get_collision_shapes():
                    shape.set_physical_material(material)

        apply(
            self.left_entity,
            self.left_gripper,
            self.left_passive_gripper_mimic,
            self.left_gripper_static_friction,
            self.left_gripper_dynamic_friction,
            self.left_gripper_restitution,
        )
        apply(
            self.right_entity,
            self.right_gripper,
            self.right_passive_gripper_mimic,
            self.right_gripper_static_friction,
            self.right_gripper_dynamic_friction,
            self.right_gripper_restitution,
        )

    def _initialize_joint_positions(self):
        def apply_initial_state(
            entity,
            active_joints,
            arm_joint_names,
            arm_home,
            gripper,
            gripper_config,
            gripper_scale,
        ):
            qpos = entity.get_qpos()
            if len(qpos) != len(active_joints):
                return

            target_by_name = {}
            target_by_name.update(self.fixed_joint_targets)
            target_by_name.update(dict(zip(arm_joint_names, arm_home)))
            if gripper:
                open_val = gripper_scale[1]
                for joint, multiplier, offset in gripper:
                    target_by_name[joint.get_name()] = open_val * multiplier + offset
                for joint_name, multiplier, offset in gripper_config.get("mimic", []):
                    target_by_name[joint_name] = open_val * multiplier + offset

            for idx, joint in enumerate(active_joints):
                name = joint.get_name()
                if name not in target_by_name:
                    continue
                target = float(target_by_name[name])
                qpos[idx] = target
                joint.set_drive_target(target)
                joint.set_drive_velocity_target(0.0)

            entity.set_qpos(qpos)
            entity.set_qvel(np.zeros_like(qpos))

        apply_initial_state(
            self.left_entity,
            self.left_active_joints,
            self.left_arm_joints_name,
            self.left_homestate,
            self.left_gripper,
            self.left_gripper_name,
            self.left_gripper_scale,
        )
        if self.right_entity is not self.left_entity:
            apply_initial_state(
                self.right_entity,
                self.right_active_joints,
                self.right_arm_joints_name,
                self.right_homestate,
                self.right_gripper,
                self.right_gripper_name,
                self.right_gripper_scale,
            )
        else:
            qpos = self.left_entity.get_qpos()
            target_by_name = dict(zip(self.right_arm_joints_name, self.right_homestate))
            for joint, multiplier, offset in self.right_gripper:
                target_by_name[joint.get_name()] = self.right_gripper_scale[1] * multiplier + offset
            for joint_name, multiplier, offset in self.right_gripper_name.get("mimic", []):
                target_by_name[joint_name] = self.right_gripper_scale[1] * multiplier + offset
            for idx, joint in enumerate(self.left_active_joints):
                name = joint.get_name()
                if name not in target_by_name:
                    continue
                target = float(target_by_name[name])
                qpos[idx] = target
                joint.set_drive_target(target)
                joint.set_drive_velocity_target(0.0)
            self.left_entity.set_qpos(qpos)
            self.left_entity.set_qvel(np.zeros_like(qpos))

    def move_to_homestate(self):
        for i, joint in enumerate(self.left_arm_joints):
            joint.set_drive_target(self.left_homestate[i])

        for i, joint in enumerate(self.right_arm_joints):
            joint.set_drive_target(self.right_homestate[i])

    def set_origin_endpose(self):
        self.left_original_pose = self.get_left_ee_pose()
        self.right_original_pose = self.get_right_ee_pose()

    def print_info(self):
        print(
            "active joints: ",
            [joint.get_name() for joint in self.left_active_joints + self.right_active_joints],
        )
        print(
            "all links: ",
            [link.get_name() for link in self.left_entity.get_links() + self.right_entity.get_links()],
        )
        print("left arm joints: ", [joint.get_name() for joint in self.left_arm_joints])
        print("right arm joints: ", [joint.get_name() for joint in self.right_arm_joints])
        print("left gripper: ", [joint[0].get_name() for joint in self.left_gripper])
        print("right gripper: ", [joint[0].get_name() for joint in self.right_gripper])
        print("left ee: ", self.left_ee.get_name())
        print("right ee: ", self.right_ee.get_name())

    def set_planner(self, scene=None):
        abs_left_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.left_curobo_yml_path)
        abs_right_curobo_yml_path = os.path.join(CONFIGS.ROOT_PATH, self.right_curobo_yml_path)

        self.communication_flag = (abs_left_curobo_yml_path != abs_right_curobo_yml_path)

        if self.is_dual_arm:
            abs_left_curobo_yml_path = abs_left_curobo_yml_path.replace("curobo.yml", "curobo_left.yml")
            abs_right_curobo_yml_path = abs_right_curobo_yml_path.replace("curobo.yml", "curobo_right.yml")

        if not self.communication_flag:
            self.left_planner = CuroboPlanner(self.left_entity_origion_pose,
                                              self.left_arm_joints_name,
                                              [joint.get_name() for joint in self.left_entity.get_active_joints()],
                                              yml_path=abs_left_curobo_yml_path)
            self.right_planner = CuroboPlanner(self.right_entity_origion_pose,
                                               self.right_arm_joints_name,
                                               [joint.get_name() for joint in self.right_entity.get_active_joints()],
                                               yml_path=abs_right_curobo_yml_path)
        else:
            self.left_conn, left_child_conn = mp.Pipe()
            self.right_conn, right_child_conn = mp.Pipe()

            left_args = {
                "origin_pose": self.left_entity_origion_pose,
                "joints_name": self.left_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.left_entity.get_active_joints()],
                "yml_path": abs_left_curobo_yml_path
            }

            right_args = {
                "origin_pose": self.right_entity_origion_pose,
                "joints_name": self.right_arm_joints_name,
                "all_joints": [joint.get_name() for joint in self.right_entity.get_active_joints()],
                "yml_path": abs_right_curobo_yml_path
            }

            self.left_proc = mp.Process(target=planner_process_worker, args=(left_child_conn, left_args))
            self.right_proc = mp.Process(target=planner_process_worker, args=(right_child_conn, right_args))

            self.left_proc.daemon = True
            self.right_proc.daemon = True

            self.left_proc.start()
            self.right_proc.start()

        if self.need_topp:
            self.left_mplib_planner = MplibPlanner(
                self.left_urdf_path,
                self.left_srdf_path,
                self.left_move_group,
                self.left_entity_origion_pose,
                self.left_entity,
                self.left_planner_type,
                scene,
            )
            self.right_mplib_planner = MplibPlanner(
                self.right_urdf_path,
                self.right_srdf_path,
                self.right_move_group,
                self.right_entity_origion_pose,
                self.right_entity,
                self.right_planner_type,
                scene,
            )

    def update_world_pcd(self, world_pcd):
        try:
            self.left_planner.update_point_cloud(world_pcd, resolution=0.02)
            self.right_planner.update_point_cloud(world_pcd, resolution=0.02)
        except:
            print("Update world pointcloud wrong!")

    def _trans_from_gripper_to_endlink(self, target_pose, arm_tag=None):
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        inv_delta_matrix = (self.left_inv_delta_matrix if arm_tag == "left" else self.right_inv_delta_matrix)
        target_pose_arr = np.array(target_pose)
        gripper_pose_pos, gripper_pose_quat = deepcopy(target_pose_arr[0:3]), deepcopy(target_pose_arr[-4:])
        gripper_pose_mat = t3d.quaternions.quat2mat(gripper_pose_quat)
        gripper_pose_pos += gripper_pose_mat @ np.array([0.12 - gripper_bias, 0, 0]).T
        gripper_pose_mat = gripper_pose_mat @ inv_delta_matrix
        gripper_pose_quat = t3d.quaternions.mat2quat(gripper_pose_mat)
        return sapien.Pose(gripper_pose_pos, gripper_pose_quat)

    def left_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.left_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_grippers(now_val, target_val)

    def right_plan_grippers(self, now_val, target_val):
        if self.communication_flag:
            self.right_conn.send({"cmd": "plan_grippers", "now_val": now_val, "target_val": target_val})
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_grippers(now_val, target_val)

    def left_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="left")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
            })
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="left",
            )

    def right_plan_multi_path(
        self,
        target_lst,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)
        target_lst_copy = deepcopy(target_lst)
        for i in range(len(target_lst_copy)):
            target_lst_copy[i] = self._trans_from_gripper_to_endlink(target_lst_copy[i], arm_tag="right")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_batch",
                "qpos": now_qpos,
                "target_pose_list": target_lst_copy,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
            })
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_batch(
                now_qpos,
                target_lst_copy,
                constraint_pose=constraint_pose,
                arms_tag="right",
            )

    def left_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="left")
        if last_qpos is None:
            now_qpos = self.left_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="left")

        if self.communication_flag:
            self.left_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "left",
            })
            return self.left_conn.recv()
        else:
            return self.left_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="left",
            )

    def right_plan_path(
        self,
        target_pose,
        constraint_pose=None,
        use_point_cloud=False,
        use_attach=False,
        last_qpos=None,
    ):
        if constraint_pose is not None:
            constraint_pose = self.get_constraint_pose(constraint_pose, arm_tag="right")
        if last_qpos is None:
            now_qpos = self.right_entity.get_qpos()
        else:
            now_qpos = deepcopy(last_qpos)

        trans_target_pose = self._trans_from_gripper_to_endlink(target_pose, arm_tag="right")

        if self.communication_flag:
            self.right_conn.send({
                "cmd": "plan_path",
                "qpos": now_qpos,
                "target_pose": trans_target_pose,
                "constraint_pose": constraint_pose,
                "arms_tag": "right",
            })
            return self.right_conn.recv()
        else:
            return self.right_planner.plan_path(
                now_qpos,
                trans_target_pose,
                constraint_pose=constraint_pose,
                arms_tag="right",
            )

    # The data of gripper has been normalized
    def get_left_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.left_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_jointState(self) -> list:
        jointState_list = []
        for joint in self.right_arm_joints:
            jointState_list.append(joint.get_drive_target()[0].astype(float))
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_arm_real_jointState(self) -> list:
        jointState_list = []
        left_joints_qpos = self.left_entity.get_qpos()
        left_active_joints = self.left_entity.get_active_joints()
        for joint in self.left_arm_joints:
            jointState_list.append(left_joints_qpos[left_active_joints.index(joint)])
        jointState_list.append(self.get_left_gripper_val())
        return jointState_list

    def get_right_arm_real_jointState(self) -> list:
        jointState_list = []
        right_joints_qpos = self.right_entity.get_qpos()
        right_active_joints = self.right_entity.get_active_joints()
        for joint in self.right_arm_joints:
            jointState_list.append(right_joints_qpos[right_active_joints.index(joint)])
        jointState_list.append(self.get_right_gripper_val())
        return jointState_list

    def get_left_gripper_val(self):
        if not self.left_gripper:
            return self.left_gripper_val
        if None in self.left_gripper:
            print("No gripper")
            return 0
        return self.left_gripper_val

    def get_right_gripper_val(self):
        if not self.right_gripper:
            return self.right_gripper_val
        if None in self.right_gripper:
            print("No gripper")
            return 0
        return self.right_gripper_val

    def is_left_gripper_open(self):
        return self.left_gripper_val > 0.8

    def is_right_gripper_open(self):
        return self.right_gripper_val > 0.8

    def is_left_gripper_open_half(self):
        return self.left_gripper_val > 0.45

    def is_right_gripper_open_half(self):
        return self.right_gripper_val > 0.45

    def is_left_gripper_close(self):
        return self.left_gripper_val < 0.2

    def is_right_gripper_close(self):
        return self.right_gripper_val < 0.2

    # get move group joint pose
    def get_left_ee_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=False)

    def get_right_ee_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=False)

    # get gripper centor pose
    def get_left_tcp_pose(self):
        return self._trans_endpose(arm_tag="left", is_endpose=True)

    def get_right_tcp_pose(self):
        return self._trans_endpose(arm_tag="right", is_endpose=True)

    def get_left_orig_endpose(self):
        pose = self.left_ee.global_pose
        global_trans_matrix = self.left_global_trans_matrix
        pose.p = pose.p - self.left_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.left_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def get_right_orig_endpose(self):
        pose = self.right_ee.global_pose
        global_trans_matrix = self.right_global_trans_matrix
        pose.p = pose.p - self.right_entity_origion_pose.p
        pose.p = t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ pose.p
        return (pose.p.tolist() + t3d.quaternions.mat2quat(
            t3d.quaternions.quat2mat(self.right_entity_origion_pose.q).T @ t3d.quaternions.quat2mat(pose.q)
            @ global_trans_matrix).tolist())

    def _trans_endpose(self, arm_tag=None, is_endpose=False):
        if arm_tag is None:
            print("No arm tag")
            return
        gripper_bias = (self.left_gripper_bias if arm_tag == "left" else self.right_gripper_bias)
        global_trans_matrix = (self.left_global_trans_matrix if arm_tag == "left" else self.right_global_trans_matrix)
        delta_matrix = (self.left_delta_matrix if arm_tag == "left" else self.right_delta_matrix)
        ee_pose = (self.left_ee.global_pose if arm_tag == "left" else self.right_ee.global_pose)
        endpose_arr = np.eye(4)
        endpose_arr[:3, :3] = (t3d.quaternions.quat2mat(ee_pose.q) @ global_trans_matrix @ delta_matrix)
        dis = gripper_bias
        if is_endpose == False:
            dis -= 0.12
        endpose_arr[:3, 3] = ee_pose.p + endpose_arr[:3, :3] @ np.array([dis, 0, 0]).T
        res = (endpose_arr[:3, 3].tolist() + t3d.quaternions.mat2quat(endpose_arr[:3, :3]).tolist())
        return res

    def _entity_qf(self, entity):
        qf = entity.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        entity.set_qf(qf)

    def set_arm_joints(self, target_position, target_velocity, arm_tag):
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)

        joint_lst = self.left_arm_joints if arm_tag == "left" else self.right_arm_joints
        for j in range(len(joint_lst)):
            joint = joint_lst[j]
            joint.set_drive_target(target_position[j])
            joint.set_drive_velocity_target(target_velocity[j])

    def get_normal_real_gripper_val(self):
        if not self.left_gripper or not self.right_gripper:
            return [self.left_gripper_val, self.right_gripper_val]
        normal_left_gripper_val = (self.left_gripper[0][0].get_drive_target()[0] - self.left_gripper_scale[0]) / (
            self.left_gripper_scale[1] - self.left_gripper_scale[0])
        normal_right_gripper_val = (self.right_gripper[0][0].get_drive_target()[0] - self.right_gripper_scale[0]) / (
            self.right_gripper_scale[1] - self.right_gripper_scale[0])
        normal_left_gripper_val = np.clip(normal_left_gripper_val, 0, 1)
        normal_right_gripper_val = np.clip(normal_right_gripper_val, 0, 1)
        return [normal_left_gripper_val, normal_right_gripper_val]

    def set_gripper(self, gripper_val, arm_tag, gripper_eps=0.1):  # gripper_val in [0,1]
        self._entity_qf(self.left_entity)
        self._entity_qf(self.right_entity)
        gripper_val = np.clip(gripper_val, 0, 1)

        if arm_tag == "left":
            entity = self.left_entity
            joints = self.left_gripper
            passive_joints = self.left_passive_gripper_mimic
            gripper_config = self.left_gripper_name
            drive_passive_mimic = self.left_drive_passive_gripper_mimic_targets
            drive_all_passive_mimic = self.left_drive_all_passive_gripper_mimic_targets
            passive_stiffness = self.left_passive_gripper_mimic_target_stiffness
            passive_damping = self.left_passive_gripper_mimic_target_damping
            gripper_velocity_gain = self.left_gripper_velocity_gain
            self.left_gripper_val = gripper_val
            gripper_scale = self.left_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[0]
        else:
            entity = self.right_entity
            joints = self.right_gripper
            passive_joints = self.right_passive_gripper_mimic
            gripper_config = self.right_gripper_name
            drive_passive_mimic = self.right_drive_passive_gripper_mimic_targets
            drive_all_passive_mimic = self.right_drive_all_passive_gripper_mimic_targets
            passive_stiffness = self.right_passive_gripper_mimic_target_stiffness
            passive_damping = self.right_passive_gripper_mimic_target_damping
            gripper_velocity_gain = self.right_gripper_velocity_gain
            self.right_gripper_val = gripper_val
            gripper_scale = self.right_gripper_scale
            real_gripper_val = self.get_normal_real_gripper_val()[1]

        if not joints:
            return

        if (gripper_val - real_gripper_val > gripper_eps
                and gripper_eps > 0) or (gripper_val - real_gripper_val < gripper_eps and gripper_eps < 0):
            gripper_val = real_gripper_val + gripper_eps  # TODO

        real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])

        for joint in joints:
            real_joint: sapien.physx.PhysxArticulationJoint = joint[0]
            drive_target = real_gripper_val * joint[1] + joint[2]
            drive_velocity_target = (
                np.clip(drive_target - real_joint.drive_target, -1.0, 1.0)
                * float(gripper_velocity_gain)
            )
            real_joint.set_drive_target(drive_target)
            real_joint.set_drive_velocity_target(drive_velocity_target)

        if drive_passive_mimic and passive_joints and not gripper_config.get("drive_mimic", True):
            mimic_targets = {
                joint_name: real_gripper_val * multiplier + offset
                for joint_name, multiplier, offset in gripper_config.get("mimic", [])
            }
            master_names = {joint[0].get_name() for joint in joints}
            for real_joint in passive_joints:
                parent_name = real_joint.parent_link.get_name()
                drive_target = mimic_targets.get(real_joint.get_name())
                if drive_target is None:
                    continue
                if (
                    not drive_all_passive_mimic
                    and any(parent_name == entity.find_joint_by_name(name).parent_link.get_name()
                            for name in master_names)
                ):
                    continue
                real_joint.set_drive_property(stiffness=float(passive_stiffness), damping=float(passive_damping))
                drive_velocity_target = (
                    np.clip(drive_target - real_joint.drive_target, -1.0, 1.0)
                    * float(gripper_velocity_gain)
                )
                real_joint.set_drive_target(float(drive_target))
                real_joint.set_drive_velocity_target(float(drive_velocity_target))
        self._sync_passive_gripper_mimic_qpos(arm_tag)

    def _is_cross_branch_passive_mimic_joint(self, entity, master_joint_names, passive_joint):
        parent_name = passive_joint.parent_link.get_name()
        return not any(
            parent_name == entity.find_joint_by_name(name).parent_link.get_name()
            for name in master_joint_names
        )

    def _sync_passive_gripper_mimic_qpos(self, arm_tag):
        if arm_tag == "left":
            entity = self.left_entity
            passive_joints = self.left_passive_gripper_mimic
            gripper_config = self.left_gripper_name
            gripper_joints = self.left_gripper
            sync_qpos = self.left_sync_passive_gripper_mimic_qpos
        else:
            entity = self.right_entity
            passive_joints = self.right_passive_gripper_mimic
            gripper_config = self.right_gripper_name
            gripper_joints = self.right_gripper
            sync_qpos = self.right_sync_passive_gripper_mimic_qpos
        if (
            not sync_qpos
            or not passive_joints
            or gripper_config.get("drive_mimic", True)
            or not gripper_joints
        ):
            return

        active_joints = entity.get_active_joints()
        qpos = entity.get_qpos()
        qvel = entity.get_qvel()
        base_name = gripper_config.get("base")
        master_joint = entity.find_joint_by_name(base_name) if base_name is not None else gripper_joints[0][0]
        if master_joint is None:
            return
        master_idx = active_joints.index(master_joint)
        master_qpos = qpos[master_idx]
        master_qvel = qvel[master_idx]
        mimic_targets = {
            joint_name: (float(master_qpos) * multiplier + offset, float(master_qvel) * multiplier)
            for joint_name, multiplier, offset in gripper_config.get("mimic", [])
        }
        changed = False
        for joint in passive_joints:
            target = mimic_targets.get(joint.get_name())
            if target is None:
                continue
            idx = active_joints.index(joint)
            qpos[idx] = target[0]
            qvel[idx] = target[1]
            joint.set_drive_target(float(target[0]))
            joint.set_drive_velocity_target(float(target[1]))
            changed = True
        if changed:
            entity.set_qpos(qpos)
            entity.set_qvel(qvel)

    def hold_passive_gripper_mimic(self, arm_tag):
        if arm_tag == "left":
            entity = self.left_entity
            joints = self.left_passive_gripper_mimic
            stiffness = self.left_passive_gripper_mimic_stiffness
            damping = self.left_passive_gripper_mimic_damping
            gripper_velocity_gain = self.left_gripper_velocity_gain
            drive_passive_mimic = self.left_drive_passive_gripper_mimic_targets
            drive_all_passive_mimic = self.left_drive_all_passive_gripper_mimic_targets
            gripper_config = self.left_gripper_name
            gripper_scale = self.left_gripper_scale
            gripper_val = self.left_gripper_val
            gripper_joints = self.left_gripper
        else:
            entity = self.right_entity
            joints = self.right_passive_gripper_mimic
            stiffness = self.right_passive_gripper_mimic_stiffness
            damping = self.right_passive_gripper_mimic_damping
            gripper_velocity_gain = self.right_gripper_velocity_gain
            drive_passive_mimic = self.right_drive_passive_gripper_mimic_targets
            drive_all_passive_mimic = self.right_drive_all_passive_gripper_mimic_targets
            gripper_config = self.right_gripper_name
            gripper_scale = self.right_gripper_scale
            gripper_val = self.right_gripper_val
            gripper_joints = self.right_gripper
        if not joints or (stiffness <= 0 and damping <= 0):
            return
        qpos = entity.get_qpos()
        active_joints = entity.get_active_joints()
        mimic_targets = {}
        master_names = set()
        if drive_passive_mimic and not gripper_config.get("drive_mimic", True):
            if gripper_joints:
                real_gripper_val = float(gripper_joints[0][0].get_drive_target()[0])
            else:
                real_gripper_val = gripper_scale[0] + gripper_val * (gripper_scale[1] - gripper_scale[0])
            mimic_targets = {
                joint_name: real_gripper_val * multiplier + offset
                for joint_name, multiplier, offset in gripper_config.get("mimic", [])
            }
            master_names = {joint[0].get_name() for joint in gripper_joints}
        for joint in joints:
            joint.set_drive_property(stiffness=float(stiffness), damping=float(damping))
            target = qpos[active_joints.index(joint)]
            if (
                drive_passive_mimic
                and joint.get_name() in mimic_targets
                and (
                    drive_all_passive_mimic
                    or self._is_cross_branch_passive_mimic_joint(entity, master_names, joint)
                )
            ):
                target = mimic_targets[joint.get_name()]
            drive_velocity_target = (
                np.clip(target - joint.drive_target, -1.0, 1.0) * float(gripper_velocity_gain)
            )
            joint.set_drive_target(float(target))
            joint.set_drive_velocity_target(float(drive_velocity_target))
        self._sync_passive_gripper_mimic_qpos(arm_tag)


def planner_process_worker(conn, args):
    import os
    from .planner import CuroboPlanner  # 或者绝对路径导入

    planner = CuroboPlanner(args["origin_pose"], args["joints_name"], args["all_joints"], yml_path=args["yml_path"])

    while True:
        try:
            msg = conn.recv()
            if msg["cmd"] == "plan_path":
                result = planner.plan_path(
                    msg["qpos"],
                    msg["target_pose"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                )
                conn.send(result)

            elif msg["cmd"] == "plan_batch":
                result = planner.plan_batch(
                    msg["qpos"],
                    msg["target_pose_list"],
                    constraint_pose=msg.get("constraint_pose", None),
                    arms_tag=msg["arms_tag"],
                )
                conn.send(result)

            elif msg["cmd"] == "plan_grippers":
                result = planner.plan_grippers(
                    msg["now_val"],
                    msg["target_val"],
                )
                conn.send(result)

            elif msg["cmd"] == "update_point_cloud":
                planner.update_point_cloud(msg["pcd"], resolution=msg.get("resolution", 0.02))
                conn.send("ok")

            elif msg["cmd"] == "reset":
                planner.motion_gen.reset(reset_seed=True)
                conn.send("ok")

            elif msg["cmd"] == "exit":
                conn.close()
                break

            else:
                conn.send({"error": f"Unknown command {msg['cmd']}"})

        except EOFError:
            break
        except Exception as e:
            conn.send({"error": str(e)})
