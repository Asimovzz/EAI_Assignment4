import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra, pdb
from omegaconf import OmegaConf
import pathlib, yaml
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import os
import zarr

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def fill_camera_shapes_from_zarr(cfg):
    zarr_path = cfg.task.dataset.zarr_path
    if not os.path.isabs(zarr_path):
        zarr_path = os.path.join(parent_directory, zarr_path)
    if not os.path.isdir(zarr_path):
        return

    root = zarr.open(zarr_path, mode="r")
    for obs_name, obs_cfg in cfg.task.shape_meta.obs.items():
        if obs_cfg.get("type", "low_dim") != "rgb":
            continue
        camera_name = obs_name.replace("_cam", "_camera")
        if camera_name in root["data"]:
            obs_cfg.shape = list(root["data"][camera_name].shape[1:])


# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("diffusion_policy", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    head_camera_type = cfg.head_camera_type
    head_camera_cfg = get_camera_config(head_camera_type)
    cfg.task.image_shape = [3, head_camera_cfg["h"], head_camera_cfg["w"]]
    if "head_cam" in cfg.task.shape_meta.obs:
        cfg.task.shape_meta.obs.head_cam.shape = [
            3,
            head_camera_cfg["h"],
            head_camera_cfg["w"],
        ]
    fill_camera_shapes_from_zarr(cfg)
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    print(cfg.task.dataset.zarr_path, cfg.task_name)
    workspace.run()


if __name__ == "__main__":
    main()
