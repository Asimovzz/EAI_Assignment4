import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

sys.path.append("./")

from envs import CONFIGS_PATH
from script.collect_data import class_decorator, get_embodiment_config


TASK_NAME = "beat_block_hammer"
DEFAULT_CLEAN_CONFIG = "galbot_demo_clean"
DEFAULT_RANDOMIZED_CONFIG = "galbot_demo_randomized"
REQUIRED_RGB_CAMERAS = ("head_camera", "left_camera", "right_camera")


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def update_embodiment_config_paths():
    subprocess.run(
        [sys.executable, "script/update_embodiment_config_path.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def load_embodiment_types():
    return load_yaml(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"))


def keep_required_static_cameras(embodiment_config, required_camera_names):
    config = deepcopy(embodiment_config)
    static_cameras = config.get("static_camera_list", [])
    required_static_names = {
        camera_name for camera_name in required_camera_names if camera_name not in ("left_camera", "right_camera")
    }
    kept_cameras = [
        deepcopy(camera) for camera in static_cameras if camera.get("name") in required_static_names
    ]
    kept_names = {camera.get("name") for camera in kept_cameras}
    missing_names = required_static_names - kept_names
    if missing_names:
        raise ValueError(f"embodiment config is missing static cameras: {sorted(missing_names)}")
    config["static_camera_list"] = kept_cameras
    return config


def resolve_robot_args(args):
    embodiment_types = load_embodiment_types()
    embodiment_type = args.get("embodiment")
    if len(embodiment_type) == 1:
        robot_file = embodiment_types[embodiment_type[0]]["file_path"]
        args["left_robot_file"] = robot_file
        args["right_robot_file"] = robot_file
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment_type[0])
    elif len(embodiment_type) == 3:
        left_robot_file = embodiment_types[embodiment_type[0]]["file_path"]
        right_robot_file = embodiment_types[embodiment_type[1]]["file_path"]
        args["left_robot_file"] = left_robot_file
        args["right_robot_file"] = right_robot_file
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment_type[0]}+{embodiment_type[1]}"
    else:
        raise ValueError("embodiment must contain one robot name or left/right/distance")

    left_config = get_embodiment_config(args["left_robot_file"])
    right_config = get_embodiment_config(args["right_robot_file"])
    args["left_embodiment_config"] = keep_required_static_cameras(left_config, REQUIRED_RGB_CAMERAS)
    args["right_embodiment_config"] = keep_required_static_cameras(right_config, REQUIRED_RGB_CAMERAS)


def make_collect_args(task_config, episode_num, save_path, force_arm_tag=None):
    config_path = Path("task_config") / f"{task_config}.yml"
    args = load_yaml(config_path)

    args["task_name"] = TASK_NAME
    args["task_config"] = task_config
    args["episode_num"] = int(episode_num)
    args["use_seed"] = False
    args["collect_data"] = True
    args["render_freq"] = 0
    args["save_path"] = str(Path(save_path) / TASK_NAME / task_config)
    args["need_plan"] = True
    args["save_data"] = True
    if force_arm_tag is not None:
        args["force_arm_tag"] = force_arm_tag

    args.setdefault("camera", {})
    args["camera"]["head_camera_type"] = args["camera"].get("head_camera_type", "D435")
    args["camera"]["wrist_camera_type"] = args["camera"].get("wrist_camera_type", "D435")
    args["camera"]["collect_head_camera"] = True
    args["camera"]["collect_wrist_camera"] = True

    args.setdefault("data_type", {})
    args["data_type"]["rgb"] = True
    args["data_type"]["third_view"] = False
    args["data_type"]["depth"] = False
    args["data_type"]["pointcloud"] = False
    args["data_type"]["observer"] = False
    args["data_type"]["mesh_segmentation"] = False
    args["data_type"]["actor_segmentation"] = False

    resolve_robot_args(args)
    return args


def remove_pointcloud_dataset(hdf5_path):
    import h5py

    with h5py.File(hdf5_path, "a") as f:
        if "pointcloud" in f:
            del f["pointcloud"]


def verify_episode_hdf5(hdf5_path):
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        observation_cameras = set(f["observation"].keys()) if "observation" in f else set()
        required_cameras = set(REQUIRED_RGB_CAMERAS)
        if observation_cameras != required_cameras:
            raise RuntimeError(
                f"{hdf5_path} observation cameras mismatch: "
                f"expected {sorted(required_cameras)}, got {sorted(observation_cameras)}"
            )
        missing = []
        for camera_name in REQUIRED_RGB_CAMERAS:
            key = f"observation/{camera_name}/rgb"
            if key not in f:
                missing.append(key)
        if missing:
            raise RuntimeError(f"{hdf5_path} missing required RGB datasets: {missing}")
        pointcloud_keys = [key for key in f.keys() if "pointcloud" in key.lower()]
        f.visit(lambda key: pointcloud_keys.append(key) if "pointcloud" in key.lower() else None)
        if pointcloud_keys:
            raise RuntimeError(f"{hdf5_path} still contains pointcloud datasets: {pointcloud_keys}")


def postprocess_and_verify(save_dir, episode_num):
    data_dir = Path(save_dir) / "data"
    for episode_idx in range(episode_num):
        hdf5_path = data_dir / f"episode{episode_idx}.hdf5"
        if not hdf5_path.exists():
            raise FileNotFoundError(f"Missing collected episode: {hdf5_path}")
        remove_pointcloud_dataset(hdf5_path)
        verify_episode_hdf5(hdf5_path)


def episode_hdf5_is_valid(save_dir, episode_idx):
    hdf5_path = Path(save_dir) / "data" / f"episode{episode_idx}.hdf5"
    if not hdf5_path.exists():
        return False
    try:
        verify_episode_hdf5(hdf5_path)
    except Exception as exc:
        print(f"[galbot collect] existing episode {episode_idx} is invalid: {exc}")
        return False
    return True


def load_seed_list(save_dir):
    seed_path = Path(save_dir) / "seed.txt"
    if not seed_path.exists():
        return []
    seeds = seed_path.read_text(encoding="utf-8").split()
    return [int(seed) for seed in seeds]


def save_seed_list(save_dir, seed_list):
    seed_path = Path(save_dir) / "seed.txt"
    seed_path.write_text(" ".join(str(seed) for seed in seed_list), encoding="utf-8")


def cleanup_episode_outputs(save_dir, episode_idx):
    save_dir = Path(save_dir)
    for relative_path in (
        Path("data") / f"episode{episode_idx}.hdf5",
        Path("video") / f"episode{episode_idx}.mp4",
        Path("video") / "head_camera" / f"episode{episode_idx}.mp4",
        Path("video") / "left_camera" / f"episode{episode_idx}.mp4",
        Path("video") / "right_camera" / f"episode{episode_idx}.mp4",
        Path("_traj_data") / f"episode{episode_idx}.pkl",
    ):
        path = save_dir / relative_path
        if path.exists():
            path.unlink()


def collect_one(
    task_config,
    episode_num,
    save_path,
    overwrite,
    max_failures_per_episode,
    force_arm_tag=None,
    seed_start=None,
    seed_stride=1,
):
    args = make_collect_args(task_config, episode_num, save_path, force_arm_tag=force_arm_tag)
    save_dir = Path(args["save_path"])
    if overwrite and save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[galbot collect] task_config={task_config} episodes={episode_num} save_dir={save_dir}")
    if args.get("force_arm_tag"):
        print(f"[galbot collect] arm selection: forced {args['force_arm_tag']}")
    else:
        print("[galbot collect] arm selection: block x < 0 -> left, else right")
    print(f"[galbot collect] required RGB cameras: {', '.join(REQUIRED_RGB_CAMERAS)}")
    print("[galbot collect] pointcloud collection: disabled")

    task = class_decorator(TASK_NAME)
    scene_info_path = save_dir / "scene_info.json"
    if not scene_info_path.exists():
        scene_info_path.write_text("{}", encoding="utf-8")
    seed_list = load_seed_list(save_dir)
    existing_valid_count = 0
    for episode_idx in range(episode_num):
        if episode_hdf5_is_valid(save_dir, episode_idx):
            existing_valid_count += 1
    seed_stride = max(1, int(seed_stride))
    if seed_start is None:
        seed = max(seed_list) + 1 if seed_list else existing_valid_count
    elif seed_list:
        seed = max(seed_list) + seed_stride
    else:
        seed = int(seed_start)

    for episode_idx in range(episode_num):
        if not overwrite and episode_hdf5_is_valid(save_dir, episode_idx):
            print(f"collect data episode {episode_idx} exists, skip.")
            continue

        failures = 0
        while True:
            if failures >= max_failures_per_episode:
                raise RuntimeError(
                    f"episode {episode_idx} exceeded {max_failures_per_episode} failed seeds"
                )

            cleanup_episode_outputs(save_dir, episode_idx)
            env_closed = False
            try:
                task.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
                info = task.play_once()
                episode_success = bool(task.plan_success and task.check_success())
                if not episode_success:
                    raise RuntimeError("plan_success/check_success failed")

                task.save_traj_data(episode_idx)

                task.close_env(clear_cache=((episode_idx + 1) % args["clear_cache_freq"] == 0))
                env_closed = True
                task.merge_pkl_to_hdf5_video()
                task.remove_data_cache()

                hdf5_path = save_dir / "data" / f"episode{episode_idx}.hdf5"
                remove_pointcloud_dataset(hdf5_path)
                verify_episode_hdf5(hdf5_path)

                info_db = json.loads(scene_info_path.read_text(encoding="utf-8"))
                info_db[f"episode_{episode_idx}"] = {
                    **info,
                    "success": episode_success,
                    "seed": seed,
                }
                scene_info_path.write_text(
                    json.dumps(info_db, ensure_ascii=False, indent=4),
                    encoding="utf-8",
                )
                seed_list.append(seed)
                save_seed_list(save_dir, seed_list)
                print(f"collect data episode {episode_idx} success! (seed = {seed})")
                seed += seed_stride
                break

            except Exception as exc:
                print(
                    f"collect data episode {episode_idx} fail! "
                    f"(seed = {seed}, error = {type(exc).__name__}: {exc})"
                )
                failures += 1
                seed += seed_stride
                if not env_closed:
                    try:
                        task.close_env()
                    except Exception:
                        pass
                try:
                    task.remove_data_cache()
                except Exception:
                    pass
                cleanup_episode_outputs(save_dir, episode_idx)
                task = class_decorator(TASK_NAME)

    postprocess_and_verify(save_dir, episode_num)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect Galbot beat_block_hammer clean/randomized RGB datasets."
    )
    parser.add_argument("--clean-config", default=DEFAULT_CLEAN_CONFIG)
    parser.add_argument("--randomized-config", default=DEFAULT_RANDOMIZED_CONFIG)
    parser.add_argument("--clean-episodes", type=int, default=50)
    parser.add_argument("--randomized-episodes", type=int, default=500)
    parser.add_argument("--save-path", default="./data")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-randomized", action="store_true")
    parser.add_argument("--skip-render-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-failures-per-episode", type=int, default=100)
    parser.add_argument("--force-arm-tag", choices=("left", "right"), default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--seed-stride", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    update_embodiment_config_paths()

    if args.dry_run:
        for task_config, episode_num, skipped in (
            (args.clean_config, args.clean_episodes, args.skip_clean),
            (args.randomized_config, args.randomized_episodes, args.skip_randomized),
        ):
            if skipped:
                continue
            collect_args = make_collect_args(
                task_config,
                episode_num,
                args.save_path,
                force_arm_tag=args.force_arm_tag,
            )
            print(
                "[galbot collect dry-run] "
                f"task_config={task_config} episodes={episode_num} "
                f"save_dir={collect_args['save_path']} "
                f"cameras={','.join(REQUIRED_RGB_CAMERAS)} "
                f"pointcloud={collect_args['data_type']['pointcloud']} "
                f"forced_arm={collect_args.get('force_arm_tag')} "
                f"seed_start={args.seed_start} seed_stride={args.seed_stride}"
            )
        return

    if not args.skip_render_test:
        from script.test_render import Sapien_TEST

        Sapien_TEST()

    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    if not args.skip_clean:
        collect_one(
            args.clean_config,
            args.clean_episodes,
            args.save_path,
            args.overwrite,
            args.max_failures_per_episode,
            args.force_arm_tag,
            args.seed_start,
            args.seed_stride,
        )
    if not args.skip_randomized:
        collect_one(
            args.randomized_config,
            args.randomized_episodes,
            args.save_path,
            args.overwrite,
            args.max_failures_per_episode,
            args.force_arm_tag,
            args.seed_start,
            args.seed_stride,
        )


if __name__ == "__main__":
    main()
