import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import yaml
import cv2
import h5py


def load_hdf5(dataset_path):
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        vector = root["/joint_action/vector"][()]
        image_dict = dict()
        for cam_name in root[f"/observation/"].keys():
            image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]

    return vector, image_dict


def main():
    parser = argparse.ArgumentParser(description="Process some episodes.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument("task_config", type=str)
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--load-dir",
        default=None,
        help="Directory containing data/episode*.hdf5. Defaults to ../../data/<task>/<config>.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Output zarr path. Defaults to ./data/<task>-<config>-<num>.zarr.",
    )
    parser.add_argument(
        "--right-arm-only",
        action="store_true",
        help="Store only right arm and right gripper joints, indices [8:16], in state/action.",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = args.load_dir or ("../../data/" + str(task_name) + "/" + str(task_config))

    total_count = 0

    save_dir = args.save_dir or f"./data/{task_name}-{task_config}-{num}.zarr"

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = (
        [],
        [],
        [],
        [],
    )

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        (
            vector_all,
            image_dict_all,
        ) = load_hdf5(load_path)
        if "head_camera" not in image_dict_all:
            raise RuntimeError(f"episode{current_ep} is missing head_camera in {load_path}")
        if current_ep == 0:
            head_camera_arrays = []

        for j in range(0, vector_all.shape[0]):

            joint_state = vector_all[j]
            if args.right_arm_only:
                joint_state = joint_state[8:16]

            if j != vector_all.shape[0] - 1:
                head_img_bit = image_dict_all["head_camera"][j]
                head_img = cv2.imdecode(np.frombuffer(head_img_bit, np.uint8), cv2.IMREAD_COLOR)
                if head_img is None:
                    raise RuntimeError(f"failed to decode head_camera frame {j} in {load_path}")
                head_img = cv2.cvtColor(head_img, cv2.COLOR_BGR2RGB)
                head_camera_arrays.append(head_img)
                state_arrays.append(joint_state)
            if j != 0:
                joint_action = vector_all[j]
                if args.right_arm_only:
                    joint_action = joint_action[8:16]
                joint_action_arrays.append(joint_action)

        current_ep += 1
        total_count += vector_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    # action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    if args.right_arm_only and (state_arrays.shape[-1] != 8 or joint_action_arrays.shape[-1] != 8):
        raise RuntimeError(
            f"right-arm-only zarr must be 8D, got state={state_arrays.shape} action={joint_action_arrays.shape}"
        )
    head_camera_arrays = np.moveaxis(np.array(head_camera_arrays), -1, 1)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)
    # action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=joint_action_arrays,
        chunks=joint_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


if __name__ == "__main__":
    main()
