import argparse
import shutil
from pathlib import Path

import zarr
from numcodecs import Blosc


REQUIRED_KEYS = ("head_camera", "state", "action")


def episode_slice(episode_ends, episode_idx):
    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])
    return slice(start, end)


def describe_zarr(path: Path):
    root = zarr.open(str(path), mode="r")
    keys = list(root["data"].keys())
    episode_ends = root["meta"]["episode_ends"][:]
    print(f"{path}")
    print(f"  episodes: {len(episode_ends)}")
    print(f"  steps: {int(episode_ends[-1]) if len(episode_ends) else 0}")
    for key in keys:
        print(f"  data/{key}: shape={root['data'][key].shape}, dtype={root['data'][key].dtype}")
    for key in REQUIRED_KEYS:
        if key not in root["data"]:
            raise KeyError(f"{path} is missing data/{key}")
    return root


def create_output(output_path: Path, first_root):
    compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    root = zarr.open(str(output_path), mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    for key in REQUIRED_KEYS:
        src = first_root["data"][key]
        chunks = (min(100, max(1, src.shape[0])),) + src.shape[1:]
        data_group.create_dataset(
            key,
            shape=(0,) + src.shape[1:],
            chunks=chunks,
            dtype=src.dtype,
            compressor=compressor,
            overwrite=True,
        )
    meta_group.create_dataset(
        "episode_ends",
        shape=(0,),
        chunks=(100,),
        dtype="int64",
        compressor=compressor,
        overwrite=True,
    )
    return root


def append_episode(output_root, input_root, episode_idx):
    input_ends = input_root["meta"]["episode_ends"][:]
    sl = episode_slice(input_ends, episode_idx)
    ep_len = sl.stop - sl.start

    output_ends = output_root["meta"]["episode_ends"]
    old_steps = int(output_ends[-1]) if output_ends.shape[0] else 0
    new_steps = old_steps + ep_len

    for key in REQUIRED_KEYS:
        src = input_root["data"][key]
        dst = output_root["data"][key]
        if src.shape[1:] != dst.shape[1:]:
            raise ValueError(
                f"shape mismatch for {key}: input {src.shape[1:]} vs output {dst.shape[1:]}"
            )
        dst.resize((new_steps,) + dst.shape[1:])
        dst[old_steps:new_steps] = src[sl]

    output_ends.resize((output_ends.shape[0] + 1,))
    output_ends[-1] = new_steps


def main():
    parser = argparse.ArgumentParser(description="Merge DP zarr datasets episode-by-episode.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input zarr path. Can be specified multiple times.",
    )
    parser.add_argument(
        "--repeat",
        action="append",
        type=int,
        default=None,
        help="Repeat count for the corresponding --input. Defaults to 1 for each input.",
    )
    parser.add_argument("--output", required=True, help="Output zarr path.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    repeats = args.repeat or [1] * len(input_paths)
    if len(repeats) != len(input_paths):
        raise ValueError("--repeat count must match --input count")

    for path in input_paths:
        if not path.is_dir():
            raise FileNotFoundError(path)

    output_path = Path(args.output)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    roots = [describe_zarr(path) for path in input_paths]
    output_root = create_output(output_path, roots[0])

    for root, path, repeat in zip(roots, input_paths, repeats):
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1 for {path}")
        n_episodes = len(root["meta"]["episode_ends"])
        for rep_idx in range(repeat):
            print(f"copy {path} repeat {rep_idx + 1}/{repeat}")
            for episode_idx in range(n_episodes):
                append_episode(output_root, root, episode_idx)

    print("merged:")
    describe_zarr(output_path)


if __name__ == "__main__":
    main()
