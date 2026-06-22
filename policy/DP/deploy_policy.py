import numpy as np
from .dp_model import DP
import yaml


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"cannot parse boolean value: {value!r}")


def _expected_agent_pos_dim(expected_obs_keys):
    if expected_obs_keys is None:
        return None
    shape_map = expected_obs_keys if isinstance(expected_obs_keys, dict) else None
    if shape_map is None:
        return None
    shape = shape_map.get("agent_pos")
    if shape is None:
        return None
    return int(shape[0])


def encode_obs(observation, expected_obs_keys=None):
    obs = {
        "head_cam": np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255,
    }
    agent_pos = observation["joint_action"]["vector"]
    if _expected_agent_pos_dim(expected_obs_keys) == 8:
        agent_pos = agent_pos[8:16]
    obs["agent_pos"] = agent_pos
    return obs


def expand_right_action_if_needed(action, observation):
    action = np.asarray(action)
    if action.shape[-1] != 8:
        return action
    full_action = np.asarray(observation["joint_action"]["vector"], dtype=action.dtype).copy()
    full_action[8:16] = action
    return full_action


def get_model(usr_args):
    ckpt_file = usr_args.get("ckpt_file")
    if ckpt_file is None:
        ckpt_file = f"./policy/DP/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = int(usr_args.get("action_dim", usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2))
    
    load_config_path = f'./policy/DP/diffusion_policy/config/robot_dp_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = int(usr_args.get("n_obs_steps", model_training_config["n_obs_steps"]))
    n_action_steps = int(usr_args.get("n_action_steps", model_training_config["n_action_steps"]))
    
    return DP(
        ckpt_file,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        use_ema=parse_bool(usr_args.get("use_ema", True)),
        inference_seed=usr_args.get("policy_inference_seed", None),
    )


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    expected_obs_keys = getattr(model, "expected_obs_keys", None)
    obs = encode_obs(observation, expected_obs_keys)
    instruction = TASK_ENV.get_instruction()

    # ======== Get Action ========
    if len(model.runner.obs) == 0:
        actions = model.get_action(obs)
    else:
        actions = model.get_action()

    for action in actions:
        env_action = expand_right_action_if_needed(action, observation)
        if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim or TASK_ENV.eval_success:
            break
        TASK_ENV.take_action(env_action)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation, expected_obs_keys)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
