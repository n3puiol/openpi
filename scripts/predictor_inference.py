import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_dataset(config):
    import openpi.training.data_loader as _data_loader

    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed_dataset = _data_loader.transform_dataset(dataset, data_config)
    return transformed_dataset


def get_episode_data_index(config):
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    repo_id = config.data.repo_id
    dataset = lerobot_dataset.LeRobotDataset(repo_id)
    return dataset.episode_data_index


def convert_to_observation(observation_dict):
    from openpi.models import model as _model

    batched_element = jax.tree.map(
        lambda x: jnp.expand_dims(jnp.array(x), axis=0), observation_dict
    )
    observation = _model.Observation.from_dict(batched_element)
    return observation


def get_observation(dataset, index):
    element = dataset[index]
    observation = convert_to_observation(element)
    return observation


def compute_regularized_reward(
    state_embedding: jnp.ndarray,
    goal_embedding: jnp.ndarray,
    baseline_embedding: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    s = state_embedding
    g = goal_embedding
    b = baseline_embedding

    direction_vector = g - b
    direction_vector_norm_sq = jnp.sum(direction_vector**2)

    s_minus_b = s - b
    projection_scalar = jnp.dot(s_minus_b, direction_vector) / jnp.maximum(
        direction_vector_norm_sq, 1e-6
    )
    projected_s = b + projection_scalar * direction_vector

    blended_embedding = (1 - alpha) * s + alpha * projected_s

    reward = 1.0 - 0.5 * jnp.sum((blended_embedding - g) ** 2)
    return reward


def normalized_rewards(rewards, window_size=20):
    y_vals = np.asarray([float(r) for r in rewards])

    # reward_series = pd.Series(y_vals)
    # smoothed_rewards = reward_series.rolling(window=window_size).mean()

    # normalized_rewards = (smoothed_rewards - np.min(smoothed_rewards)) / (
    #     np.max(smoothed_rewards) - np.min(smoothed_rewards)
    # )
    # return normalized_rewards
    return y_vals


def plot_rewards(past_rewards, future_rewards, episode):
    import matplotlib.pyplot as plt

    past_normalized = normalized_rewards(
        past_rewards[10:], window_size=10
    )
    print("Past normalized length:", len(past_normalized))
    future_normalized = normalized_rewards(
        future_rewards[: len(future_rewards) - 10], window_size=10
    )
    print("Future normalized length:", len(future_normalized))

    steps = np.array(list(range(0, len(past_normalized))))

    plt.figure(figsize=(10, 6))
    plt.plot(
        steps,
        past_normalized,
        linestyle="-",
        color="blue",
        label="Past Estimated Reward",
    )

    plt.plot(
        steps,
        future_normalized,
        linestyle="-",
        color="orange",
        label="Future Estimated Reward",
    )
    plt.title(f"Estimated Reward over Episode {episode}")

    plt.scatter(
        steps,
        past_normalized,
        color="blue",
        s=10,
        label="Past Estimated Reward Points",
    )
    plt.scatter(
        steps,
        future_normalized,
        color="orange",
        s=10,
        label="Future Estimated Reward Points",
    )

    for i in range(0, len(past_normalized), 5):
        plt.axvline(
            x=i,
            color="red",
            linestyle="--",
            linewidth=0.5,
            label="Horizon" if i == 0 else "",
        )

    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()
    plt.savefig(f"regularized_reward_plots/past_future_rewards_{episode}.png")


def get_model(config_path: str, checkpoint_path: str):
    from openpi.models.pi0_predictor import Pi0Predictor
    from openpi.shared import download
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    config = _config.get_config(config_path)
    checkpoint_dir = download.maybe_download(checkpoint_path)
    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    model: Pi0Predictor = policy._model
    return model, config


# config = _config.get_config(config_path)
# checkpoint_dir = download.maybe_download(checkpoint_path)
# model: Pi0Predictor = config.model.load(
#     _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
# )
# return model, config


def inference_dataset(episode: int):
    model, config = get_model(
        "pi0_libero_predictor",
        "/scratch/s5649552/openpi/checkpoints/pi0_libero_predictor/predictor_v7/19999",
    )
    episode_data_index = get_episode_data_index(config)
    dataset = get_dataset(config)
    ep_start = episode_data_index["from"][episode].cpu().numpy()
    ep_end = episode_data_index["to"][episode].cpu().numpy()
    print(f"Processing episode {episode}, from {ep_start} to {ep_end}")

    past_rewards = []
    future_rewards = []
    for i in tqdm(range(ep_start, ep_end, model.action_horizon)):
        observation = get_observation(dataset, int(i))
        actions = dataset[int(i)]["actions"].unsqueeze(0).cpu().numpy()
        actions = jnp.array(actions)
        if actions.shape[1] != model.action_horizon:
            print(
                f"Skipping chunk at index {i} due to mismatched past embedding length."
            )
            continue

        rng = jax.random.PRNGKey(0)
        past_emb, future_emb = model.predict_future(rng, observation, actions)
        print(f"Predicted past embedding shape: {past_emb.shape}")
        print(f"Predicted future embedding shape: {future_emb.shape}")

        for i in range(len(past_emb[0])):
            past_emb_i = past_emb[0][i]
            past_reward = model.compute_regularized_reward(state_embedding=past_emb_i)
            past_rewards.append(past_reward)

        for i in range(len(future_emb[0])):
            future_emb_i = future_emb[0][i]
            future_reward = model.compute_regularized_reward(
                state_embedding=future_emb_i
            )
            future_rewards.append(future_reward)

    jnp.save(f"past_rewards_{episode}.npy", jnp.array(past_rewards))
    jnp.save(f"future_rewards_{episode}.npy", jnp.array(future_rewards))

    return past_rewards, future_rewards


def inference_libero(episode: str):
    pass


# episode = "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
episode = 1
# past_rewards, future_rewards = inference_dataset(episode)

past_rewards = jnp.load(f"past_rewards_{episode}.npy")
future_rewards = jnp.load(f"future_rewards_{episode}.npy")

print("Past rewards length:", len(past_rewards))
print("Future rewards length:", len(future_rewards))

plot_rewards(past_rewards, future_rewards, episode)
