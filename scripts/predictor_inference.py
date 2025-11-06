import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.shared import download
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def get_dataset(config):
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    transformed_dataset = _data_loader.transform_dataset(dataset, data_config)
    return transformed_dataset


def get_episode_data_index(config):
    repo_id = config.data.repo_id
    dataset = lerobot_dataset.LeRobotDataset(repo_id)
    return dataset.episode_data_index

def convert_to_observation(observation_dict):
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

config = _config.get_config("pi0_fast_libero_predictor")
checkpoint_dir = download.maybe_download("/scratch/s5649552/openpi/checkpoints/pi0_fast_libero_predictor/predictor_v1/1999")
model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

episode_data_index = get_episode_data_index(config)
dataset = get_dataset(config)

episode = 0
ep_start = episode_data_index["from"][episode].cpu().numpy()
ep_end = episode_data_index["to"][episode].cpu().numpy()
observation = get_observation(dataset, int(ep_start))
actions = dataset[0]["actions"].unsqueeze(0).cpu().numpy()
actions = jnp.array(actions)
print("Actions shape:", actions.shape)

rng = jax.random.PRNGKey(0)
past_emb, future_emb = model.predict_future(rng, observation, actions)
print("Past embedding shape:", past_emb[0].shape)
print("Future embedding shape:", future_emb[0].shape)

goal_embedding = jnp.load("/scratch/s5649552/openpi/src/reward_estimator/goal_embedding_1.npy")
baseline_embedding = jnp.load("/scratch/s5649552/openpi/src/reward_estimator/baseline_embedding_1.npy")

past_rewards = []
future_rewards = []

for past_emb_i in past_emb:
    past_reward = compute_regularized_reward(
        state_embedding=past_emb_i,
        goal_embedding=goal_embedding,
        baseline_embedding=baseline_embedding,
        alpha=0.5,
    )
    past_rewards.append(past_reward)

for future_emb in future_emb:
    future_reward = compute_regularized_reward(
        state_embedding=future_emb,
        goal_embedding=goal_embedding,
        baseline_embedding=baseline_embedding,
        alpha=0.5,
    )
    future_rewards.append(future_reward)

print("Past reward:", past_rewards)
print("Future reward:", future_rewards)