import argparse
import jax
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax.numpy as jnp

from openpi.training import config as _config
import openpi.training.data_loader as _data_loader
import openpi.shared.array_typing as at
import openpi.training.weight_loaders as _weight_loaders
from openpi.models import model as _model
from openpi.models.pi0_predictor import Pi0Predictor, make_attn_mask

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from reward_estimator.plot_reward import plot_values


def _load_weights_and_validate(
    loader: _weight_loaders.WeightLoader, params_shape: at.Params
) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    # at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded_params).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )


def get_fused_embedding(model: Pi0Predictor, observation: _model.Observation, rng):
    image_embeddings = model.embed_inputs(observation, train=False, rng=rng)
    fused_embeddings = model.get_fused_embedding(image_embeddings, observation)
    return fused_embeddings


def compute_goal_embedding(model: Pi0Predictor, dataset, episode_data_index):
    print("Computing goal embedding...")
    end_episodes = episode_data_index["to"].cpu().numpy()
    num_episodes = len(end_episodes)
    goal_embeddings = []
    rng = jax.random.key(0)
    for i in range(num_episodes):
        print(f"Processing episode {i+1}/{num_episodes}", end="\r")
        index = int(end_episodes[i]) - 1
        observation = get_observation(dataset, index, model._image_key)
        rng, sub_rng = jax.random.split(rng)
        goal_embedding = get_fused_embedding(model, observation, sub_rng)[0]
        goal_embeddings.append(goal_embedding)
    goal_embeddings = jnp.stack(goal_embeddings, axis=0)
    goal_embeddings = jnp.mean(goal_embeddings, axis=0)
    return goal_embeddings


def compute_baseline_embedding(model: Pi0Predictor, dataset, episode_data_index):
    print("Computing baseline embedding...")
    start_episodes = episode_data_index["from"].cpu().numpy()
    num_episodes = len(start_episodes)
    baseline_embeddings = []
    rng = jax.random.key(0)
    for i in range(num_episodes):
        print(f"Processing episode {i+1}/{num_episodes}", end="\r")
        index = int(start_episodes[i])
        observation = get_observation(dataset, index, model._image_key)
        rng, sub_rng = jax.random.split(rng)
        baseline_embedding = get_fused_embedding(model, observation, sub_rng)[0]
        baseline_embeddings.append(baseline_embedding)
    baseline_embeddings = jnp.stack(baseline_embeddings, axis=0)
    baseline_embeddings = jnp.mean(baseline_embeddings, axis=0)
    return baseline_embeddings


def cosine_similarity(emb1, emb2):
    """Calculates the cosine similarity between two embeddings."""
    dot_product = jnp.dot(emb1, emb2)
    norm_emb1 = jnp.linalg.norm(emb1)
    norm_emb2 = jnp.linalg.norm(emb2)
    return dot_product / (norm_emb1 * norm_emb2)


def distance_similarity(emb1, emb2):
    """Calculates the Euclidean distance between two embeddings."""
    return jnp.linalg.norm(emb1 - emb2)


def load_model(config) -> Pi0Predictor:
    rng = jax.random.key(42)  # or any seed
    model_rng, _ = jax.random.split(rng)

    model = config.model.create(model_rng)

    params_shape = nnx.state(model).to_pure_dict()

    loaded_params = _load_weights_and_validate(config.weight_loader, params_shape)

    graphdef, state = nnx.split(model)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)
    return model


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


def get_observation(dataset, index, image_key):
    element = dataset[index]
    batched_element = jax.tree.map(
        lambda x: jnp.expand_dims(jnp.array(x), axis=0), element
    )
    horizon = batched_element["image"][image_key].shape[1] // 2
    batched_element["image"][image_key] = batched_element["image"][image_key][
        :, horizon : horizon + 1, :, :, :
    ]
    observation = _model.Observation.from_dict(batched_element)
    return observation


def save_embedding(embedding, filename):
    jnp.save(filename, embedding)
    print(f"Saved embedding to {filename}")


def compute_regularized_reward(
    state_embedding: jnp.ndarray,
    goal_embedding: jnp.ndarray,
    baseline_embedding: jnp.ndarray,
    alpha: float,
) -> jnp.ndarray:
    """
    Computes the CLIP reward with Goal-Baseline Regularization.
    Based on Definition 1 from the VLM-RMs paper.

    Args:
        state_embedding: The L2-normalized embedding of the current observation.
        goal_embedding: The L2-normalized embedding of the goal prompt.
        baseline_embedding: The L2-normalized embedding of the baseline prompt.
        alpha: The regularization strength hyperparameter (between 0 and 1).

    Returns:
        The scalar reward value.
    """
    # For convenience, let's rename to match the paper's notation
    s = state_embedding
    g = goal_embedding
    b = baseline_embedding

    # 1. Define the direction vector for the projection
    direction_vector = g - b
    direction_vector_norm_sq = jnp.sum(direction_vector**2)

    # 2. Project the state embedding 's' onto the line defined by 'b' and 'g'
    # The projection of (s - b) onto the direction_vector gives the component along that line.
    s_minus_b = s - b
    projection_scalar = jnp.dot(s_minus_b, direction_vector) / jnp.maximum(
        direction_vector_norm_sq, 1e-6
    )
    # The final projected embedding in the original space
    projected_s = b + projection_scalar * direction_vector

    # 3. Interpolate between the original and projected embeddings using alpha
    blended_embedding = (1 - alpha) * s + alpha * projected_s

    # 4. Calculate the final reward as the negative squared L2 distance to the goal
    # The paper formulates it as 1 - 0.5 * ||...||^2
    reward = 1.0 - 0.5 * jnp.sum((blended_embedding - g) ** 2)

    return reward


def compute(config_path: str):
    config = _config.get_config(config_path)
    model = load_model(config)
    dataset = get_dataset(config)
    episode_data_index = get_episode_data_index(config)
    baseline_embedding = compute_baseline_embedding(model, dataset, episode_data_index)
    save_embedding(
        baseline_embedding,
        f"reward_estimation_embeddings/baseline_embedding_{config_path}.npy",
    )
    goal_embedding = compute_goal_embedding(model, dataset, episode_data_index)
    save_embedding(
        goal_embedding, f"reward_estimation_embeddings/goal_embedding_{config_path}.npy"
    )


def evaluate(config_path: str, episode: int, skip: int = 20):
    config = _config.get_config(config_path)
    episode_data_index = get_episode_data_index(config)
    baseline_embedding = jnp.load(
        f"reward_estimation_embeddings/baseline_embedding_{config_path}.npy"
    )
    print(f"Loaded baseline embedding shape: {baseline_embedding.shape}")
    goal_embedding = jnp.load(f"reward_estimation_embeddings/goal_embedding_{config_path}.npy")
    print(f"Loaded goal embedding shape: {goal_embedding.shape}")

    dataset = get_dataset(config)
    model = load_model(config)
    ep_start = episode_data_index["from"][episode].cpu().numpy()
    ep_end = episode_data_index["to"][episode].cpu().numpy()
    print(f"Evaluating episode {episode}, steps {ep_start} to {ep_end}")
    rewards = []
    rng = jax.random.key(0)
    for i in range(ep_start, ep_end, skip):
        observation = get_observation(dataset, i, model._image_key)
        rng, sub_rng = jax.random.split(rng)
        embedding = get_fused_embedding(model, observation, sub_rng)[0]
        reward = compute_regularized_reward(
            state_embedding=embedding,
            goal_embedding=goal_embedding,
            baseline_embedding=baseline_embedding,
            alpha=0.5,
        )
        rewards.append(reward)
        print(f"Step {i}: regularized reward: {reward}")
    plot_values(ep_start, ep_end, skip, rewards, episode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute or evaluate goal embeddings")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="pi0_libero_predictor",
        help="Path or name of the config to load",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["compute", "evaluate"],
        default="evaluate",
        help="Run mode: compute goal embedding or evaluate an episode",
    )
    parser.add_argument(
        "--episode",
        "-e",
        type=int,
        default=2,
        help="Episode index to evaluate (only used in evaluate mode)",
    )
    parser.add_argument(
        "--skip",
        "-s",
        type=int,
        default=5,
        help="Skip interval between steps when evaluating",
    )

    args = parser.parse_args()

    if args.mode == "compute":
        compute(args.config)
    else:
        evaluate(args.config, args.episode, args.skip)
