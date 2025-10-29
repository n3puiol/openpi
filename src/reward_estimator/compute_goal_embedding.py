from openpi.training import config as _config
import jax
import flax.nnx as nnx
from scripts.train import _load_weights_and_validate
from PIL import Image
import jax.numpy as jnp
import openpi.transforms as _transforms
import numpy as np
import openpi.training.data_loader as _data_loader
import openpi.models.tokenizer as _tokenizer
import openpi.training.sharding as sharding
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
from openpi.models import model as _model
from openpi.models.pi0_fast import Pi0FAST, make_attn_mask
import argparse
from plot_reward import plot_values


def get_fused_embedding(model: Pi0FAST, observation: _model.Observation):
    """
    Computes a single, fused embedding vector from an observation.

    This function performs the full forward pass to get the context-aware
    latent representation from the PaliGemma model and then pools it into
    a single vector.

    Args:
        model: An instance of your Pi0FAST model.
        observation: The input observation containing images and a prompt.

    Returns:
        A single JAX array representing the fused embedding.
    """
    # 1. Get the concatenated sequence of image and text token embeddings
    # This combines the outputs of the image encoder and the text embedder.
    input_token_embeddings, input_mask, ar_mask = model.embed_inputs(observation)

    # 2. Create the attention mask that governs the fusion process
    attn_mask = make_attn_mask(input_mask, ar_mask)

    # 3. Pass the full sequence through the LLM to perform fusion
    # We call the LLM to get the final hidden states before the vocabulary projection.
    # This is the 'pre_logits' output from the transformer.
    fused_sequence_embeddings, _, _ = model.PaliGemma.llm(
        embedded_prefix=input_token_embeddings,
        mask=attn_mask,
        return_prelogits=True,
    )

    # 4. Pool the fused sequence into a single representative vector
    # We use global average pooling across the sequence length.
    # We apply the input_mask to only average over valid (non-padding) tokens.

    # Expand input_mask to be broadcastable with the embeddings
    mask_expanded = jnp.expand_dims(input_mask, axis=-1)

    # Sum the embeddings of valid tokens
    summed_embeddings = jnp.sum(fused_sequence_embeddings * mask_expanded, axis=1)

    # Count the number of valid tokens
    num_valid_tokens = jnp.sum(input_mask, axis=1, keepdims=True)

    # Compute the mean
    pooled_fused_embedding = summed_embeddings / jnp.maximum(num_valid_tokens, 1)

    return pooled_fused_embedding


def compute_goal_embedding(model: Pi0FAST, dataset, episode_data_index):
    print("Computing goal embedding...")
    end_episodes = episode_data_index["to"].cpu().numpy()
    num_episodes = len(end_episodes)
    goal_embeddings = []
    for i in range(num_episodes):
        print(f"Processing episode {i+1}/{num_episodes}", end="\r")
        index = int(end_episodes[i]) - 1
        observation = get_observation(dataset, index)
        goal_embedding = get_fused_embedding(model, observation)[0]
        goal_embeddings.append(goal_embedding)
    goal_embeddings = jnp.stack(goal_embeddings, axis=0)
    goal_embeddings = jnp.mean(goal_embeddings, axis=0)
    return goal_embeddings


def compute_baseline_embedding(model: Pi0FAST, dataset, episode_data_index):
    print("Computing baseline embedding...")
    start_episodes = episode_data_index["from"].cpu().numpy()
    num_episodes = len(start_episodes)
    baseline_embeddings = []
    for i in range(num_episodes):
        print(f"Processing episode {i+1}/{num_episodes}", end="\r")
        index = int(start_episodes[i])
        observation = get_observation(dataset, index)
        baseline_embedding = get_fused_embedding(model, observation)[0]
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


def load_model(config) -> Pi0FAST:
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


def get_observation(dataset, index):
    element = dataset[index]
    observation = convert_to_observation(element["observation"])
    return observation

def convert_to_observation(observation_dict):
    batched_element = jax.tree.map(
        lambda x: jnp.expand_dims(jnp.array(x), axis=0), observation_dict
    )
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
    save_embedding(baseline_embedding, "baseline_embedding_1.npy")
    goal_embedding = compute_goal_embedding(model, dataset, episode_data_index)
    save_embedding(goal_embedding, "goal_embedding_1.npy")


def evaluate(config_path: str, episode: int, skip: int = 20):
    config = _config.get_config(config_path)
    episode_data_index = get_episode_data_index(config)
    goal_embedding = jnp.load("goal_embedding_1.npy")
    print(f"Loaded goal embedding shape: {goal_embedding.shape}")
    baseline_embedding = jnp.load("baseline_embedding_1.npy")
    print(f"Loaded baseline embedding shape: {baseline_embedding.shape}")

    dataset = get_dataset(config)
    model = load_model(config)
    ep_start = episode_data_index["from"][episode].cpu().numpy()
    ep_end = episode_data_index["to"][episode].cpu().numpy()
    print(f"Evaluating episode {episode}, steps {ep_start} to {ep_end}")
    rewards = []
    for i in range(ep_start, ep_end, skip):
        observation = get_observation(dataset, i)
        embedding = get_fused_embedding(model, observation)[0]
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
        default="pi0_fast_libero",
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
