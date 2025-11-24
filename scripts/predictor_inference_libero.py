import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from openpi.models import model as _model
from openpi.models.pi0_predictor import Pi0Predictor
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
import openpi.training.data_loader as _data_loader

config = _config.get_config("pi0_libero_predictor")
checkpoint_dir = download.maybe_download(
    "/scratch/s5649552/openpi/checkpoints/pi0_libero_predictor/predictor_rollout/16000"
)
# model: Pi0Predictor = config.model.load(
#     _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
# )
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
model: Pi0Predictor = policy._model

episode = "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate"
observations = np.load(
    f"LIBERO_VLA/observations_{episode}.npy",
    allow_pickle=True,
)
images = np.load(
    f"LIBERO_VLA/images_{episode}.npy",
    allow_pickle=True,
)
actions = np.load(
    f"LIBERO_VLA/actions_{episode}.npy",
    allow_pickle=True,
)

print("Observations shape:", observations.shape)
print("Images shape:", images.shape)
print("Actions shape:", actions.shape)

action_horizon = model.action_horizon
print("Action horizon:", action_horizon)

past_rewards = []
future_rewards = []
for i in tqdm(range(0, len(observations), action_horizon)):
    obs_chunk_img = images[i : i + action_horizon].astype(float)
    obs_chunk = observations[i]
    obs_chunk["observation/image"] = obs_chunk_img
    print(f"Observation chunk image shape: {obs_chunk['observation/image'].shape}")

    inputs = jax.tree.map(lambda x: x, obs_chunk)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    observation = _model.Observation.from_dict(inputs)

    action_chunk = actions[i : i + action_horizon].astype(float)
    # add batch dimension
    action_chunk = jnp.array(action_chunk)[jnp.newaxis, ...]
    print(f"Action chunk shape: {action_chunk.shape}")

    rng = jax.random.PRNGKey(0)
    past_emb, future_emb = model.predict_future(rng, observation, action_chunk)
    print(f"Predicted past embedding shape: {past_emb.shape}")
    print(f"Predicted future embedding shape: {future_emb.shape}")

    for i in range(0, len(past_emb[0])):
        past_emb_i = past_emb[0][i]
        past_reward = model.compute_regularized_reward(state_embedding=past_emb_i)
        past_rewards.append(past_reward)

    for i in range(0, len(future_emb[0])):
        future_emb_i = future_emb[0][i]
        future_reward = model.compute_regularized_reward(state_embedding=future_emb_i)
        future_rewards.append(future_reward)

jnp.save(f"past_rewards_{episode}.npy", jnp.array(past_rewards))
jnp.save(f"future_rewards_{episode}.npy", jnp.array(future_rewards))
