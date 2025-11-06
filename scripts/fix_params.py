# Script to load original Pi0 params and combine them with predictor
from openpi.training import config as _config
import openpi.models.model as _model
from openpi.shared import download
from flax import nnx
import jax
from flax import nnx
import orbax.checkpoint as ocp
import pathlib

jax.config.update('jax_platform_name', 'cpu')

config = _config.get_config("pi0_libero_predictor")
checkpoint_dir = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi0_libero"
)
pali_params = _model.restore_params(checkpoint_dir / "params")
paligemma = pali_params["PaliGemma"]
del pali_params

model = config.model.create(jax.random.key(0))
graphdef, state = nnx.split(model)

state.replace_by_pure_dict({"PaliGemma": paligemma})
model = nnx.merge(graphdef, state)
graphdef, state = nnx.split(model)

checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
ckpt_dir = ocp.test_utils.erase_and_create_empty('/scratch/s5649552/.cache/openpi/openpi-assets/checkpoints/pi0_libero_predictor')
checkpointer.save(ckpt_dir / 'params', {"params": state})

params_path = pathlib.Path(
    "/scratch/s5649552/.cache/openpi/openpi-assets/checkpoints/pi0_libero_predictor/params"
).resolve()
with ocp.PyTreeCheckpointer() as ckptr:
    metadata = ckptr.metadata(params_path)
    
print(metadata['params'].keys())
