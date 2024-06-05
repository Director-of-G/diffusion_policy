import numpy as np
import hydra
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)

with hydra.initialize('../diffusion_policy/config'):
    cfg = hydra.compose('train_diffusion_unet_real_image_workspace')
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.task.dataset)

from matplotlib import pyplot as plt
import pdb; pdb.set_trace()
normalizer = dataset.get_normalizer()
nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
dists = np.linalg.norm(nactions, axis=-1)
_ = plt.hist(dists, bins=100); plt.title('real action velocity')
plt.show()
