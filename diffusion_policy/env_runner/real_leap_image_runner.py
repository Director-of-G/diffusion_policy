"""
    Modified from real_pusht_image_runner.py, currently do not
    support hardware runner.
"""

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class RealLeapImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()
