import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import logging
from src.utils.batch import BatchedData
from einops import rearrange
from src.utils.logging import get_logger

logger = get_logger(__name__)
descriptor_sizes = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class AENet(pl.LightningModule):
    """
    AutoEncoder Network based on DINOv2 models.

    Attributes:
        model_name (str): Name of the DINOv2 model.
        dinov2_model: The DINOv2 model instance.
        descriptor_size (int): Size of the descriptor.
        max_batch_size (int): Maximum batch size for processing.
        patch_size (int): Size of the image patches.
    """
    def __init__(
        self,
        model_name,
        dinov2_model,
        descriptor_size,
        max_batch_size,
        patch_size=14,
        **kwargs,
    ):
        '''
        Init AENet
        
        Args:
            model_name (str): Name of the DINOv2 model to use.
            dinov2_model: The DINOv2 model instance.
            descriptor_size (int): Size of the descriptor.
            max_batch_size (int): Maximum batch size for processing.
            patch_size (int): Size of the image patches. Default is 14.
            **kwargs: Additional keyword arguments.
        '''

        super().__init__()
        self.model_name = model_name
        self.dinov2_model = dinov2_model
        self.descriptor_size = descriptor_size
        self.max_batch_size = max_batch_size
        self.patch_size = patch_size
        logger.info("Initialize AENet done!")

    def reset_with_pretrained_weights(self):
        device = self.device
        self.dinov2_model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.dinov2_model = self.dinov2_model.to(device)

    def get_toUpdate_parameters(self):
        """
        Get the parameters of the DINOv2 model that should be updated during training.

        Returns:
            iterator: An iterator over the model parameters.
        """

        return self.dinov2_model.parameters()

    def compute_features(self, images):
        """
        Compute features using the DINOv2 model.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            dict: A dictionary containing the computed features.
        """

        # with torch.no_grad():  # no gradients
        features = self.dinov2_model.forward_features(images)
        return features

    def reshape_local_features(self, local_features, num_patches):
        """
        Reshape local features into a 2D spatial arrangement.

        Args:
            local_features (torch.Tensor): Local features to reshape.
            num_patches (list): Number of patches in height and width.

        Returns:
            torch.Tensor: Reshaped local features.
        """

        local_features = rearrange(
            local_features, "b (h w) c -> b c h w", h=num_patches[0], w=num_patches[1]
        )
        return local_features

    def forward_by_chunk(self, processed_rgbs, patch_dim=[2, 3]):
        """
        Process input images in chunks to compute features.

        Args:
            processed_rgbs (torch.Tensor): Processed RGB images.
            patch_dim (list): Dimensions for patch extraction. Default is [2, 3].

        Returns:
            torch.Tensor: Normalized patch features.
        """
        
        batch_rgbs = BatchedData(batch_size=self.max_batch_size, data=processed_rgbs)
        patch_features = BatchedData(batch_size=self.max_batch_size)

        num_patche_h = processed_rgbs.shape[patch_dim[0]] // self.patch_size
        num_patche_w = processed_rgbs.shape[patch_dim[1]] // self.patch_size

        for idx_sample in range(len(batch_rgbs)):
            feats = self.compute_features(batch_rgbs[idx_sample])
            patch_feats = self.reshape_local_features(
                feats["x_prenorm"][:, 1:, :],
                num_patches=[num_patche_h, num_patche_w],
            )
            patch_features.cat(patch_feats)
        return F.normalize(patch_features.data, dim=1)

    def forward(self, images):
        features = self.forward_by_chunk(images)
        return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
    from PIL import Image
    from hydra.experimental import compose, initialize

    with initialize(config_path="../../../configs/"):
        cfg = compose(config_name="train.yaml")

    model = instantiate(cfg.model.ae_net)
    model = model.to("cuda")

    images = torch.rand(2, 3, 224, 224).to(device="cuda")
    features = model(images)
    print(features.shape)
