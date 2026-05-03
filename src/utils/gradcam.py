import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


class CLIPGradCAM:
    """Attention-based heatmap visualization for CLIP ViT.

    Uses the self-attention map from the last transformer layer.
    The [CLS] token's attention to all patch tokens is averaged over heads
    to produce a spatial heatmap.
    """

    def __init__(self, clip_wrapper):
        self.model = clip_wrapper.model
        self.device = clip_wrapper.device
        self.processor = clip_wrapper.processor
        self.vision_model = self.model.vision_model
        # Set attention implementation to 'eager' so output_attentions works
        self.vision_model.config._attn_implementation = "eager"

    def generate_heatmap(self, pixel_values):
        """Generate attention heatmap from the last ViT layer.

        Args:
            pixel_values: (1, C, H, W) tensor

        Returns:
            heatmap: (H_patches, W_patches) numpy array in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            output = self.vision_model(
                pixel_values=pixel_values.to(self.device),
                output_attentions=True,
            )

        # attentions is a tuple of (layer_0_attn, ..., layer_N_attn)
        # Each has shape: (batch, heads, seq_len, seq_len)
        last_attn = output.attentions[-1]  # (1, 12, 197, 197) for ViT-B/16
        # Average over heads
        attn = last_attn[0].mean(dim=0)  # (197, 197)
        # CLS token's attention to patches (exclude CLS self-attention)
        cls_attn = attn[0, 1:]  # (196,)
        num_patches = int(np.sqrt(cls_attn.shape[0]))  # 14

        # Reshape to 2D spatial grid
        heatmap = cls_attn.reshape(num_patches, num_patches).cpu().numpy()

        # Normalize to [0, 1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def visualize(self, pixel_values, image_path, save_path=None):
        """Generate and overlay heatmap on the original image.

        Args:
            pixel_values: (1, C, H, W) tensor
            image_path: path to the original image for overlay
            save_path: optional path to save the figure

        Returns:
            fig: matplotlib figure (if save_path is None, caller should close it)
        """
        heatmap = self.generate_heatmap(pixel_values)

        # Load original image
        img = Image.open(image_path).convert("RGB")

        # Resize heatmap to match original image
        heatmap_img = Image.fromarray(np.uint8(heatmap * 255)).resize(
            img.size, Image.BICUBIC
        )
        heatmap_resized = np.array(heatmap_img)

        # Create overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(heatmap_resized, cmap="jet", alpha=0.7)
        axes[1].set_title("Attention Heatmap")
        axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].imshow(heatmap_resized, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay")
        axes[2].axis("off")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close()
        return fig
