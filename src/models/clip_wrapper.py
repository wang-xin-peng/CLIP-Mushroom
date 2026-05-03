import torch
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper:
    """Wrapper around HuggingFace CLIP model for easy inference.

    Provides unified interface for image/text encoding and zero-shot prediction.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch16", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def encode_images(self, pixel_values):
        """Extract normalized image features.

        Args:
            pixel_values: (B, C, H, W) tensor
        Returns:
            (B, D) normalized feature tensor
        """
        output = self.model.get_image_features(pixel_values.to(self.device))
        features = output.pooler_output if hasattr(output, 'pooler_output') else output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def encode_text(self, text_list):
        """Extract normalized text features.

        Args:
            text_list: list of strings
        Returns:
            (N, D) normalized feature tensor
        """
        inputs = self.processor(
            text=text_list, return_tensors="pt", padding=True
        ).to(self.device)
        output = self.model.get_text_features(**inputs)
        features = output.pooler_output if hasattr(output, 'pooler_output') else output
        features = features / features.norm(dim=-1, keepdim=True)
        return features

    @torch.no_grad()
    def zero_shot_predict(self, pixel_values, class_names, template="a photo of {}"):
        """Zero-shot classification via text-image similarity.

        Args:
            pixel_values: (B, C, H, W) image tensor
            class_names: list of class name strings
            template: prompt template string with {} placeholder
        Returns:
            probs: (B, C) probability tensor
            logits: (B, C) logit tensor (cosine sim * logit_scale)
        """
        prompts = [template.format(name) for name in class_names]
        text_features = self.encode_text(prompts)  # (C, D)
        image_features = self.encode_images(pixel_values)  # (B, D)
        logits = image_features @ text_features.T * self.model.logit_scale.exp()
        probs = torch.softmax(logits, dim=-1)
        return probs, logits
