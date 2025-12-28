# import torch
# from transformers import CLIPModel, CLIPProcessor
# from PIL import Image
# from typing import List
# from app.core.config import settings

# class Vectorizer:
#     def __init__(self) -> None:
#         # Decide device
#         if settings.DEVICE == "auto":
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = settings.DEVICE

#         self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
#         self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

#     def get_image_embedding(self, image: Image.Image) -> List[float]:
#         """Converts a PIL image into a normalized embedding vector."""
#         inputs = self.processor(images=image, return_tensors="pt").to(self.device)
#         with torch.no_grad():
#             image_features = self.model.get_image_features(**inputs)

#         image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#         return image_features.cpu().numpy().flatten().tolist()

# # Singleton-style accessor so we only load CLIP once
# _vectorizer_instance: Vectorizer | None = None

# def get_vectorizer() -> Vectorizer:
#     global _vectorizer_instance
#     if _vectorizer_instance is None:
#         _vectorizer_instance = Vectorizer()
#     return _vectorizer_instance
import torch
from transformers import AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor
from PIL import Image
from typing import List, Dict, Optional
import numpy as np
from app.core.config import settings

class Vectorizer:
    def __init__(self, use_ensemble: bool = True) -> None:
        if settings.DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.DEVICE

        self.use_ensemble = use_ensemble

        # ✅ DINOv2-Large for structural/geometric features (1024-dim)
        self.dinov2_model_name = "facebook/dinov2-large"
        self.dinov2_processor = AutoImageProcessor.from_pretrained(self.dinov2_model_name)
        self.dinov2_model = AutoModel.from_pretrained(self.dinov2_model_name).to(self.device)
        self.dinov2_model.eval()

        # ✅ CLIP ViT-L/14 for semantic understanding (768-dim)
        if self.use_ensemble:
            self.clip_model_name = "openai/clip-vit-large-patch14"
            self.clip_processor = CLIPProcessor.from_pretrained(self.clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(self.clip_model_name).to(self.device)
            self.clip_model.eval()

    def get_dinov2_embedding(self, image: Image.Image) -> np.ndarray:
        """Get DINOv2 embedding (structural/geometric features)."""
        inputs = self.dinov2_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding (semantic understanding)."""
        if not self.use_ensemble:
            raise ValueError("CLIP embedding requires use_ensemble=True")
        
        inputs = self.clip_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)

        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def get_image_embedding(self, image: Image.Image, return_separate: bool = False) -> List[float] | Dict[str, List[float]]:
        """
        Get embedding for vector database storage/indexing.
        
        NOTE: Always returns DINOv2 embedding (1024-dim) for compatibility with datastore.
        For ensemble search, use get_ensemble_embeddings() to get both DINOv2 and CLIP separately.
        
        Args:
            image: Input PIL image
            return_separate: If True, return dict with separate embeddings; if False, return DINOv2 only
        
        Returns:
            DINOv2 embedding list (1024-dim), or dict with separate embeddings if return_separate=True
        """
        dinov2_emb = self.get_dinov2_embedding(image)
        
        if return_separate:
            result = {"dinov2": dinov2_emb.tolist()}
            if self.use_ensemble:
                clip_emb = self.get_clip_embedding(image)
                result["clip"] = clip_emb.tolist()
            return result
        
        # Always return DINOv2 for storage/indexing (compatible with 1024-dim datastore)
        return dinov2_emb.tolist()
    
    def get_ensemble_embeddings(self, image: Image.Image) -> Dict[str, np.ndarray]:
        """
        Get both DINOv2 and CLIP embeddings separately for ensemble search.
        
        This method is used for querying, where we can combine similarity scores
        from both models without needing to store 1792-dim vectors.
        
        Args:
            image: Input PIL image
        
        Returns:
            Dict with 'dinov2' and 'clip' embeddings (numpy arrays)
        """
        dinov2_emb = self.get_dinov2_embedding(image)
        result = {"dinov2": dinov2_emb}
        if self.use_ensemble:
            clip_emb = self.get_clip_embedding(image)
            result["clip"] = clip_emb
        return result

    def get_text_embedding(self, text: str) -> List[float]:
        """
        Get CLIP text embedding for cross-modal search.
        
        Args:
            text: Input text query
        
        Returns:
            Text embedding vector
        """
        if not self.use_ensemble:
            raise ValueError("Text embedding requires use_ensemble=True")
        
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.clip_model.get_text_features(**inputs)

        embedding = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten().tolist()


# IMPORTANT: module-level singleton, nothing else
_vectorizer_instance: Vectorizer | None = None
_vectorizer_use_ensemble: bool = True

def get_vectorizer(use_ensemble: bool = True) -> Vectorizer:
    """
    Get singleton vectorizer instance.
    
    IMPORTANT: For storage/indexing, always use DINOv2 embeddings (1024-dim).
    The use_ensemble parameter only affects query-time ensemble search, not storage.
    
    Args:
        use_ensemble: Whether to enable CLIP for ensemble search (query-time only)
    
    Returns:
        Vectorizer instance
    """
    global _vectorizer_instance, _vectorizer_use_ensemble
    # Recreate if use_ensemble setting changed
    if _vectorizer_instance is None or _vectorizer_use_ensemble != use_ensemble:
        _vectorizer_instance = Vectorizer(use_ensemble=use_ensemble)
        _vectorizer_use_ensemble = use_ensemble
    return _vectorizer_instance

