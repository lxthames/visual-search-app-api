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
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
from typing import List
from app.core.config import settings

class Vectorizer:
    def __init__(self) -> None:
        if settings.DEVICE == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = settings.DEVICE

        self.model_name = "facebook/dinov2-base"
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

    def get_image_embedding(self, image: Image.Image) -> List[float]:
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

        return embedding.cpu().numpy().flatten().tolist()


# IMPORTANT: module-level singleton, nothing else
_vectorizer_instance: Vectorizer | None = None

def get_vectorizer() -> Vectorizer:
    global _vectorizer_instance
    if _vectorizer_instance is None:
        _vectorizer_instance = Vectorizer()
    return _vectorizer_instance

