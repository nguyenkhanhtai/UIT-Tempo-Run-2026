from abc import ABC, abstractmethod
import numpy as np

class BaseEmbedding(ABC):
    @abstractmethod
    def encode_images(self, pil_images: list, batch_size: int = 64) -> np.ndarray:
        pass

    @abstractmethod
    def encode_texts(self, texts: list, batch_size: int = 256) -> np.ndarray:
        pass
