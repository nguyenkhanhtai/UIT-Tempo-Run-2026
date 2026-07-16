from abc import ABC, abstractmethod

class BaseCaption(ABC):
    @abstractmethod
    def extract(self, imgs: list, batch_size: int = 8, **kwargs):
        pass
