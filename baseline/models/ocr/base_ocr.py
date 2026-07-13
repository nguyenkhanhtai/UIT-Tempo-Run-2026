from abc import ABC, abstractmethod

class BaseOCR(ABC):
    @abstractmethod
    def extract(self, imgs: list):
        """
        Takes a list of images (numpy arrays or PIL images).
        Returns a list of strings (the extracted text for each image).
        """
        pass
