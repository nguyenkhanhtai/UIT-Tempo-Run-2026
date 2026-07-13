from abc import ABC, abstractmethod

class BaseOD(ABC):
    @abstractmethod
    def extract(self, imgs: list):
        """
        Takes a list of images (numpy arrays or PIL images).
        Returns a list of lists of strings (the detected object classes for each image).
        """
        pass
