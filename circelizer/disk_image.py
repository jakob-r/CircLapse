"""
Disk-based image storage abstraction.
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import logging
import pickle
import gzip
from typing import Optional

logger = logging.getLogger(__name__)

class DiskImage:
    """
    An image object that stores data on disk but behaves like a regular numpy array.
    Automatically handles loading and saving to disk transparently using pickle with compression.
    """
    
    def __init__(self, image: np.ndarray, filename: str, temp_dir: Path):
        """
        Initialize a DiskImage with an image and store it on disk.
        
        Args:
            image: The image as a numpy array
            filename: Base filename for the image
            temp_dir: Temporary directory to store the image
        """
        self._temp_path = temp_dir / f"{filename}_intermediate.pkl.gz"
        self._filename = filename
        self._temp_dir = temp_dir
        
        # Save the image to disk using pickle with compression
        try:
            with gzip.open(self._temp_path, 'wb') as f:
                pickle.dump(image, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            raise RuntimeError(f"Failed to save image to {self._temp_path}: {repr(e)}")
    
    def load(self) -> np.ndarray:
        """
        Load the image from disk.
        
        Returns:
            The image as a numpy array
        """
        try:
            with gzip.open(self._temp_path, 'rb') as f:
                image = pickle.load(f)
            return image
        except Exception as e:
            raise RuntimeError(f"Failed to load image from {self._temp_path}: {repr(e)}")
    
    def save(self, output_path: Path) -> bool:
        """
        Save the image to a specific output path.
        
        Args:
            output_path: Path where to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image = self.load()
            success = cv2.imwrite(str(output_path), image)
            return success
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {repr(e)}")
            return False
    
    @property
    def filename(self) -> str:
        """Get the base filename."""
        return self._filename
    
    @property
    def temp_path(self) -> Path:
        """Get the temporary file path."""
        return self._temp_path
    
    def __repr__(self) -> str:
        return f"DiskImage(filename='{self._filename}', path='{self._temp_path}')" 