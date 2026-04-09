import os
import random
from typing import Tuple, List, Optional
import torch
from PIL import Image
from torchvision.transforms import ToTensor


class BatchImagePairLoader:
    """

    Attributes:
        data_dir: Directory containing fisheye (distorted) images
        label_dir: Directory containing rectified (ground truth) images
        height: Target height for image resizing
        width: Target width for image resizing
        batch_size: Number of image pairs per batch
        shuffle: Whether to shuffle the dataset
        target_channel: Number of color channels (default: 3 for RGB)
    """
    
    def __init__(
        self, 
        data_dir: str, 
        label_dir: str, 
        height: int, 
        width: int, 
        batch_size: int = 32, 
        shuffle: bool = True, 
        target_channel: int = 3
    ):
        """
        Initialize the BatchImagePairLoader.
        
        Args:
            data_dir: Path to directory containing fisheye images
            label_dir: Path to directory containing rectified images
            height: Target height for resizing images
            width: Target width for resizing images
            batch_size: Number of images per batch
            shuffle: Whether to shuffle the data
            target_channel: Number of channels in output images
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.height = height
        self.width = width
        self.target_channel = target_channel
        
        # Get sorted lists of image files
        self.data_files = self._get_image_files(data_dir)
        self.label_files = self._get_image_files(label_dir)
        
        # Validate that both directories have the same number of files
        if len(self.data_files) != len(self.label_files):
            raise ValueError(
                f"Mismatch in number of files: "
                f"{len(self.data_files)} fisheye images vs "
                f"{len(self.label_files)} rectified images"
            )
        
        # Create indices for shuffling
        self.indexes = list(range(len(self.data_files)))
        if self.shuffle:
            random.shuffle(self.indexes)
        
        # Initialize transform (simple tensor conversion)
        self.transform = ToTensor()
    
    def _get_image_files(self, directory: str) -> List[str]:
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        files = [
            os.path.join(directory, f) 
            for f in sorted(os.listdir(directory)) 
            if f.lower().endswith(valid_extensions)
        ]
        return files
    
    def __len__(self) -> int:
        return len(self.data_files) // self.batch_size
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Get indices for this batch
        batch_indexes = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        
        # Get file paths for this batch
        batch_data_files = [self.data_files[i] for i in batch_indexes]
        batch_label_files = [self.label_files[i] for i in batch_indexes]
        
        # Load and process images
        batch_data = []
        batch_labels = []
        
        for data_file, label_file in zip(batch_data_files, batch_label_files):
            # Load images
            data_img = Image.open(data_file).convert('RGB')
            label_img = Image.open(label_file).convert('RGB')
            
            # Resize images
            data_img = data_img.resize((self.width, self.height), Image.BILINEAR)
            label_img = label_img.resize((self.width, self.height), Image.BILINEAR)
            
            # Convert to tensors
            data_tensor = self.transform(data_img)
            label_tensor = self.transform(label_img)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                data_tensor = data_tensor.cuda()
                label_tensor = label_tensor.cuda()
            
            batch_data.append(data_tensor)
            batch_labels.append(label_tensor)
        
        # Stack into batch tensors
        batch_data = torch.stack(batch_data)
        batch_labels = torch.stack(batch_labels)
        
        return batch_data, batch_labels
    
    def get_sample_batch(self, num_samples: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
        # Temporarily set batch size
        original_batch_size = self.batch_size
        self.batch_size = num_samples
        
        # Get sample batch
        sample_batch = self[0]
        
        # Restore original batch size
        self.batch_size = original_batch_size
        
        return sample_batch
    
    def reshuffle(self) -> None:
        if self.shuffle:
            random.shuffle(self.indexes)