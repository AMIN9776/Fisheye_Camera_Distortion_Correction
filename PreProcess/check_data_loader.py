import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
from datasets import BatchImagePairLoader
import matplotlib.pyplot as plt

# Parameters
batch_size = 2
data_dir = '/home/nabizadz/Projects/Amin/Fisheye/PreProcess/crop_distorted_train'
label_dir = '/home/nabizadz/Projects/Amin/Fisheye/PreProcess/rectified_train'
height = 256
width = 256
shuffle = True

# Create an instance of the BatchImagePairLoader
dataset = BatchImagePairLoader(data_dir, label_dir, height, width, batch_size=batch_size, shuffle=shuffle)
data_loader = DataLoader(dataset, batch_size=None)

# Function to plot a batch of images
def plot_batch(data_batch, label_batch):
    batch_size = len(data_batch)
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, batch_size * 3))
    for i in range(batch_size):
        # Move tensors to CPU and convert to numpy arrays
        data_image = data_batch[i].cpu().permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        label_image = label_batch[i].cpu().permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        
        # Plot the input data image
        axes[i, 0].imshow(data_image)
        axes[i, 0].set_title("Data Image")
        axes[i, 0].axis("off")
        
        # Plot the corresponding label image
        axes[i, 1].imshow(label_image)
        axes[i, 1].set_title("Label Image")
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    plt.show()

# Iterate through the DataLoader and plot the batches
for batch_idx, (batch_data, batch_labels) in enumerate(data_loader):
    print(f"Batch {batch_idx + 1}:")
    print("Batch data shape:", batch_data.shape)
    print("Batch label shape:", batch_labels.shape)

    # Plot the batch
    plot_batch(batch_data, batch_labels)
    break  # Visualize only the first batch
