import torch
from torch.utils.data import Dataset

class ArrowDataset(Dataset):
    def __init__(self, images, angles, transform=None):
        """
        images: numpy array of shape (N, H, W, 1)
        angles: numpy array of shape (N,)
        transform: optional torchvision transforms
        """
        self.images = images
        self.angles = angles
        self.transform = transform

    def __len__(self):
        return len(self.angles)

    def __getitem__(self, idx):
        image = self.images[idx]
        angle = self.angles[idx]

        # Convert to float32 tensor and move channel first
        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)  # (C,H,W)

        # Convert target to float32 tensor
        angle = torch.tensor(angle, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, angle