import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# ── Label mapping ─────────────────────────────────────────────────────────────
CLASS_NAMES = {0: "With Mask", 1: "Without Mask"}
CLASS_DIRS  = {"with_mask": 0, "without_mask": 1}

# ── Transforms ────────────────────────────────────────────────────────────────
def get_transforms(is_train: bool):
    """
    Training   → augment (flip, color jitter, rotation) to reduce overfitting.
    Validation → only resize + normalize.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

# ── Custom Dataset ────────────────────────────────────────────────────────────
class MaskDataset(Dataset):
    """
    Loads face images from disk. Reads all valid images from both subfolders directly.
    """
    def __init__(self, root_dir: str, transform=None):
        self.transform = transform
        self.samples = []

        for class_name, label in CLASS_DIRS.items():
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(class_path, fname), label))
                        
        print(f"Loaded {len(self.samples)} images from '{root_dir}'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ── Custom Subset Wrapper for Transformations ─────────────────────────────────
class SubsetWrapper(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # Since images in the dataset are already loaded as tensors by `__getitem__`
        # we can just apply transformations here.
        if self.transform:
            image = self.transform(image)
        return image, label


# ── DataLoader factory ────────────────────────────────────────────────────────
def get_dataloaders(data_dir: str = "data", batch_size: int = 32):
    """
    Returns train and validation DataLoaders.
    """
    # Initialize the base dataset from the root data directory
    full_dataset = MaskDataset(root_dir=data_dir, transform=None) 
    
    # Split the dataset 80% train, 20% validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Wrap subsets to apply specific transforms
    train_dataset = SubsetWrapper(train_subset, transform=get_transforms(is_train=True))
    val_dataset = SubsetWrapper(val_subset, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader