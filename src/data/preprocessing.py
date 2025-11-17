"""
Image preprocessing and augmentation pipeline.

Implements:
1. ImageNet normalization for transfer learning
2. Data augmentation (rotation, scaling, color jittering)
3. On-device augmentation techniques
"""

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
from PIL import Image
import random


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class RandomCrop:
    """
    Random crop operation for on-device augmentation.

    Reference: Equation (11) in the paper
    I_crop(x,y) = I(x + x0, y + y0)
    """
    def __init__(self, size, padding=None):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.padding = padding

    def __call__(self, img):
        if self.padding:
            img = TF.pad(img, self.padding)

        w, h = img.size
        th, tw = self.size

        if w == tw and h == th:
            return img

        # Random coordinates (x0, y0)
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        return TF.crop(img, i, j, th, tw)


class RandomHorizontalFlip:
    """
    Random horizontal flip operation.

    Reference: Equation (12) in the paper
    I_flip(x,y) = I(x, W - y)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return TF.hflip(img)
        return img


class ColorJitter:
    """
    Color jittering transformation.

    Reference: Equation (13) in the paper
    I_jitter(x,y) = min(max(I(x,y) + δ, 0), 255)
    where δ ∈ [-Δ, Δ] and Δ = 25.5 (10% intensity range)
    """
    def __init__(self, brightness=0.1, contrast=0.1, saturation=0.1):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        # Apply color jittering
        fn_idx = torch.randperm(3)

        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness_factor = random.uniform(
                    max(0, 1 - self.brightness),
                    1 + self.brightness
                )
                img = TF.adjust_brightness(img, brightness_factor)

            elif fn_id == 1 and self.contrast is not None:
                contrast_factor = random.uniform(
                    max(0, 1 - self.contrast),
                    1 + self.contrast
                )
                img = TF.adjust_contrast(img, contrast_factor)

            elif fn_id == 2 and self.saturation is not None:
                saturation_factor = random.uniform(
                    max(0, 1 - self.saturation),
                    1 + self.saturation
                )
                img = TF.adjust_saturation(img, saturation_factor)

        return img


def get_train_transforms(image_size=224, augment=True):
    """
    Get training image transformations.

    Args:
        image_size: Target image size
        augment: Whether to apply data augmentation

    Returns:
        Composed transforms
    """
    transform_list = []

    if augment:
        # Data augmentation optimized for entry-level smartphone photography
        transform_list.extend([
            transforms.Resize(256),
            RandomCrop(image_size),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(15),
        ])
    else:
        transform_list.extend([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
        ])

    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    return transforms.Compose(transform_list)


def get_val_transforms(image_size=224):
    """
    Get validation/test image transformations.

    Args:
        image_size: Target image size

    Returns:
        Composed transforms
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image_path, image_size=224, for_training=False):
    """
    Preprocess a single image.

    Args:
        image_path: Path to image file or PIL Image
        image_size: Target image size
        for_training: Whether to apply training augmentations

    Returns:
        Preprocessed tensor of shape (1, 3, H, W)
    """
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path.convert('RGB')

    # Get appropriate transforms
    if for_training:
        transform = get_train_transforms(image_size, augment=True)
    else:
        transform = get_val_transforms(image_size)

    # Apply transforms
    tensor = transform(image)

    # Add batch dimension
    return tensor.unsqueeze(0)


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """
    Denormalize image tensor for visualization.

    Args:
        tensor: Normalized tensor of shape (C, H, W) or (B, C, H, W)
        mean: Mean values used for normalization
        std: Std values used for normalization

    Returns:
        Denormalized tensor
    """
    if tensor.dim() == 3:
        # Single image (C, H, W)
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
    else:
        # Batch of images (B, C, H, W)
        for i in range(tensor.size(0)):
            for t, m, s in zip(tensor[i], mean, std):
                t.mul_(s).add_(m)

    return tensor


def tensor_to_image(tensor):
    """
    Convert tensor to PIL Image.

    Args:
        tensor: Tensor of shape (C, H, W)

    Returns:
        PIL Image
    """
    # Denormalize
    tensor = denormalize(tensor.clone())

    # Clip values
    tensor = torch.clamp(tensor, 0, 1)

    # Convert to numpy
    numpy_image = tensor.permute(1, 2, 0).cpu().numpy()

    # Convert to uint8
    numpy_image = (numpy_image * 255).astype(np.uint8)

    # Convert to PIL
    return Image.fromarray(numpy_image)


class OnDeviceAugmentation:
    """
    Lightweight on-device augmentation for mobile deployment.

    Reference: Equations (11), (12), (13) in the paper
    """
    def __init__(self, image_size=224):
        self.image_size = image_size

        # Augmentation parameters
        self.crop_range = (0.8, 1.0)  # Crop scale range
        self.flip_prob = 0.5
        self.color_delta = 25.5  # 10% of 255

    def apply(self, image):
        """
        Apply lightweight augmentations.

        Args:
            image: PIL Image or numpy array

        Returns:
            Augmented image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Random crop
        w, h = image.size
        scale = random.uniform(*self.crop_range)
        new_w, new_h = int(w * scale), int(h * scale)

        if new_w < w or new_h < h:
            i = random.randint(0, h - new_h) if h > new_h else 0
            j = random.randint(0, w - new_w) if w > new_w else 0
            image = TF.crop(image, i, j, new_h, new_w)

        # Resize to target size
        image = TF.resize(image, (self.image_size, self.image_size))

        # Random horizontal flip (Equation 12)
        if random.random() < self.flip_prob:
            image = TF.hflip(image)

        # Color jitter (Equation 13)
        image = np.array(image).astype(np.float32)
        delta = random.uniform(-self.color_delta, self.color_delta)
        image = np.clip(image + delta, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)

        return image


if __name__ == "__main__":
    # Test preprocessing
    print("Testing preprocessing pipeline...")

    # Create a dummy image
    dummy_image = Image.new('RGB', (512, 512), color=(128, 128, 128))

    # Test train transforms
    train_transform = get_train_transforms(224, augment=True)
    train_tensor = train_transform(dummy_image)
    print(f"Train tensor shape: {train_tensor.shape}")
    print(f"Train tensor range: [{train_tensor.min():.3f}, {train_tensor.max():.3f}]")

    # Test val transforms
    val_transform = get_val_transforms(224)
    val_tensor = val_transform(dummy_image)
    print(f"Val tensor shape: {val_tensor.shape}")

    # Test preprocess_image
    preprocessed = preprocess_image(dummy_image, image_size=224, for_training=False)
    print(f"Preprocessed shape: {preprocessed.shape}")

    # Test denormalization
    denorm = denormalize(train_tensor.clone())
    print(f"Denormalized range: [{denorm.min():.3f}, {denorm.max():.3f}]")

    # Test on-device augmentation
    on_device = OnDeviceAugmentation(image_size=224)
    augmented = on_device.apply(dummy_image)
    print(f"On-device augmented size: {augmented.size}")

    print("\nPreprocessing pipeline test completed successfully!")
