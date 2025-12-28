"""
Query augmentation and multi-view generation for improved search accuracy.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from typing import List, Tuple, Optional
import cv2
import logging
import random

logger = logging.getLogger(__name__)


class QueryAugmenter:
    """Generate multiple views/augmentations of query images for robust matching."""
    
    def __init__(self):
        """Initialize query augmenter."""
        pass
    
    def generate_rotations(self, img: Image.Image, angles: List[int] = [90, 180, 270]) -> List[Image.Image]:
        """
        Generate rotated versions of the image.
        
        Args:
            img: Input image
            angles: List of rotation angles in degrees
            
        Returns:
            List of rotated images
        """
        views = [img]
        for angle in angles:
            rotated = img.rotate(angle, expand=True)
            views.append(rotated)
        return views
    
    def generate_crops(self, img: Image.Image, crop_ratios: List[Tuple[float, float, float, float]] = None) -> List[Image.Image]:
        """
        Generate cropped views focusing on different regions.
        
        Args:
            img: Input image
            crop_ratios: List of (x1, y1, x2, y2) ratios (0-1)
            
        Returns:
            List of cropped images
        """
        if crop_ratios is None:
            # Default: center crop, top crop, bottom crop, left crop, right crop
            w, h = img.size
            crop_ratios = [
                (0.1, 0.1, 0.9, 0.9),  # Center crop (80% of image)
                (0.1, 0.1, 0.9, 0.5),  # Top half
                (0.1, 0.5, 0.9, 0.9),  # Bottom half
                (0.1, 0.1, 0.5, 0.9),  # Left half
                (0.5, 0.1, 0.9, 0.9),  # Right half
            ]
        
        views = []
        w, h = img.size
        for x1_ratio, y1_ratio, x2_ratio, y2_ratio in crop_ratios:
            x1 = int(w * x1_ratio)
            y1 = int(h * y1_ratio)
            x2 = int(w * x2_ratio)
            y2 = int(h * y2_ratio)
            crop = img.crop((x1, y1, x2, y2))
            if crop.size[0] > 50 and crop.size[1] > 50:  # Only keep reasonable crops
                views.append(crop)
        
        return views
    
    def generate_augmentations(self, img: Image.Image) -> List[Image.Image]:
        """
        Generate augmented versions with brightness/contrast variations.
        
        Args:
            img: Input image
            
        Returns:
            List of augmented images
        """
        views = [img]
        
        # Brightness variations
        for factor in [0.9, 1.1]:
            enhancer = ImageEnhance.Brightness(img)
            views.append(enhancer.enhance(factor))
        
        # Contrast variations
        for factor in [0.9, 1.1]:
            enhancer = ImageEnhance.Contrast(img)
            views.append(enhancer.enhance(factor))
        
        # Slight rotations (±5 degrees)
        for angle in [-5, 5]:
            rotated = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))
            views.append(rotated)
        
        return views
    
    def generate_multi_views(
        self, 
        img: Image.Image,
        include_rotations: bool = True,
        include_crops: bool = True,
        include_augmentations: bool = False,
        max_views: int = 10
    ) -> List[Image.Image]:
        """
        Generate comprehensive multi-view set for robust matching.
        
        Args:
            img: Input image
            include_rotations: Include rotated views
            include_crops: Include cropped regions
            include_augmentations: Include brightness/contrast variations
            max_views: Maximum number of views to generate
            
        Returns:
            List of view images
        """
        views = [img]  # Always include original
        
        if include_rotations:
            rotated = self.generate_rotations(img, angles=[90, 180, 270])
            views.extend(rotated)
        
        if include_crops:
            # Focus on center and logo region (if known)
            crops = self.generate_crops(img)
            views.extend(crops[:3])  # Limit to top 3 crops
        
        if include_augmentations and len(views) < max_views:
            augs = self.generate_augmentations(img)
            views.extend(augs[:max_views - len(views)])
        
        # Limit total views
        return views[:max_views]
    
    def generate_smart_views(
        self,
        img: Image.Image,
        logo_bbox: List[int] | None = None
    ) -> List[Image.Image]:
        """
        Generate smart views focusing on important regions.
        
        Args:
            img: Input image
            logo_bbox: Optional logo bounding box [x1, y1, x2, y2]
            
        Returns:
            List of view images
        """
        views = [img]  # Original
        
        w, h = img.size
        
        # 1. Full image rotations
        views.extend(self.generate_rotations(img, angles=[90, 180, 270]))
        
        # 2. Center crop (focus on product)
        center_crop = img.crop((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)))
        views.append(center_crop)
        
        # 3. Logo region crop (if available)
        if logo_bbox and len(logo_bbox) == 4:
            x1, y1, x2, y2 = logo_bbox
            # Expand logo region by 50% for context
            expand_w = (x2 - x1) * 0.5
            expand_h = (y2 - y1) * 0.5
            logo_crop = img.crop((
                max(0, int(x1 - expand_w)),
                max(0, int(y1 - expand_h)),
                min(w, int(x2 + expand_w)),
                min(h, int(y2 + expand_h))
            ))
            if logo_crop.size[0] > 50 and logo_crop.size[1] > 50:
                views.append(logo_crop)
        
        # 4. Top half (often contains logo/brand)
        top_half = img.crop((0, 0, w, int(h * 0.6)))
        views.append(top_half)
        
        # 5. Bottom half (often contains product info)
        bottom_half = img.crop((0, int(h * 0.4), w, h))
        views.append(bottom_half)
        
        return views

    def generate_perspective_transforms(self, img: Image.Image, num_transforms: int = 2) -> List[Image.Image]:
        """
        Generate perspective-transformed views to simulate different camera angles.
        
        Args:
            img: Input image
            num_transforms: Number of perspective transforms to generate
        
        Returns:
            List of transformed images
        """
        views = []
        try:
            # Convert PIL Image to numpy array (ensure RGB format)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            # Validate image array
            if img_array.size == 0 or len(img_array.shape) < 2:
                logger.warning("Invalid image for perspective transform")
                return []
            # Handle grayscale or other formats
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            h, w = img_array.shape[:2]
            if h < 10 or w < 10:
                logger.warning("Image too small for perspective transform")
                return []
        except Exception as e:
            logger.warning(f"Image conversion failed in perspective transform: {e}")
            return []
        
        for _ in range(num_transforms):
            # Define source points (corners of image)
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Define destination points with slight perspective distortion
            # Randomly shift corners by up to 10% of image size
            shift = min(w, h) * 0.1
            dst_points = np.float32([
                [random.uniform(-shift, shift), random.uniform(-shift, shift)],
                [w + random.uniform(-shift, shift), random.uniform(-shift, shift)],
                [w + random.uniform(-shift, shift), h + random.uniform(-shift, shift)],
                [random.uniform(-shift, shift), h + random.uniform(-shift, shift)]
            ])
            
            # Compute perspective transform matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            
            # Apply perspective transform
            transformed = cv2.warpPerspective(img_array, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
            views.append(Image.fromarray(transformed))
        
        return views

    def generate_color_jitter(self, img: Image.Image, num_variations: int = 3) -> List[Image.Image]:
        """
        Generate color-jittered versions with random brightness, contrast, saturation, hue.
        
        Args:
            img: Input image
            num_variations: Number of color variations to generate
        
        Returns:
            List of color-jittered images
        """
        views = []
        
        for _ in range(num_variations):
            # Random brightness (0.8 to 1.2)
            brightness = random.uniform(0.8, 1.2)
            img_bright = ImageEnhance.Brightness(img).enhance(brightness)
            
            # Random contrast (0.8 to 1.2)
            contrast = random.uniform(0.8, 1.2)
            img_contrast = ImageEnhance.Contrast(img_bright).enhance(contrast)
            
            # Random saturation (0.8 to 1.2)
            saturation = random.uniform(0.8, 1.2)
            img_sat = ImageEnhance.Color(img_contrast).enhance(saturation)
            
            views.append(img_sat)
        
        return views

    def generate_gaussian_blur(self, img: Image.Image, num_variations: int = 2) -> List[Image.Image]:
        """
        Generate blurred versions to handle motion blur in shelf images.
        
        Args:
            img: Input image
            num_variations: Number of blur variations
        
        Returns:
            List of blurred images
        """
        views = []
        img_array = np.array(img)
        
        # Different blur levels
        blur_sizes = [3, 5]  # Kernel sizes
        
        for blur_size in blur_sizes[:num_variations]:
            blurred = cv2.GaussianBlur(img_array, (blur_size, blur_size), 0)
            views.append(Image.fromarray(blurred))
        
        return views

    def generate_mixup(self, img: Image.Image, background_color: Tuple[int, int, int] = (255, 255, 255), 
                       mixup_alpha: float = 0.3) -> List[Image.Image]:
        """
        Generate MixUp augmentation by blending with background color (simulates shelf conditions).
        
        Args:
            img: Input image
            background_color: Background color to blend with (RGB)
            mixup_alpha: Mixing factor (0.0 = original, 1.0 = full background)
        
        Returns:
            List of mixup images
        """
        views = []
        img_array = np.array(img)
        
        # Create background image
        background = np.full_like(img_array, background_color, dtype=np.uint8)
        
        # Blend images
        mixed = (1 - mixup_alpha) * img_array.astype(np.float32) + mixup_alpha * background.astype(np.float32)
        mixed = np.clip(mixed, 0, 255).astype(np.uint8)
        
        views.append(Image.fromarray(mixed))
        return views

    def generate_elastic_deformation(self, img: Image.Image, alpha: int = 100, sigma: int = 10) -> List[Image.Image]:
        """
        Generate elastically deformed versions to simulate product deformation.
        
        Args:
            img: Input image
            alpha: Deformation strength
            sigma: Smoothness of deformation
        
        Returns:
            List of deformed images
        """
        try:
            from scipy.ndimage import map_coordinates, gaussian_filter
            
            views = []
            img_array = np.array(img)
            h, w = img_array.shape[:2]
            
            # Generate random displacement fields
            dx = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((np.random.rand(h, w) * 2 - 1), sigma) * alpha
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            x_new = x + dx
            y_new = y + dy
            
            # Apply deformation to each channel
            if len(img_array.shape) == 3:
                deformed = np.zeros_like(img_array)
                for i in range(img_array.shape[2]):
                    deformed[:, :, i] = map_coordinates(img_array[:, :, i], [y_new, x_new], order=1, mode='reflect')
            else:
                deformed = map_coordinates(img_array, [y_new, x_new], order=1, mode='reflect')
            
            views.append(Image.fromarray(deformed.astype(np.uint8)))
            return views
        except ImportError:
            logger.warning("scipy not available, skipping elastic deformation")
            return []

    def generate_advanced_augmentations(
        self,
        img: Image.Image,
        include_perspective: bool = True,
        include_color_jitter: bool = True,
        include_blur: bool = True,
        include_mixup: bool = True,
        include_elastic: bool = False,
        max_augmentations: int = 5
    ) -> List[Image.Image]:
        """
        Generate advanced augmentations for robust matching.
        
        Args:
            img: Input image
            include_perspective: Include perspective transforms
            include_color_jitter: Include color jittering
            include_blur: Include Gaussian blur
            include_mixup: Include MixUp with background
            include_elastic: Include elastic deformation (requires scipy)
            max_augmentations: Maximum number of augmentations to generate
        
        Returns:
            List of augmented images
        """
        views = []
        
        if include_perspective and len(views) < max_augmentations:
            perspective_views = self.generate_perspective_transforms(img, num_transforms=2)
            views.extend(perspective_views[:max_augmentations - len(views)])
        
        if include_color_jitter and len(views) < max_augmentations:
            jitter_views = self.generate_color_jitter(img, num_variations=2)
            views.extend(jitter_views[:max_augmentations - len(views)])
        
        if include_blur and len(views) < max_augmentations:
            blur_views = self.generate_gaussian_blur(img, num_variations=1)
            views.extend(blur_views[:max_augmentations - len(views)])
        
        if include_mixup and len(views) < max_augmentations:
            mixup_views = self.generate_mixup(img, mixup_alpha=0.2)
            views.extend(mixup_views[:max_augmentations - len(views)])
        
        if include_elastic and len(views) < max_augmentations:
            elastic_views = self.generate_elastic_deformation(img)
            views.extend(elastic_views[:max_augmentations - len(views)])
        
        return views[:max_augmentations]

    def generate_enhanced_smart_views(
        self,
        img: Image.Image,
        logo_bbox: List[int] | None = None,
        include_advanced_aug: bool = True,
        max_views: int = 12
    ) -> List[Image.Image]:
        """
        Generate enhanced smart views with advanced augmentations.
        
        Args:
            img: Input image
            logo_bbox: Optional logo bounding box [x1, y1, x2, y2]
            include_advanced_aug: Include advanced augmentations
            max_views: Maximum number of views to generate
        
        Returns:
            List of view images
        """
        views = []
        
        # Start with basic smart views
        basic_views = self.generate_smart_views(img, logo_bbox)
        views.extend(basic_views)
        
        # Add advanced augmentations if requested
        if include_advanced_aug and len(views) < max_views:
            advanced_views = self.generate_advanced_augmentations(
                img,
                include_perspective=True,
                include_color_jitter=True,
                include_blur=True,
                include_mixup=True,
                max_augmentations=max_views - len(views)
            )
            views.extend(advanced_views)
        
        return views[:max_views]


# Singleton instance
_augmenter_instance: QueryAugmenter | None = None


def get_query_augmenter() -> QueryAugmenter:
    """Get singleton query augmenter."""
    global _augmenter_instance
    if _augmenter_instance is None:
        _augmenter_instance = QueryAugmenter()
    return _augmenter_instance

