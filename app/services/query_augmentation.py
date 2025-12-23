"""
Query augmentation and multi-view generation for improved search accuracy.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from typing import List, Tuple
import logging

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


# Singleton instance
_augmenter_instance: QueryAugmenter | None = None


def get_query_augmenter() -> QueryAugmenter:
    """Get singleton query augmenter."""
    global _augmenter_instance
    if _augmenter_instance is None:
        _augmenter_instance = QueryAugmenter()
    return _augmenter_instance

