"""
Advanced matching features for robust visual search:
- Color Histogram Analysis
- Geometric Pattern Recognition
- Shape Consistency
"""
from __future__ import annotations

import numpy as np
import cv2
from PIL import Image
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ColorHistogramAnalyzer:
    """Analyze and compare color histograms for robust matching."""
    
    def __init__(self, bins: int = 32):
        """
        Initialize color histogram analyzer.
        
        Args:
            bins: Number of bins per channel (default: 32 for 32^3 = 32768 total bins)
        """
        self.bins = bins
    
    def compute_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute 3D color histogram for an image.
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Normalized histogram (bins^3,)
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB (H, W, 3)")
        
        # Reshape to (N, 3) where N = H*W
        pixels = image.reshape(-1, 3)
        
        # Compute histogram for each channel
        hist_r = np.histogram(pixels[:, 0], bins=self.bins, range=(0, 256))[0]
        hist_g = np.histogram(pixels[:, 1], bins=self.bins, range=(0, 256))[0]
        hist_b = np.histogram(pixels[:, 2], bins=self.bins, range=(0, 256))[0]
        
        # Combine into 3D histogram (flattened)
        # Using outer product to create 3D histogram
        hist_3d = np.outer(hist_r, np.outer(hist_g, hist_b)).flatten()
        
        # Normalize
        hist_3d = hist_3d / (hist_3d.sum() + 1e-10)
        
        return hist_3d
    
    def compute_histogram_cv2(self, image: np.ndarray) -> np.ndarray:
        """
        Compute color histogram using OpenCV (more efficient).
        
        Args:
            image: RGB image array (H, W, 3)
            
        Returns:
            Normalized histogram
        """
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Compute histogram
        hist = cv2.calcHist(
            [image_bgr],
            [0, 1, 2],  # All channels
            None,  # No mask
            [self.bins, self.bins, self.bins],  # Bins per channel
            [0, 256, 0, 256, 0, 256]  # Range for each channel
        )
        
        # Normalize
        hist = hist / (hist.sum() + 1e-10)
        
        return hist.flatten()
    
    def compare_histograms(
        self, 
        hist1: np.ndarray, 
        hist2: np.ndarray, 
        method: str = "correlation"
    ) -> float:
        """
        Compare two histograms using various methods.
        
        Args:
            hist1: First histogram
            hist2: Second histogram
            method: Comparison method ('correlation', 'chi_square', 'intersection', 'bhattacharyya')
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if method == "correlation":
            # Correlation coefficient (0-1, higher is more similar)
            corr = cv2.compareHist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), cv2.HISTCMP_CORREL)
            return max(0.0, corr)  # Ensure non-negative
        
        elif method == "chi_square":
            # Chi-square distance (lower is more similar, so invert)
            chi_sq = cv2.compareHist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), cv2.HISTCMP_CHISQR)
            # Normalize to 0-1 range (inverse relationship)
            return 1.0 / (1.0 + chi_sq / 1000.0)
        
        elif method == "intersection":
            # Histogram intersection (0-1, higher is more similar)
            intersection = cv2.compareHist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), cv2.HISTCMP_INTERSECT)
            # Normalize by min sum
            min_sum = min(hist1.sum(), hist2.sum())
            return intersection / (min_sum + 1e-10)
        
        elif method == "bhattacharyya":
            # Bhattacharyya distance (lower is more similar, so invert)
            bhat = cv2.compareHist(hist1.reshape(-1, 1), hist2.reshape(-1, 1), cv2.HISTCMP_BHATTACHARYYA)
            return 1.0 - bhat  # Invert so higher is more similar
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_color_similarity(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        method: str = "correlation"
    ) -> float:
        """
        Get color similarity between two images.
        
        Args:
            img1: First RGB image
            img2: Second RGB image
            method: Comparison method
            
        Returns:
            Similarity score (0-1)
        """
        try:
            hist1 = self.compute_histogram_cv2(img1)
            hist2 = self.compute_histogram_cv2(img2)
            return self.compare_histograms(hist1, hist2, method)
        except Exception as e:
            logger.warning(f"Color histogram comparison failed: {e}")
            return 0.0


class GeometricPatternRecognizer:
    """Recognize geometric patterns and features for robust matching."""
    
    def __init__(self):
        """Initialize geometric pattern recognizer."""
        # SIFT detector (free alternative to patented SURF)
        self.sift = cv2.SIFT_create(nfeatures=500)
        # ORB as fallback (faster, no license issues)
        self.orb = cv2.ORB_create(nfeatures=500)
        # FLANN matcher for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Brute force matcher for ORB
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract_features(self, image: np.ndarray, use_sift: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract keypoints and descriptors from image.
        
        Args:
            image: RGB image array
            use_sift: Use SIFT (True) or ORB (False)
            
        Returns:
            (keypoints, descriptors) or (None, None) if extraction fails
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        try:
            if use_sift:
                keypoints, descriptors = self.sift.detectAndCompute(gray, None)
            else:
                keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is None or len(keypoints) < 4:
                return None, None
            
            return keypoints, descriptors
        except Exception as e:
            logger.warning(f"Feature extraction failed: {e}")
            return None, None
    
    def match_features(
        self, 
        desc1: np.ndarray, 
        desc2: np.ndarray,
        use_sift: bool = True,
        ratio_threshold: float = 0.75
    ) -> Tuple[int, float]:
        """
        Match features between two descriptors.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            use_sift: Whether using SIFT descriptors
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            (number_of_matches, match_quality_score)
        """
        try:
            if use_sift:
                # Use FLANN for SIFT
                matches = self.flann.knnMatch(desc1, desc2, k=2)
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)
            else:
                # Use brute force for ORB
                matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < ratio_threshold * n.distance:
                            good_matches.append(m)
            
            num_matches = len(good_matches)
            
            # Calculate match quality score (0-1)
            # Based on number of matches and their quality
            max_possible_matches = min(len(desc1), len(desc2))
            if max_possible_matches == 0:
                return 0, 0.0
            
            match_ratio = num_matches / max_possible_matches
            
            # Also consider average match distance (lower is better)
            if good_matches:
                avg_distance = np.mean([m.distance for m in good_matches])
                # Normalize distance (assuming max distance ~300 for SIFT, ~100 for ORB)
                max_dist = 300.0 if use_sift else 100.0
                distance_score = 1.0 - min(1.0, avg_distance / max_dist)
                quality_score = (match_ratio * 0.6 + distance_score * 0.4)
            else:
                quality_score = 0.0
            
            return num_matches, quality_score
            
        except Exception as e:
            logger.warning(f"Feature matching failed: {e}")
            return 0, 0.0
    
    def get_geometric_similarity(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray,
        use_sift: bool = True
    ) -> float:
        """
        Get geometric similarity between two images.
        
        Args:
            img1: First RGB image
            img2: Second RGB image
            use_sift: Use SIFT (True) or ORB (False)
            
        Returns:
            Similarity score (0-1)
        """
        try:
            kp1, desc1 = self.extract_features(img1, use_sift=use_sift)
            kp2, desc2 = self.extract_features(img2, use_sift=use_sift)
            
            if desc1 is None or desc2 is None:
                return 0.0
            
            num_matches, quality_score = self.match_features(desc1, desc2, use_sift=use_sift)
            
            # Require minimum number of matches
            min_matches = 10 if use_sift else 15
            if num_matches < min_matches:
                return 0.0
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"Geometric similarity calculation failed: {e}")
            return 0.0


class ShapeConsistencyChecker:
    """Check shape consistency beyond basic bottle/can classification."""
    
    def __init__(self):
        """Initialize shape consistency checker."""
        pass
    
    def compute_aspect_ratio(self, image: np.ndarray) -> float:
        """
        Compute aspect ratio of image.
        
        Args:
            image: Image array
            
        Returns:
            Aspect ratio (width/height)
        """
        h, w = image.shape[:2]
        return w / (h + 1e-10)  # Avoid division by zero
    
    def compute_compactness(self, image: np.ndarray) -> float:
        """
        Compute compactness (4π*area/perimeter^2).
        Higher values indicate more circular/compact shapes.
        
        Args:
            image: Binary or grayscale image
            
        Returns:
            Compactness score (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        if perimeter == 0:
            return 0.0
        
        # Compactness = 4π*area/perimeter^2
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        
        return min(1.0, compactness)  # Clamp to 0-1
    
    def compute_rectangularity(self, image: np.ndarray) -> float:
        """
        Compute rectangularity (area / bounding_box_area).
        Higher values indicate more rectangular shapes.
        
        Args:
            image: Image array
            
        Returns:
            Rectangularity score (0-1)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Threshold
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
        
        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox_area = w * h
        
        if bbox_area == 0:
            return 0.0
        
        rectangularity = area / bbox_area
        return min(1.0, rectangularity)
    
    def compute_shape_features(self, image: np.ndarray) -> Dict[str, float]:
        """
        Compute all shape features for an image.
        
        Args:
            image: RGB image array
            
        Returns:
            Dictionary of shape features
        """
        return {
            "aspect_ratio": self.compute_aspect_ratio(image),
            "compactness": self.compute_compactness(image),
            "rectangularity": self.compute_rectangularity(image),
        }
    
    def compare_shape_features(
        self, 
        features1: Dict[str, float], 
        features2: Dict[str, float]
    ) -> float:
        """
        Compare shape features between two images.
        
        Args:
            features1: Shape features from first image
            features2: Shape features from second image
            
        Returns:
            Similarity score (0-1)
        """
        # Compute similarity for each feature
        aspect_sim = 1.0 - abs(features1["aspect_ratio"] - features2["aspect_ratio"]) / max(
            features1["aspect_ratio"], features2["aspect_ratio"], 1.0
        )
        
        compact_sim = 1.0 - abs(features1["compactness"] - features2["compactness"])
        rect_sim = 1.0 - abs(features1["rectangularity"] - features2["rectangularity"])
        
        # Weighted average
        # Aspect ratio is most important for distinguishing bottle vs can
        similarity = (
            aspect_sim * 0.5 +
            compact_sim * 0.25 +
            rect_sim * 0.25
        )
        
        return max(0.0, min(1.0, similarity))
    
    def get_shape_consistency(
        self, 
        img1: np.ndarray, 
        img2: np.ndarray
    ) -> float:
        """
        Get shape consistency between two images.
        
        Args:
            img1: First RGB image
            img2: Second RGB image
            
        Returns:
            Similarity score (0-1)
        """
        try:
            features1 = self.compute_shape_features(img1)
            features2 = self.compute_shape_features(img2)
            return self.compare_shape_features(features1, features2)
        except Exception as e:
            logger.warning(f"Shape consistency check failed: {e}")
            return 0.0


# Singleton instances
_color_analyzer: Optional[ColorHistogramAnalyzer] = None
_geometric_recognizer: Optional[GeometricPatternRecognizer] = None
_shape_checker: Optional[ShapeConsistencyChecker] = None


def get_color_analyzer() -> ColorHistogramAnalyzer:
    """Get singleton color histogram analyzer."""
    global _color_analyzer
    if _color_analyzer is None:
        _color_analyzer = ColorHistogramAnalyzer()
    return _color_analyzer


def get_geometric_recognizer() -> GeometricPatternRecognizer:
    """Get singleton geometric pattern recognizer."""
    global _geometric_recognizer
    if _geometric_recognizer is None:
        _geometric_recognizer = GeometricPatternRecognizer()
    return _geometric_recognizer


def get_shape_checker() -> ShapeConsistencyChecker:
    """Get singleton shape consistency checker."""
    global _shape_checker
    if _shape_checker is None:
        _shape_checker = ShapeConsistencyChecker()
    return _shape_checker

