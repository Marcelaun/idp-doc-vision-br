"""
Blur Detector usando Laplacian Variance
Não precisa de treinamento ou GPU
"""

import cv2
import numpy as np

class BlurDetector:
    def __init__(self, threshold=100):
        self.threshold = threshold
    
    def detect_blur(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:
            level = "MUITO_BORRADA"
        elif laplacian_var < 100:
            level = "BORRADA"
        elif laplacian_var < 400:
            level = "ACEITAVEL"
        else:
            level = "BOA"
        
        return {
            'is_blurry': laplacian_var < self.threshold,
            'score': float(laplacian_var),
            'level': level
        }
