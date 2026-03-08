# compare.py
import cv2
import numpy as np
from pathlib import Path

def create_comparison(input_path: str, output_path: str, save_path: str):
    """Create side-by-side comparison image"""
    img1 = cv2.imread(input_path)
    img2 = cv2.imread(output_path)
    
    if img1 is None or img2 is None:
        print("✗ Failed to load images")
        return
    
    # Resize output to match input height for comparison
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Scale down output to match input height
    scale = h1 / h2
    new_w = int(w2 * scale)
    img2_resized = cv2.resize(img2, (new_w, h1))
    
    # Add labels
    label1 = cv2.putText(img1.copy(), 'ORIGINAL', (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    label2 = cv2.putText(img2_resized.copy(), 'UPSCALED 4x', (10, 30), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Combine side by side
    comparison = np.hstack([label1, label2])
    cv2.imwrite(save_path, comparison)
    print(f"✓ Comparison saved: {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        create_comparison(sys.argv[1], sys.argv[2], 'comparison.jpg')
    else:
        # Auto-find latest images
        input_files = list(Path('input').glob('*.jpg'))
        output_files = list(Path('output').glob('*.jpg'))
        if input_files and output_files:
            create_comparison(str(input_files[0]), str(output_files[0]), 'comparison.jpg')