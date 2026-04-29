import os
import cv2
from pathlib import Path
from tqdm import tqdm

def prepare_dataset():
    GT_DIR = 'ground_truth'
    INPUT_DIR = 'input'
    SCALE = 4

    os.makedirs(GT_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)

    formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    gt_files = [f for f in Path(GT_DIR).iterdir() if f.suffix.lower() in formats]

    if not gt_files:
        print(f"⚠️ No high-res images found in '{GT_DIR}'.")
        print(f"Please place some high-resolution photos in the '{GT_DIR}' folder first.")
        return

    print(f"Found {len(gt_files)} images in '{GT_DIR}'. Resizing to maintain aspect ratio with a max dimension...")
    
    # Clear input directory
    for f in Path(INPUT_DIR).iterdir():
        if f.is_file():
            f.unlink()

    for gt_path in tqdm(gt_files, desc="Downscaling"):
        img = cv2.imread(str(gt_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # Calculate new dimensions preserving aspect ratio
        max_dim = max(h, w)
        if max_dim > 768:
            scale = 768 / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
        else:
            new_w, new_h = w, h
        
        # Downscale using High-Quality Bicubic Interpolation
        if (new_w, new_h) != (w, h):
            lr_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            lr_img = img
        
        # Save to input folder
        out_path = Path(INPUT_DIR) / gt_path.name
        cv2.imwrite(str(out_path), lr_img)

    print(f"\n✅ Created {len(gt_files)} low-res testing images in '{INPUT_DIR}'!")
    print("\nYou can now run the upscaler:")
    print("  python span_upscale.py")
    print("\nAnd finally evaluate the results:")
    print("  python evaluate.py")

if __name__ == '__main__':
    prepare_dataset()
