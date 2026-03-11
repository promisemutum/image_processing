import cv2
import sys
import numpy as np
from pathlib import Path

def create_comparison(input_name: str):
    """Create a side-by-side comparison of the original image and all models."""
    input_path = Path('input') / input_name
    if not input_path.exists():
        print(f"✗ Input image not found: {input_path}")
        return
        
    img_orig = cv2.imread(str(input_path))
    if img_orig is None:
        print("✗ Failed to load input image")
        return
        
    outputs = []
    labels = []
    
    # Find all outputs for this image
    output_files = sorted(Path('output').glob(f'*_{input_name}'))
    if not output_files:
        print(f"✗ No output images found for {input_name}")
        return

    # Upscale the original to match the 4x resolution using Nearest Neighbor
    first_out = cv2.imread(str(output_files[0]))
    h_out, w_out = first_out.shape[:2]
    
    img_orig_scaled = cv2.resize(img_orig, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
    outputs.append(img_orig_scaled)
    labels.append("Original (Nearest 4x)")
    
    for output_file in output_files:
        model_name = output_file.stem.replace(f'_{input_path.stem}', '')
        img_out = cv2.imread(str(output_file))
        if img_out is not None:
            if img_out.shape[:2] != (h_out, w_out):
                img_out = cv2.resize(img_out, (w_out, h_out))
            outputs.append(img_out)
            labels.append(model_name)
            
    # Add text labels to each image
    labeled_outputs = []
    for img, label in zip(outputs, labels):
        img_copy = img.copy()
        # Add a black background for text for readability
        cv2.rectangle(img_copy, (0, 0), (img_copy.shape[1], 40), (0,0,0), -1)
        cv2.putText(img_copy, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        labeled_outputs.append(img_copy)
        
    # Combine horizontally
    comparison = np.hstack(labeled_outputs)
    
    # Scale down the combined image if it's too large for standard viewing
    MAX_WIDTH = 3840 # typical 4k width
    if comparison.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / comparison.shape[1]
        comparison = cv2.resize(comparison, (MAX_WIDTH, int(comparison.shape[0] * scale)))
        
    save_path = f'comparison_{input_path.stem}.jpg'
    cv2.imwrite(save_path, comparison)
    print(f"\n✓ Comparison generated successfully!")
    print(f"  Includes: {', '.join(labels)}")
    print(f"  Saved as: {save_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        create_comparison(sys.argv[1])
    else:
        # Auto-find first input image
        formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        input_files = [f for f in Path('input').iterdir() if f.suffix.lower() in formats]
        if input_files:
            create_comparison(input_files[0].name)
        else:
            print("No input images found in standard formats.")