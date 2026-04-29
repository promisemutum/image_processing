import sys
import cv2
import numpy as np
from pathlib import Path

MAX_WIDTH = 3840

def draw_label(img: np.ndarray, label: str) -> None:
    """Draws a black background bar and text label on the image in-place."""
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

def create_comparison(input_name: str) -> None:
    input_path = Path('input') / input_name
    if not input_path.exists():
        print(f"Input image not found: {input_path}")
        return
        
    img_orig = cv2.imread(str(input_path))
    if img_orig is None:
        print("Failed to load input image")
        return
        
    output_files = sorted(Path('output').glob(f'*_{input_name}'))
    if not output_files:
        print(f"✗ No output images found for {input_name}")
        return

    # Load first output to determine target dimensions
    first_out = cv2.imread(str(output_files[0]))
    if first_out is None:
        print(f"Failed to load output image: {output_files[0]}")
        return
    h_out, w_out = first_out.shape[:2]
    
    # Upscale the original and draw label
    img_orig_scaled = cv2.resize(img_orig, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
    draw_label(img_orig_scaled, "Original (Nearest 4x)")
    
    outputs = [img_orig_scaled]
    labels = ["Original (Nearest 4x)"]
    
    # Process output images
    for output_file in output_files:
        model_name = output_file.stem.replace(f'_{input_path.stem}', '')
        img_out = cv2.imread(str(output_file))
        
        if img_out is not None:
            if img_out.shape[:2] != (h_out, w_out):
                img_out = cv2.resize(img_out, (w_out, h_out))
            draw_label(img_out, model_name)
            outputs.append(img_out)
            labels.append(model_name)
            
    # Combine horizontally
    comparison = np.hstack(outputs)
    
    # Scale down the combined image if it's too large for standard viewing
    if comparison.shape[1] > MAX_WIDTH:
        scale = MAX_WIDTH / comparison.shape[1]
        comparison = cv2.resize(comparison, (MAX_WIDTH, int(comparison.shape[0] * scale)))
        
    save_path = f'comparison_{input_path.stem}.jpg'
    cv2.imwrite(save_path, comparison)
    print(f"\n Comparison generated!")
    print(f"  Includes: {', '.join(labels)}")
    print(f"  Saved as: {save_path}")

def main() -> None:
    if len(sys.argv) >= 2:
        create_comparison(sys.argv[1])
    else:
        # Auto-find first input image efficiently using a generator
        formats = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
        try:
            input_file = next(f for f in Path('input').iterdir() if f.suffix.lower() in formats)
            create_comparison(input_file.name)
        except StopIteration:
            print("No input images found in standard formats.")

if __name__ == "__main__": 
    main()