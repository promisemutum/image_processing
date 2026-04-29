import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
ort.set_default_logger_severity(3) # Hide ScatterND warning
from pathlib import Path
import time
from tqdm import tqdm

class SPANUpscaler:
    def __init__(self, model_path: str, scale: int = 4):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        providers = ['CUDAExecutionProvider']
        ort.set_default_logger_severity(3) # Hide verbose graph optimizations
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.scale = scale
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]
    
    def upscale_image(self, input_path: str, output_path: str, tile_size: int = 256, show_progress: bool = False):
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        start_time = time.time()
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        if h > tile_size or w > tile_size:
            output = self._upscale_tiled(img_float, tile_size)
        else:
            output = self._upscale_single(img_float)
        
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        
        elapsed = time.time() - start_time
        return True
    
    def _upscale_single(self, img_float: np.ndarray) -> np.ndarray:
        """Upscale image without tiling."""
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        output = self.session.run([self.output_name], {self.input_name: img_tensor})[0]
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))
        return output
    
    def _upscale_tiled(self, img_float: np.ndarray, tile_size: int) -> np.ndarray:
        """Upscale image with tiling"""
        h, w, _ = img_float.shape
        scale = self.scale
        tile_pad = 10
        
        output_h, output_w = h * scale, w * scale
        output = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_map = np.zeros((output_h, output_w), dtype=np.float32)
        
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y_start = max(0, y - tile_pad)
                y_end = min(h, y + tile_size + tile_pad)
                x_start = max(0, x - tile_pad)
                x_end = min(w, x + tile_size + tile_pad)
                
                tile = img_float[y_start:y_end, x_start:x_end]
                tile_tensor = np.transpose(tile, (2, 0, 1))
                tile_tensor = np.expand_dims(tile_tensor, axis=0)
                tile_output = self.session.run([self.output_name], {self.input_name: tile_tensor})[0]
                tile_output = np.squeeze(tile_output)
                tile_output = np.transpose(tile_output, (1, 2, 0))
                
                out_y_start = y_start * scale
                out_y_end = y_end * scale
                out_x_start = x_start * scale
                out_x_end = x_end * scale
                
                output[out_y_start:out_y_end, out_x_start:out_x_end] += tile_output
                weight_map[out_y_start:out_y_end, out_x_start:out_x_end] += 1
        
        output = output / np.maximum(weight_map[:, :, np.newaxis], 1e-8)
        return output


def main():
    MODELS_DIR = 'models'
    INPUT_FOLDER = 'input'
    OUTPUT_FOLDER = 'output'
    TILE_SIZE = 256  # Reverted back to 256 to prevent CUDA OOM bad allocation on heavy models
    model_files = sorted(list(Path(MODELS_DIR).glob('*.onnx')))
    
    if not model_files:
        print(f"No models found in: {MODELS_DIR}")
        exit(1)
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    input_files = [f for f in Path(INPUT_FOLDER).iterdir() if f.suffix.lower() in formats]
    
    if not input_files:
        print(f"No images in {INPUT_FOLDER}/")
        exit(0)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print(f"Models: {len(model_files)}")
    print(f"Images: {len(input_files)}")
    print(f"Total Jobs: {len(model_files) * len(input_files)}")
    print(f"Tile Size: {TILE_SIZE}px")
    
    # Track statistics
    total_jobs = len(model_files) * len(input_files)
    completed_jobs = 0
    failed_jobs = 0
    start_time = time.time()
    
    # Process ALL models
    for model_idx, model_path in enumerate(tqdm(model_files, desc="Models", position=0, leave=False), 1):
        model_name = model_path.stem
        print(f"MODEL [{model_idx}/{len(model_files)}]: {model_name}")
        
        try:
            model_tile_size = 128 if "Transformer" in model_name else TILE_SIZE
            upscaler = SPANUpscaler(str(model_path), scale=4)
            print(f"Provider: {upscaler.active_provider} | Tile Size: {model_tile_size}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            failed_jobs += len(input_files)
            continue
        
        # Process all images with this model
        for img_idx, img_file in enumerate(tqdm(input_files, desc="Images", position=1, leave=False), 1):
            output_file = Path(OUTPUT_FOLDER) / f"{model_name}_{img_file.name}"
            
            if output_file.exists():
                print(f"  - [{img_idx}/{len(input_files)}] Skipping {img_file.name} (already exists)")
                completed_jobs += 1
                continue
                
            try:
                print(f"  -> Processing: {img_file.name}...", end="", flush=True)
                success = upscaler.upscale_image(str(img_file), str(output_file), model_tile_size)
                if success:
                    completed_jobs += 1
                    print(f"  [OK] [{img_idx}/{len(input_files)}] {img_file.name}")
                else:
                    failed_jobs += 1
                    print(f"  [FAIL] [{img_idx}/{len(input_files)}] {img_file.name} (processing failed)")
            except Exception as e:
                failed_jobs += 1
                print(f"  [FAIL] [{img_idx}/{len(input_files)}] {img_file.name} ({str(e)[:50]})")
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  UPSCALING COMPLETE")
    print("=" * 70)
    print(f"  Total Jobs: {total_jobs}")
    print(f"  Successful: {completed_jobs}")
    print(f"  Failed: {failed_jobs}")
    print(f"  Total Time: {elapsed:.2f}s")
    if completed_jobs > 0:
        print(f"  Avg Time/Image: {elapsed/completed_jobs:.2f}s")
    print(f"  Output Folder: {OUTPUT_FOLDER}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        exit(1)