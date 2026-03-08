# span_upscale.py
import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

class SPANUpscaler:
    def __init__(self, model_path: str, scale: int = 4):
        """Initialize SPAN upscaler"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Initialize with GPU priority (TensorRT > CUDA > CPU)
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.scale = scale
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]
        
        print(f"✓ Model: {os.path.basename(model_path)}")
        print(f"✓ Provider: {self.active_provider}")
        print(f"✓ Scale: {scale}x")
    
    def upscale_image(self, input_path: str, output_path: str, tile_size: int = 256):
        """Upscale single image"""
        print(f"\nProcessing: {Path(input_path).name}")
        
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ✗ Failed to load")
            return False
        
        h, w = img.shape[:2]
        print(f"  Input: {w}x{h}")
        
        start_time = time.time()
        
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Upscale (with tiling for large images)
        if h > tile_size or w > tile_size:
            output = self._upscale_tiled(img_float, tile_size)
        else:
            output = self._upscale_single(img_float)
        
        # Convert back and save
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_bgr)
        
        elapsed = time.time() - start_time
        new_h, new_w = output_bgr.shape[:2]
        
        print(f"  Output: {new_w}x{new_h}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  ✓ Saved: {output_path}")
        
        return True
    
    def _upscale_single(self, img_float: np.ndarray) -> np.ndarray:
        """Upscale without tiling"""
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        output = self.session.run([self.output_name], {self.input_name: img_tensor})[0]
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))
        return output
    
    def _upscale_tiled(self, img_float: np.ndarray, tile_size: int) -> np.ndarray:
        """Upscale with tiling for memory efficiency"""
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
    
    def upscale_folder(self, input_folder: str, output_folder: str, tile_size: int = 256):
        """Batch process folder"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        files = [f for f in input_path.iterdir() if f.suffix.lower() in formats]
        
        if not files:
            print(f"\n✗ No images found in {input_folder}")
            return
        
        print(f"\n{'=' * 50}")
        print(f"  BATCH: {len(files)} images")
        print(f"{'=' * 50}")
        
        success = 0
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            output_file = output_path / f"span_{file.name}"
            if self.upscale_image(str(file), str(output_file), tile_size):
                success += 1
        
        print(f"\n{'=' * 50}")
        print(f"  DONE: {success}/{len(files)} processed")
        print(f"{'=' * 50}")


if __name__ == "__main__":
    MODEL_PATH = 'models/4xSPANkendata_fp32.onnx'
    INPUT_FOLDER = 'input'
    OUTPUT_FOLDER = 'output'
    TILE_SIZE = 256  # Adjust for VRAM (128-512)
    
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Model not found: {MODEL_PATH}")
        print("Download from: https://github.com/terrainer/AI-Upscaling-Models")
        exit(1)
    
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    
    upscaler = SPANUpscaler(MODEL_PATH, scale=4)
    
    input_files = list(Path(INPUT_FOLDER).glob('*.png')) + \
                  list(Path(INPUT_FOLDER).glob('*.jpg'))
    
    if not input_files:
        print(f"\n⚠️ No images in {INPUT_FOLDER}/")
        print("Place images in 'input' folder and run again")
        exit(0)
    
    upscaler.upscale_folder(INPUT_FOLDER, OUTPUT_FOLDER, TILE_SIZE)