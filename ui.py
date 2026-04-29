import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time
import gradio as gr
from PIL import Image

ort.set_default_logger_severity(3)

class SPANUpscaler:
    def __init__(self, model_path: str, scale: int = 4):
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort.set_default_logger_severity(3)
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.scale = scale
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]
    
    def upscale_image(self, img_pil: Image.Image, tile_size: int = 256, progress=gr.Progress()) -> Image.Image:
        img_np = np.array(img_pil.convert('RGB'))
        h, w = img_np.shape[:2]
        
        img_float = img_np.astype(np.float32) / 255.0
        
        if h > tile_size or w > tile_size:
            output = self._upscale_tiled(img_float, tile_size, progress)
        else:
            progress(0, desc="Upscaling single pass")
            output = self._upscale_single(img_float)
            progress(1, desc="Done")
        
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(output)
    
    def _upscale_single(self, img_float: np.ndarray) -> np.ndarray:
        img_tensor = np.transpose(img_float, (2, 0, 1))
        img_tensor = np.expand_dims(img_tensor, axis=0)
        output = self.session.run([self.output_name], {self.input_name: img_tensor})[0]
        output = np.squeeze(output)
        output = np.transpose(output, (1, 2, 0))
        return output
    
    def _upscale_tiled(self, img_float: np.ndarray, tile_size: int, progress=gr.Progress()) -> np.ndarray:
        h, w, _ = img_float.shape
        scale = self.scale
        tile_pad = 10
        
        output_h, output_w = h * scale, w * scale
        output = np.zeros((output_h, output_w, 3), dtype=np.float32)
        weight_map = np.zeros((output_h, output_w), dtype=np.float32)
        
        y_coords = list(range(0, h, tile_size))
        x_coords = list(range(0, w, tile_size))
        total_tiles = len(y_coords) * len(x_coords)
        
        current_tile = 0
        for y in y_coords:
            for x in x_coords:
                current_tile += 1
                progress(current_tile / total_tiles, desc=f"Processing tile {current_tile}/{total_tiles}")
                
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

MODELS_DIR = 'models'

def get_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    return [f.name for f in Path(MODELS_DIR).glob('*.onnx')]

active_upscaler = None
active_model_name = None

def run_upscale(input_img: Image.Image, model_name: str, progress=gr.Progress()):
    global active_upscaler, active_model_name
    
    if input_img is None:
        raise gr.Error("Please upload an image.")
    if not model_name:
        raise gr.Error("Please select a model.")
        
    model_path = os.path.join(MODELS_DIR, model_name)
    
    if active_upscaler is None or active_model_name != model_name:
        progress(0, desc="Loading model...")
        active_upscaler = SPANUpscaler(model_path)
        active_model_name = model_name
        
    tile_size = 128 if "Transformer" in model_name else 256
    
    start_time = time.time()
    result = active_upscaler.upscale_image(input_img, tile_size=tile_size, progress=progress)
    elapsed = time.time() - start_time
    
    return result, f"✅ Done in {elapsed:.2f}s"

with gr.Blocks(title="SPAN Upscaler") as demo:
    gr.Markdown("# 🚀 SPAN Image Upscaler")
    
    models = get_models()
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Image")
            model_dropdown = gr.Dropdown(choices=models, value=models[0] if models else None, label="Select Model")
            process_btn = gr.Button("Upscale Image", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(type="pil", label="Result")
            status_text = gr.Markdown()

    process_btn.click(
        fn=run_upscale,
        inputs=[input_image, model_dropdown],
        outputs=[output_image, status_text]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)