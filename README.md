# IMG_PROJECT - AI Image Upscaling Evaluation Toolkit

A high-performance toolkit for upscaling low-resolution images by 4x and evaluating state-of-the-art Super-Resolution architectures. This project facilitates high-quality reconstruction using GPU-accelerated inference via ONNX Runtime and provides tools to quantitatively and qualitatively compare different AI models.

---

## 🌟 Key Features

- **Multi-Model Support & Evaluation**: Compare different upscale technologies instantly.
- **Dynamic Selection**: Run the script to choose between available upscaling specializations.
- **GPU-Accelerated**: Prioritizes NVIDIA GPUs via `CUDAExecutionProvider` for high-speed processing.
- **Tiled Processing**: Uses smart memory management (tiling) to process large images without crashing on limited VRAM.
- **Batch Processing**: Automatically processes all images in the `input/` folder and saves them with model-specific prefixes.
- **Side-by-Side Comparison**: Built-in tool to generate panoramic comparisons featuring the Original image and all upscaled versions.

---

## 🧪 Available Models
- **4x-ESRGAN**: (Enhanced Super-Resolution Generative Adversarial Network) Best for general sharpness, clarity, and texture hallucination.
- **4x-Transformer**: (Dual Aggregation Transformer / DAT) Optimized for realistic photo reconstruction and removing heavy web compression (JPEG/WebP) artifacts.
- **4x-SPAN**: (Swift Parameter-free Attention Network) High-performance, lightweight model balancing speed and quality.

---

## 📂 Project Structure

- `span_upscale.py`: The main upscaling engine. Loads models and executes the upscaling pipeline.
- `compare.py`: Generates a horizontal comparison image of the original and upscaled results.
- `test.py`: A diagnostic tool to verify your Python environment and GPU status.
- `testgpu.py`: A quick benchmark to verify CUDA execution and inference speed.
- `models/`: Folder to store your `.onnx` model files.
- `input/`: Place your source images or evaluation datasets (`.jpg`, `.png`, `.webp`, etc.) here.
- `output/`: Folder where upscaled images are saved.
- `venv/`: Python virtual environment for isolated dependencies.

---

## 🚀 Getting Started

### 1. Requirements
Ensure you have an NVIDIA GPU for the best experience. The project relies on:
- Python 3.8+
- ONNX Runtime (GPU/CUDA version)
- PyTorch (used as a library provider for CUDA DLLs on Windows)
- OpenCV, NumPy, tqdm

### 2. Setup (Windows)
1. Ensure your virtual environment is activated:
   ```powershell
   .\venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Usage
- **To Upscale Images**:
  1. Add images to the `input/` folder.
  2. Run the upscaler:
     ```bash
     python span_upscale.py
     ```
- **To Compare Results**:
  Run the comparison script to see all models compared to the original:
  ```bash
  python compare.py
  ```

---

## 🛠️ System Diagnostics
Run `python test.py` to check:
- CUDA Availability & GPU Name
- VRAM Status
- ONNX Provider initialization
- Missing DLLs or dependencies
