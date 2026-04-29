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

- `download_hf_dataset.py`: Streams and downloads testing datasets from HuggingFace to the `ground_truth` folder.
- `prepare_dataset.py`: Scales down high-resolution images to generate testing sets inside the `input/` folder.
- `span_upscale.py`: The main upscaling engine. Loads models and executes the upscaling pipeline.
- `evaluate.py`: Calculates quantitative metrics (PSNR, SSIM, and LPIPS) across generated outputs.
- `compare.py`: Generates a horizontal comparison image of the original and upscaled results.
- `test.py`: A diagnostic tool to verify your Python environment and GPU status.
- `models/`: Folder to store your `.onnx` model files.
- `ground_truth/`: High-resolution originals used for scaling metrics and benchmarking.
- `input/`: Source testing datasets (`.jpg`, `.png`, `.webp`, etc.).
- `output/`: Folder where upscaled output images are saved.
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

### 3. Usage Pipeline
- **Step 1: Download Testing Data**:
  ```bash
  python download_hf_dataset.py
  ```
- **Step 2: Prepare Input Images**:
  ```bash
  python prepare_dataset.py
  ```
- **Step 3: Run Upscaler**:
  ```bash
  python span_upscale.py
  ```
- **Step 4: Quality Evaluation**:
  Calculate LPIPS, PSNR, and SSIM values.
  ```bash
  python evaluate.py
  ```
- **Step 5: Visual Comparison**:
  Generate panoramic visual comparison for specific inputs.
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
