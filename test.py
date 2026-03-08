import torch
import cv2
import numpy as np
import torchvision 
import onnxruntime as ort

print("=" * 50)
print("INSTALLATION VERIFICATION")
print("=" * 50)
print(f"✓ Python: {torch.__version__.split('+')[0]}")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ Torchvision: {torchvision.__version__}")
print(f"✓ CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"✓ CUDA Version: {torch.version.cuda}")

print(f"✓ OpenCV: {cv2.__version__}")
print(f"✓ ONNX Runtime: {ort.__version__}")
print(f"✓ ONNX Providers: {ort.get_available_providers()}")
print("=" * 50)
print("✅ ALL PACKAGES INSTALLED!")
print("=" * 50)