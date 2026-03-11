import sys
import subprocess
import platform

def check_python():
    print(f"Python: {sys.version}")
    print(f"  Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return sys.version_info.major >= 3 and sys.version_info.minor >= 8

def check_packages():
    packages = {
        'onnxruntime': 'ONNX Runtime',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'torch': 'PyTorch'
    }
    
    results = {}
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'Unknown')
            print(f"✓ {name}: {version}")
            results[pkg] = True
        except ImportError:
            print(f"✗ {name}: NOT INSTALLED")
            results[pkg] = False
    return results

def check_cuda():
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✓ CUDA: Available")
            print(f"  GPU Count: {torch.cuda.device_count()}")
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print(f"⚠ CUDA: Not Available (CPU mode only)")
        return cuda_available
    except:
        print(f"✗ CUDA: Cannot verify (PyTorch not installed)")
        return False

def check_onnx_providers():
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"✓ ONNX Providers: {providers}")
        
        has_gpu = 'CUDAExecutionProvider' in providers or 'TensorrtExecutionProvider' in providers
        if has_gpu:
            print(f"  ✓ GPU Acceleration: ENABLED")
        else:
            print(f"  ⚠ GPU Acceleration: DISABLED (CPU only)")
        return has_gpu
    except:
        print(f"✗ ONNX Runtime: Cannot verify")
        return False

def check_models():
    from pathlib import Path
    models = list(Path('models').glob('*.onnx'))
    if models:
        print(f"✓ Models Found: {len(models)}")
        for m in models:
            size_mb = m.stat().st_size / 1024 / 1024
            print(f"  - {m.name} ({size_mb:.1f} MB)")
    else:
        print(f"✗ Models: None found in models/")
    return len(models) > 0

def main():
    print("=" * 60)
    print("  IMG_PROJECT - ENVIRONMENT DIAGNOSTIC")
    print("=" * 60)
    print()
    
    checks = {
        'Python': check_python(),
        'Packages': all(check_packages().values()),
        'CUDA': check_cuda(),
        'ONNX': check_onnx_providers(),
        'Models': check_models()
    }
    
    print()
    print("=" * 60)
    if all(checks.values()):
        print("  ✓ ALL CHECKS PASSED - Ready to upscale!")
    else:
        print("  ⚠ SOME CHECKS FAILED - Review above")
        if not checks['Packages']:
            print("\n  Install: pip install -r requirements.txt")
        if not checks['Models']:
            print("\n  Download ONNX models to the 'models/' folder")
    print("=" * 60)

if __name__ == "__main__":
    main()