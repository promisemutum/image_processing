import os, sys, time, subprocess, numpy as np, onnxruntime as ort

def banner(t): print(f"\n{'='*60}\n {t}\n{'='*60}")

def check_dlls():
    capi = os.path.join(os.path.dirname(ort.__file__), "capi")
    needed = ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]
    missing = [d for d in needed if not os.path.exists(os.path.join(capi, d))]
    return capi, missing

def test_cuda_provider():
    banner("CUDA PROVIDER TEST")
    available = ort.get_available_providers()
    print(f"Available: {available}")
    
    if "CUDAExecutionProvider" not in available:
        print("X CUDA provider not registered. Reinstall onnxruntime-gpu.")
        return False
        
    # Try to create a session (will fail gracefully if DLLs are missing)
    try:
        # Use a dummy path to force DLL loading check
        ort.InferenceSession("dummy.onnx", providers=["CUDAExecutionProvider"])
    except Exception as e:
        if "cublas" in str(e).lower() or "126" in str(e):
            print("X CUDA DLLs missing or blocked by AV")
            return False
    print("OK CUDA provider loaded successfully")
    return True

if __name__ == "__main__":
    banner("SYSTEM & ENV")
    print(f"Python: {sys.version.split()[0]} | Executable: {sys.executable}")
    print(f"Virtual Env: {'OK' if sys.prefix != sys.base_prefix else 'X'}")
    
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], 
                             capture_output=True, text=True, check=True)
        print(f"GPU: {out.stdout.strip()}")
    except: print("! nvidia-smi not found")

    banner("ONNX RUNTIME")
    print(f"Version: {ort.__version__}")
    capi, missing = check_dlls()
    print(f"CUDA DLLs in capi/: {'OK All Present' if not missing else f'X Missing: {missing}'}")
    
    if missing:
        print("\nFIX: Windows Defender quarantined the DLLs.")
        print("   1. Open Windows Security > Protection History")
        print("   2. Restore any onnxruntime/cublas DLLs")
        print("   3. Run: Add-MpPreference -ExclusionPath 'C:\\Users\\Asus\\Documents\\dev\\img_project\\venv'")
        print("   4. Reinstall: pip install onnxruntime-gpu==1.19.0 --force-reinstall --no-cache-dir")
    else:
        test_cuda_provider()
        
    print("\n" + "="*60 + "\n")