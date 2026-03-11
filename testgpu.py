# testgpu2.py
import torch
import numpy as np
import onnxruntime as ort
import time

print("Testing ONNX Runtime GPU inference...")

dummy_input = np.random.randn(1, 3, 64, 64).astype(np.float32)

try:
    session = ort.InferenceSession(
        "models/4xSPANkendata_fp32.onnx",
        providers=['CUDAExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    _ = session.run([output_name], {input_name: dummy_input})
    
    start = time.time()
    for _ in range(5):
        _ = session.run([output_name], {input_name: dummy_input})
    elapsed = (time.time() - start) / 5
    
    print(f"✓ CUDA Execution: SUCCESS")
    print(f"  Avg inference time: {elapsed*1000:.2f}ms")
    print(f"  Provider used: {session.get_providers()[0]}")
    
except Exception as e:
    print(f"✗ CUDA Execution: FAILED")
    print(f"  Error: {e}")
