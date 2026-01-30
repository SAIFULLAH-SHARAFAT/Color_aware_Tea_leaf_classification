import torch
import time
import os
import tempfile
from model import SwinRGBHSV  # Adjust class name if different in model.py

def benchmark_model():
    print("-" * 30)
    print("Starting Model Benchmark")
    print("-" * 30)

    # 1. Device Configuration
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: Apple MPS (Metal Performance Shaders) - Accelerated")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: NVIDIA CUDA - Accelerated")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU - Standard")

    # 2. Instantiate Model
    try:
        model = SwinRGBHSV(num_classes=7) # Assuming 7 classes as per dataset
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    # 3. Count Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    # 4. Measure Model Size on Disk
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        torch.save(model.state_dict(), tmp.name)
        size_bytes = os.path.getsize(tmp.name)
        size_mb = size_bytes / (1024 * 1024)
        print(f"Model Size (State Dict): {size_mb:.2f} MB")
        os.unlink(tmp.name) # Cleanup

    # 5. Inference Latency Test
    print("\nRunning Inference Latency Test...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
    
    # Measurement
    iterations = 100
    print(f"Measuring over {iterations} iterations...")
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(dummy_input)
            # For MPS/CUDA, synchronization is needed for accurate timing
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
                
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_ms = (total_time / iterations) * 1000
    
    print("-" * 30)
    print(f"Average Inference Time: {avg_time_ms:.2f} ms")
    print("-" * 30)

if __name__ == "__main__":
    benchmark_model()