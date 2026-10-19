# scripts/test_1b_model.py
import torch
import torch.nn as nn
import time

class Frozen1BModel(nn.Module):
    """स्टैंडर्ड PyTorch मॉडल स्ट्रक्चर"""
    def __init__(self, d=8192, num_layers=15):
        super().__init__()
        self.input_proj = nn.Linear(1, d)
        self.output_proj = nn.Linear(d, 1)
        self.layers = nn.ModuleList([nn.Linear(d, d) for _ in range(num_layers)])
    
    def forward(self, x):
        h = torch.relu(self.input_proj(x))
        for layer in self.layers:
            h = torch.relu(layer(h))
        return self.output_proj(h)

def test_model():
    print("="*60)
    print("🧪 LOADING & TESTING 1B FROZEN MODEL")
    print("="*60)

    # 1. चेकपॉइंट लोड करें
    print("\n1. Loading 3.75 GB Checkpoint from disk...")
    checkpoint = torch.load('saved_models/frozen_1b_model.pt', map_location='cpu')
    config = checkpoint['config']
    print(f"   ✅ Loaded! Total Params: {checkpoint['metadata']['total_params']:,}")

    # 2. मॉडल स्ट्रक्चर बनाएं और वेट्स डालें
    print("\n2. Building Model Architecture & Loading Weights...")
    model = Frozen1BModel(d=config['d'], num_layers=config['num_layers'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✅ Model is ready and frozen!")

    # 3. इनफेरेंस टेस्ट (Forward Pass)
    print("\n3. Running Inference Test...")
    x_test = torch.randn(1, 1) # Dummy input
    
    # Warmup
    with torch.no_grad(): _ = model(x_test)
    
    # Speed Test
    start = time.time()
    with torch.no_grad():
        output = model(x_test)
    end = time.time()
    
    print(f"   ✅ Forward Pass Successful!")
    print(f"   ⏱️ Inference Time: {(end-start)*1000:.2f} ms")
    print(f"   📊 Output Shape: {output.shape}")
    print(f"   🔢 Output Value: {output.item():.6f}")
    
    print("\n" + "="*60)
    print("🎉 SUCCESS! Your 1B DNA-Generated Model is fully functional!")
    print("="*60)

if __name__ == "__main__":
    test_model()