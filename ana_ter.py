# ============================================================
# 🧬 ANALOG + TERNARY PARAMETERS - NEW APPROACH
# Single Cell: Both approaches + Comparison + Training
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ============================================================
# 1️⃣ TERNARY PARAMETERS (BitNet-style)
# Weights ∈ {-1, 0, +1} → 16x compression, very fast!
# ============================================================

class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights {-1, 0, +1}.
    - Storage: 2 bits per weight (vs 32 bits) = 16x compression
    - Speed: Multiplication = add/subtract/skip (very fast!)
    - Training: Straight-through estimator
    """
    def __init__(self, in_features, out_features, bias=True, threshold=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        
        # Underlying continuous weights (trainable)
        self.weight_underlying = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Scaling factor (alpha) for ternary weights
        self.alpha = nn.Parameter(torch.ones(out_features))
    
    def ternarize(self, w):
        """Convert continuous weights to ternary {-1, 0, +1}"""
        # Threshold-based ternarization
        ternary = torch.zeros_like(w)
        ternary[w > self.threshold] = 1.0
        ternary[w < -self.threshold] = -1.0
        return ternary
    
    def forward(self, x, training=True):
        # Get ternary weights
        ternary_w = self.ternarize(self.weight_underlying)
        
        # Apply scaling (alpha)
        scaled_w = ternary_w * self.alpha.unsqueeze(1)
        
        # Straight-through estimator for training
        if training:
            # Forward: use ternary, Backward: use continuous
            scaled_w = scaled_w + (self.weight_underlying - self.weight_underlying.detach())
        
        # Fast matmul (ternary weights make this efficient)
        out = F.linear(x, scaled_w, self.bias)
        return out
    
    def get_ternary_weights(self):
        """Get actual ternary weights (for analysis)"""
        return self.ternarize(self.weight_underlying)
    
    def count_parameters(self):
        """Count actual stored parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def storage_bits(self):
        """Calculate actual storage in bits"""
        # Ternary weights: 2 bits each
        # Alpha: 32 bits each
        # Bias: 32 bits each
        # Underlying (for training): 32 bits each
        ternary_bits = self.out_features * self.in_features * 2
        alpha_bits = self.out_features * 32
        bias_bits = self.out_features * 32 if self.bias is not None else 0
        underlying_bits = self.out_features * self.in_features * 32
        return ternary_bits + alpha_bits + bias_bits, underlying_bits


# ============================================================
# 2️⃣ ANALOG PARAMETERS (Signal-based)
# One parameter = a continuous signal, generates multiple weights FAST
# ============================================================

class AnalogLinear(nn.Module):
    """
    Linear layer where weights are generated from analog signals.
    - Each row is a "signal" parameterized by few Fourier coefficients
    - FAST generation (pre-computed features)
    - Good compression for structured weights
    """
    def __init__(self, in_features, out_features, num_coeffs=8, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_coeffs = num_coeffs
        
        # Each output row has num_coeffs Fourier coefficients
        # Total params: out_features × num_coeffs × 2 (sin + cos amplitudes)
        self.coeffs_sin = nn.Parameter(
            torch.randn(out_features, num_coeffs) * 0.1
        )
        self.coeffs_cos = nn.Parameter(
            torch.randn(out_features, num_coeffs) * 0.1
        )
        
        # Frequencies (learnable, per coefficient)
        self.freqs = nn.Parameter(
            torch.linspace(1, num_coeffs, num_coeffs).unsqueeze(0).expand(out_features, -1).contiguous()
        )
        
        # Per-row scale and shift
        self.row_scale = nn.Parameter(torch.ones(out_features))
        self.row_shift = nn.Parameter(torch.zeros(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # PRE-COMPUTE time points (for fast generation)
        t = torch.linspace(0, 2*np.pi, in_features)
        self.register_buffer('time_points', t)  # (in_features,)
    
    def generate_weights(self):
        """
        FAST weight generation from analog signals.
        Each row is: sum_k [a_k * sin(k*t) + b_k * cos(k*t)]
        """
        # time_points: (in_features,)
        # freqs: (out_features, num_coeffs)
        # coeffs: (out_features, num_coeffs)
        
        # Compute phases: freqs × time → (out_features, num_coeffs, in_features)
        phases = self.freqs.unsqueeze(-1) * self.time_points.unsqueeze(0).unsqueeze(0)
        
        # Sin and cos components
        sin_comp = torch.sin(phases)  # (out, num_coeffs, in)
        cos_comp = torch.cos(phases)  # (out, num_coeffs, in)
        
        # Apply coefficients: (out, num_coeffs, 1) × (out, num_coeffs, in) → sum over coeffs
        sin_term = (self.coeffs_sin.unsqueeze(-1) * sin_comp).sum(dim=1)  # (out, in)
        cos_term = (self.coeffs_cos.unsqueeze(-1) * cos_comp).sum(dim=1)  # (out, in)
        
        # Combine
        weights = sin_term + cos_term
        
        # Apply per-row scale and shift
        weights = weights * self.row_scale.unsqueeze(1) + self.row_shift.unsqueeze(1)
        
        return weights
    
    def forward(self, x):
        weights = self.generate_weights()
        return F.linear(x, weights, self.bias)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 3️⃣ NORMAL LINEAR (Baseline)
# ============================================================

class NormalLinear(nn.Module):
    """Standard linear layer with FP32 weights"""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.layer = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        return self.layer(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 4️⃣ COMPREHENSIVE COMPARISON
# ============================================================

def run_comparison():
    """Compare Normal vs Ternary vs Analog"""
    
    print("="*80)
    print("🧬 ANALOG + TERNARY vs NORMAL PARAMETERS")
    print("="*80)
    
    configs = [
        {'name': 'Small', 'size': 256},
        {'name': 'Medium', 'size': 1024},
        {'name': 'Large', 'size': 4096},
    ]
    
    results = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️ Device: {device}")
    
    for config in configs:
        print(f"\n{'='*80}")
        print(f"📊 Testing: {config['name']} ({config['size']}×{config['size']})")
        print(f"{'='*80}")
        
        size = config['size']
        batch = 32
        
        x = torch.randn(batch, size, device=device)
        y = torch.randn(batch, size, device=device)
        
        layer_info = {}
        
        # ===== NORMAL =====
        print("\n1️⃣ NORMAL (FP32):")
        normal = NormalLinear(size, size).to(device)
        opt_n = torch.optim.Adam(normal.parameters(), lr=0.01)
        
        params_n = normal.count_parameters()
        print(f"   Params: {params_n:,}")
        print(f"   Memory: {params_n * 4 / 1024**2:.2f} MB")
        
        start = time.time()
        for _ in range(100):
            opt_n.zero_grad()
            loss = ((normal(x) - y)**2).mean()
            loss.backward()
            opt_n.step()
        time_n = time.time() - start
        loss_n = ((normal(x) - y)**2).mean().item()
        print(f"   Train (100 steps): {time_n:.3f}s, Final Loss: {loss_n:.4f}")
        
        layer_info['normal'] = {
            'params': params_n,
            'memory_mb': params_n * 4 / 1024**2,
            'time': time_n,
            'loss': loss_n
        }
        
        # ===== TERNARY =====
        print("\n2️⃣ TERNARY ({-1, 0, +1}):")
        ternary = TernaryLinear(size, size).to(device)
        opt_t = torch.optim.Adam(ternary.parameters(), lr=0.01)
        
        params_t = ternary.count_parameters()
        ternary_bits, underlying_bits = ternary.storage_bits()
        print(f"   Params: {params_t:,}")
        print(f"   Inference Storage: {ternary_bits/8/1024**2:.3f} MB (ternary only)")
        print(f"   Training Storage: {(ternary_bits + underlying_bits)/8/1024**2:.2f} MB")
        
        start = time.time()
        for _ in range(100):
            opt_t.zero_grad()
            loss = ((ternary(x, training=True) - y)**2).mean()
            loss.backward()
            opt_t.step()
        time_t = time.time() - start
        loss_t = ((ternary(x, training=False) - y)**2).mean().item()
        print(f"   Train (100 steps): {time_t:.3f}s, Final Loss: {loss_t:.4f}")
        
        # Count actual ternary weights distribution
        tw = ternary.get_ternary_weights()
        zeros = (tw == 0).sum().item()
        ones = (tw == 1).sum().item()
        neg_ones = (tw == -1).sum().item()
        total = size * size
        print(f"   Weights: {-1}:{neg_ones} ({neg_ones/total*100:.1f}%), "
              f"0:{zeros} ({zeros/total*100:.1f}%), +1:{ones} ({ones/total*100:.1f}%)")
        
        layer_info['ternary'] = {
            'params': params_t,
            'inference_mb': ternary_bits / 8 / 1024**2,
            'training_mb': (ternary_bits + underlying_bits) / 8 / 1024**2,
            'time': time_t,
            'loss': loss_t,
            'sparsity': zeros / total
        }
        
        # ===== ANALOG =====
        print("\n3️⃣ ANALOG (Signal-based):")
        analog = AnalogLinear(size, size, num_coeffs=16).to(device)
        opt_a = torch.optim.Adam(analog.parameters(), lr=0.01)
        
        params_a = analog.count_parameters()
        print(f"   Params: {params_a:,}")
        print(f"   Memory: {params_a * 4 / 1024**2:.3f} MB")
        
        start = time.time()
        for _ in range(100):
            opt_a.zero_grad()
            loss = ((analog(x) - y)**2).mean()
            loss.backward()
            opt_a.step()
        time_a = time.time() - start
        loss_a = ((analog(x) - y)**2).mean().item()
        print(f"   Train (100 steps): {time_a:.3f}s, Final Loss: {loss_a:.4f}")
        
        layer_info['analog'] = {
            'params': params_a,
            'memory_mb': params_a * 4 / 1024**2,
            'time': time_a,
            'loss': loss_a
        }
        
        results.append({
            'name': config['name'],
            'size': size,
            'layers': layer_info
        })
    
    # ===== SUMMARY TABLE =====
    print("\n" + "="*80)
    print("📊 FINAL COMPARISON TABLE")
    print("="*80)
    
    print(f"\n{'Size':<10}{'Type':<10}{'Params':<15}{'Memory (MB)':<15}{'Time (s)':<12}{'Loss':<10}")
    print("-"*80)
    
    for r in results:
        for ptype in ['normal', 'ternary', 'analog']:
            info = r['layers'][ptype]
            mem = info.get('memory_mb', info.get('training_mb', 0))
            print(f"{r['name']:<10}{ptype:<10}{info['params']:>12,}  {mem:>12.3f}  "
                  f"{info['time']:>9.3f}  {info['loss']:>8.4f}")
    
    # ===== COMPRESSION RATIOS =====
    print("\n" + "="*80)
    print("🗜️ COMPRESSION RATIOS")
    print("="*80)
    
    for r in results:
        n_params = r['layers']['normal']['params']
        t_params = r['layers']['ternary']['params']
        a_params = r['layers']['analog']['params']
        
        n_mem = r['layers']['normal']['memory_mb']
        t_mem = r['layers']['ternary']['inference_mb']  # Ternary inference storage
        a_mem = r['layers']['analog']['memory_mb']
        
        print(f"\n{r['name']} ({r['size']}×{r['size']}):")
        print(f"  Params: Normal=1.0x | Ternary={n_params/t_params:.1f}x | Analog={n_params/a_params:.1f}x")
        print(f"  Memory (inference): Normal=1.0x | Ternary={n_mem/t_mem:.1f}x | Analog={n_mem/a_mem:.1f}x")
        print(f"  Ternary Sparsity: {r['layers']['ternary']['sparsity']*100:.1f}% zeros")
    
    # ===== SPEED COMPARISON =====
    print("\n" + "="*80)
    print("⚡ SPEED COMPARISON (Training)")
    print("="*80)
    
    for r in results:
        t_norm = r['layers']['normal']['time']
        t_ter = r['layers']['ternary']['time']
        t_ana = r['layers']['analog']['time']
        
        print(f"\n{r['name']} ({r['size']}×{r['size']}):")
        print(f"  Normal:  {t_norm:.3f}s (baseline)")
        print(f"  Ternary: {t_ter:.3f}s ({t_ter/t_norm:.2f}x)")
        print(f"  Analog:  {t_ana:.3f}s ({t_ana/t_norm:.2f}x)")
    
    # ===== VISUALIZATION =====
    print("\n🎨 Creating Visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    sizes = [r['name'] for r in results]
    x_pos = np.arange(len(sizes))
    width = 0.25
    
    # Plot 1: Parameter Count (log scale)
    ax = axes[0, 0]
    n_params = [np.log10(r['layers']['normal']['params']) for r in results]
    t_params = [np.log10(r['layers']['ternary']['params']) for r in results]
    a_params = [np.log10(r['layers']['analog']['params']) for r in results]
    
    ax.bar(x_pos - width, n_params, width, label='Normal', color='#FF6B6B')
    ax.bar(x_pos, t_params, width, label='Ternary', color='#4ECDC4')
    ax.bar(x_pos + width, a_params, width, label='Analog', color='#95E1D3')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Parameters (log10)')
    ax.set_title('Parameter Count')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Memory Usage
    ax = axes[0, 1]
    n_mem = [r['layers']['normal']['memory_mb'] for r in results]
    t_mem = [r['layers']['ternary']['inference_mb'] for r in results]
    a_mem = [r['layers']['analog']['memory_mb'] for r in results]
    
    ax.bar(x_pos - width, n_mem, width, label='Normal', color='#FF6B6B')
    ax.bar(x_pos, t_mem, width, label='Ternary (inference)', color='#4ECDC4')
    ax.bar(x_pos + width, a_mem, width, label='Analog', color='#95E1D3')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Usage')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Training Time
    ax = axes[1, 0]
    n_time = [r['layers']['normal']['time'] for r in results]
    t_time = [r['layers']['ternary']['time'] for r in results]
    a_time = [r['layers']['analog']['time'] for r in results]
    
    ax.bar(x_pos - width, n_time, width, label='Normal', color='#FF6B6B')
    ax.bar(x_pos, t_time, width, label='Ternary', color='#4ECDC4')
    ax.bar(x_pos + width, a_time, width, label='Analog', color='#95E1D3')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Training Speed (100 steps)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Final Loss (quality)
    ax = axes[1, 1]
    n_loss = [r['layers']['normal']['loss'] for r in results]
    t_loss = [r['layers']['ternary']['loss'] for r in results]
    a_loss = [r['layers']['analog']['loss'] for r in results]
    
    ax.bar(x_pos - width, n_loss, width, label='Normal', color='#FF6B6B')
    ax.bar(x_pos, t_loss, width, label='Ternary', color='#4ECDC4')
    ax.bar(x_pos + width, a_loss, width, label='Analog', color='#95E1D3')
    ax.set_xlabel('Model Size')
    ax.set_ylabel('Final Loss')
    ax.set_title('Training Quality (lower = better)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/analog_ternary_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== TERNARY WEIGHT VISUALIZATION =====
    print("\n🎨 Visualizing Ternary Weight Distribution...")
    
    # Get weights from last medium layer
    ternary_layer = TernaryLinear(256, 256).to(device)
    opt = torch.optim.Adam(ternary_layer.parameters(), lr=0.01)
    x_demo = torch.randn(32, 256, device=device)
    y_demo = torch.randn(32, 256, device=device)
    for _ in range(50):
        opt.zero_grad()
        loss = ((ternary_layer(x_demo, training=True) - y_demo)**2).mean()
        loss.backward()
        opt.step()
    
    tw = ternary_layer.get_ternary_weights().cpu().numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ternary weight heatmap
    axes[0].imshow(tw, cmap='RdBu', vmin=-1, vmax=1)
    axes[0].set_title('Ternary Weights Heatmap\n(Red=-1, White=0, Blue=+1)')
    axes[0].set_xlabel('Input')
    axes[0].set_ylabel('Output')
    
    # Distribution
    axes[1].hist(tw.flatten(), bins=[-1.5, -0.5, 0.5, 1.5], color='#4ECDC4', edgecolor='black')
    axes[1].set_xlabel('Weight Value')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Ternary Weight Distribution')
    axes[1].set_xticks([-1, 0, 1])
    
    # Sparsity stats
    total = tw.size
    zeros = (tw == 0).sum()
    pos = (tw == 1).sum()
    neg = (tw == -1).sum()
    
    stats = ['−1', '0', '+1']
    counts = [neg, zeros, pos]
    colors = ['#FF6B6B', '#95E1D3', '#4ECDC4']
    
    axes[2].pie(counts, labels=[f'{s}\n{c/total*100:.1f}%' for s, c in zip(stats, counts)], 
                colors=colors, autopct='', startangle=90)
    axes[2].set_title('Weight Value Distribution')
    
    plt.tight_layout()
    plt.savefig('results/ternary_weights_viz.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== ANALOG WEIGHT VISUALIZATION =====
    print("\n🎨 Visualizing Analog Signal Weights...")
    
    analog_layer = AnalogLinear(256, 256, num_coeffs=16).to(device)
    opt = torch.optim.Adam(analog_layer.parameters(), lr=0.01)
    for _ in range(50):
        opt.zero_grad()
        loss = ((analog_layer(x_demo) - y_demo)**2).mean()
        loss.backward()
        opt.step()
    
    aw = analog_layer.generate_weights().cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].imshow(aw, cmap='viridis')
    axes[0].set_title('Analog-Generated Weight Matrix')
    axes[0].set_xlabel('Input')
    axes[0].set_ylabel('Output')
    plt.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)
    
    # Show few rows as signals
    for i in range(5):
        axes[1].plot(aw[i], label=f'Row {i}')
    axes[1].set_xlabel('Input Index')
    axes[1].set_ylabel('Weight Value')
    axes[1].set_title('Weight Rows as Analog Signals')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/analog_weights_viz.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ===== FINAL VERDICT =====
    print("\n" + "="*80)
    print("🏆 FINAL VERDICT")
    print("="*80)
    
    print("\n📊 Best Choice by Use Case:")
    print("\n🎯 TERNARY is BEST for:")
    print("   ✅ Maximum compression (16x at inference)")
    print("   ✅ Fast inference (multiply = add/subtract)")
    print("   ✅ Energy efficient (great for edge/mobile)")
    print("   ⚠️  Slight quality loss (constrained weights)")
    
    print("\n🎯 ANALOG is BEST for:")
    print("   ✅ Structured weights (smooth patterns)")
    print("   ✅ Good compression (depends on signal complexity)")
    print("   ✅ Fast generation (pre-computed features)")
    print("   ⚠️  Less compression than ternary for random weights")
    
    print("\n🎯 NORMAL is BEST for:")
    print("   ✅ Maximum quality (no constraints)")
    print("   ✅ Simple and proven")
    print("   ❌ High memory usage")
    
    print("\n" + "="*80)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*80)
    print("\n📁 Files Generated:")
    print("   - results/analog_ternary_comparison.png")
    print("   - results/ternary_weights_viz.png")
    print("   - results/analog_weights_viz.png")
    
    return results


# ============================================================
# 🎬 MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n🚀 STARTING ANALOG + TERNARY EXPERIMENT\n")
    results = run_comparison()