# src/dna_architecture.py
import torch
import torch.nn as nn
import numpy as np

class FourierDNA(nn.Module):
    """
    DNA Nanna: यह कोऑर्डिनेट्स (i, j) से वेट मैट्रिक्स जनरेट करता है।
    """
    def __init__(self, hidden=128, num_freqs=64):
        super().__init__()
        self.hidden = hidden
        self.num_freqs = num_freqs
        
        # 1. फ़ोरियर प्रोजेक्शन मैट्रिक्स (Fixed - यह ट्रेन नहीं होगा)
        # यह कोऑर्डिनेट्स को हाई-डायमेंशनल स्पेस में मैप करता है।
        self.register_buffer('B', torch.randn(num_freqs, 2) * 20.0)
        input_dim = num_freqs * 2
        
        # 2. DNA नेटवर्क (Trainable - यह सीखेगा कि वेट्स कैसे बनाएं)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1)
        )
        
        # 3. स्केल पैरामीटर (Xavier Initialization के लिए)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def generate_weights(self, out_dim, in_dim):
        """
        किसी भी साइज (out_dim x in_dim) का वेट मैट्रिक्स जनरेट करता है।
        """
        device = self.B.device
        
        # स्टेप 1: कोऑर्डिनेट्स ग्रिड बनाना [-1, 1] के बीच
        i = torch.linspace(-1, 1, out_dim, device=device)
        j = torch.linspace(-1, 1, in_dim, device=device)
        grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
        coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)
        
        # स्टेप 2: फ़ोरियर फीचर्स निकालना (Spectral Bias को हटाने के लिए)
        proj = 2 * np.pi * coords @ self.B.T
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        
        # स्टेप 3: DNA नेटवर्क से वेट्स predict करना
        raw_W = self.net(feats).view(out_dim, in_dim)
        
        # स्टेप 4: स्केलिंग (ताकि वेट्स बहुत बड़े या छोटे न हों)
        scale_factor = (2.0 / (in_dim + out_dim)) ** 0.5
        W = raw_W * self.scale * scale_factor
        
        return W

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())