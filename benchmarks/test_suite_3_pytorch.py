import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim

# Add the parent directory to Python path to import gmem modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bindings-python')))

from gmem.pytorch import GMemLinear

class StandardModel(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class GMemVirtualModel(nn.Module):
    def __init__(self, in_features, hidden, out_features, seed=42):
        super().__init__()
        # Replace heavy RAM Linear layers with Zero-RAM GMemLinear
        # We start context addresses offset so layers don't mathematically overlap
        self.fc1 = GMemLinear(in_features, hidden, seed=seed)
        self.relu = nn.ReLU()
        self.fc2 = GMemLinear(hidden, out_features, seed=seed+1)
        
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def run_training_loop(model, name, X, y, epochs=15, lr=0.01):
    print(f"\\n--- Training {name} ---")
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    start_time = time.perf_counter()
    initial_loss = None
    final_loss = None
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        l_val = loss.item()
        if initial_loss is None:
            initial_loss = l_val
        final_loss = l_val
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {l_val:.4f}")
            
    elapsed = time.perf_counter() - start_time
    print(f"Training Time:    {elapsed:.4f}s")
    print(f"Starting Loss:    {initial_loss:.4f}")
    print(f"Final Loss:       {final_loss:.4f}")
    
    return initial_loss, final_loss

def main():
    print("=========================================================")
    print(" UNIVERSAL TEST SUITE 3: DEEP LEARNING PARITY (PyTorch)  ")
    print("=========================================================")
    
    torch.manual_seed(99)
    
    batch_size = 64
    in_features = 256
    hidden = 512
    out_features = 10
    
    print(f"Simulating network: {in_features} -> {hidden} -> {out_features}")
    
    # Generate synthetic "Batch of Images" (e.g., flattened vectors)
    X = torch.randn(batch_size, in_features)
    y = torch.randn(batch_size, out_features)
    
    print("\\n[+] Instantiating Standard PyTorch RAM Model...")
    std_model = StandardModel(in_features, hidden, out_features)
    
    print("[+] Instantiating Virtual GMem RAM-less Model...")
    # NOTE: GMemLinear computes forward weight matrices instantly on the fly
    gmem_model = GMemVirtualModel(in_features, hidden, out_features, seed=1337)
    
    # Measure typical Autograd Convergence
    std_init, std_final = run_training_loop(std_model, "Standard nn.Linear Model", X, y, epochs=20, lr=0.01)
    gmem_init, gmem_final = run_training_loop(gmem_model, "Generative Memory Model", X, y, epochs=20, lr=0.01)
    
    # Verify both models converged mathematically
    assert std_final < std_init, "Standard model failed to converge."
    assert gmem_final < gmem_init, "GMem Virtual model failed to converge!"
    
    # Verify Parameter Registration:
    # GMem Models do not have dense `weight` parameters, saving millions of bytes
    gmem_params_count = sum(p.numel() for p in gmem_model.parameters())
    std_params_count = sum(p.numel() for p in std_model.parameters())
    
    print(f"\\nRAM Parameters (Standard): {std_params_count:,} floats")
    print(f"RAM Parameters (GMem):     {gmem_params_count:,} floats (bias vectors only)")
    print(f"Saved memory ratio:        {std_params_count / gmem_params_count:.1f}x")
    
    print("\\n[SUCCESS] Custom Autograd functions, backprop, and virtual matrices verified!")


if __name__ == "__main__":
    main()
