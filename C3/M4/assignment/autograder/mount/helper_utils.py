# Utils for c3m4_assignment

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from tqdm.notebook import tqdm

def display_some_images(dataset,
    indices_to_show = [0, 1, 2, 274, 275, 276, 530, 531, 532]):
    plt.figure(figsize=(15, 10))
    for idx, img_idx in enumerate(indices_to_show):
        # Get image and label
        img, label = dataset[img_idx]
        
        # Convert tensor to numpy array and transpose to correct dimensions
        img = img.permute(1, 2, 0).numpy()
        
        # Denormalize the image
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot
        plt.subplot(3, 3, idx + 1)
        plt.imshow(img)
        plt.title(f'Label: {dataset.classes[label]}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def compute_accuracy(model, loader, device):
    from tqdm.notebook import tqdm
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        # Create progress bar for evaluation
        pbar = tqdm(loader, desc='Computing Accuracy', leave=False)
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            # Update progress bar with current accuracy
            acc = correct / max(total, 1)
            pbar.set_description(f'Accuracy: {acc:.4f}')
    return correct / max(total, 1)

def make_checkpoint(epoch, model, optimizer, loss, extra=None):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": float(loss),
    }
    if extra:
        ckpt.update(extra)
    return ckpt

def train_model(model, train_loader, dev_loader, num_epochs, optimizer, device, checkpoint=None, save_path="best_model.pt"):
    
    # Load from checkpoint if provided
    start_epoch = 0
    best_acc = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if 'val_acc' in checkpoint:
            best_acc = checkpoint['val_acc']
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc='Epochs', position=0, leave=True)
    
    for epoch in epoch_pbar:
        # Create progress bar for batches
        batch_pbar = tqdm(train_loader, desc='Training', position=1, leave=False)
        
        # Train for one epoch
        model.train()
        total_loss, total = 0.0, 0
        for x, y in batch_pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            # Update batch progress bar
            batch_pbar.set_description(f"Batch Loss: {loss.item():.4f}")
            
        train_loss = total_loss / max(total, 1)
        batch_pbar.close()
        
        # Evaluate
        val_acc = compute_accuracy(model, dev_loader, device)
        
        # Update epoch progress bar description
        epoch_pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = make_checkpoint(epoch, model, optimizer, train_loss,
                                 extra={"val_acc": val_acc})
            torch.save(ckpt, "best_model.pt")
            epoch_pbar.write(f"New best accuracy: {val_acc:.4f}, saved model to best_model.pt")
    
    # Save final model
    ckpt = make_checkpoint(num_epochs-1, model, optimizer, train_loss,
                          extra={"val_acc": val_acc})
    torch.save(ckpt, save_path)
    
    # Final status update
    print(f"\nTraining completed:")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Final accuracy: {val_acc:.4f}")
    print(f"Final model saved to final_model.pt")
    
    return model, best_acc

def _iter_prunable_modules(model):
    """Yield (qualified_name, module) for Conv2d and Linear layers only."""
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m

def sparsity_report(model):
    """
    Compute per-layer and global sparsity (fraction of zeros) over Conv2d/Linear weights.
    Returns
    -------
    dict
        {
          "layers": { "<name>.weight": 0.52, ... },
          "global_sparsity": 0.47
        }
    """
    layers = {}
    zeros_total = 0
    elems_total = 0

    for name, module in _iter_prunable_modules(model):
        if not hasattr(module, "weight"):
            continue
        w = module.weight.detach()
        z = (w == 0).sum().item()
        n = w.numel()
        layers[f"{name}.weight"] = (z / n) if n > 0 else 0.0
        zeros_total += z
        elems_total += n

    global_sparsity = (zeros_total / elems_total) if elems_total > 0 else 0.0
    return {"layers": layers, "global_sparsity": global_sparsity}

def bench(m, iters=20, shape = (1, 3, 224, 224)):
    m.eval()
    x = torch.randn(shape)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iters):
            _ = m(x)
    return (time.perf_counter() - start) / iters

class ToyNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=False),
        )
        self.block = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),                 # Conv + ReLU
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),                      # Conv + BN
            nn.ReLU(inplace=False),                 # Conv + BN + ReLU (across indices)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 16),
            nn.ReLU(inplace=False),                 # Linear + ReLU
            nn.Linear(16, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.block(x)
        x = self.head(x)
        return x

# Utility: pretty-print a few layers
def list_children(module, title):
    print(f"\n== {title} ==")
    for name, child in module.named_modules():
        # show only immediate children of top-level sequentials
        if name in {"stem", "block", "head"}:
            print(f"\n[{name}]")
            for i, sub in enumerate(child.children()):
                print(f"  {i}: {sub.__class__.__name__}")


# Utility: count intrinsic fused layers by class name
def count_fused_layers(model):
    names = []
    for m in model.modules():
        cls = m.__class__.__name__
        if any(k in cls for k in ["ConvReLU2d", "ConvBn2d", "ConvBnReLU2d", "LinearReLU"]):
            names.append(cls)
    return {k: names.count(k) for k in sorted(set(names))}
