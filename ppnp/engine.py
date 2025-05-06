import torch
import torch.nn.functional as F
import threading
from torch.cuda.amp import autocast, GradScaler
import gc

# Create a global scaler for mixed precision training
scaler = GradScaler()

# Set memory optimization settings
torch.cuda.empty_cache()
gc.collect()

def train(model, data, optimizer, loader=None):
    """Train the model for one epoch using mixed precision training. Supports full-graph and mini-batch."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_examples = 0
    
    # Get model device
    device = next(model.parameters()).device

    if loader is not None:
        # Mini-batch training (Reddit)
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            batch.x.requires_grad_(True)  # Enable gradient tracking for features
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * batch.train_mask.sum().item()
            total_examples += batch.train_mask.sum().item()
            
            # Clean up memory
            del out, batch
            # Only clear cache periodically to avoid slowdowns
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
            
            # Periodically print progress for Reddit (which has many batches)
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1} batches, current loss: {total_loss / max(total_examples, 1):.4f}")
                
        return total_loss / max(total_examples, 1)
    else:
        # Full-graph training (Cora/Citeseer/PubMed)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.item()


@torch.no_grad()
def evaluate(model, data, val_loader=None, test_loader=None):
    """Evaluate the model on train, validation, and test sets using mixed precision.
    For Reddit dataset, uses loaders for validation and test to avoid OOM errors.
    """
    model.eval()
    
    # Get model device
    device = next(model.parameters()).device
    
    # Mini-batch evaluation for Reddit
    if val_loader is not None and test_loader is not None:
        # Initialize counters
        train_correct, train_total = 0, 0
        val_correct, val_total = 0, 0
        test_correct, test_total = 0, 0
        
        # Get validation accuracy
        print("Evaluating validation set...")
        for i, batch in enumerate(val_loader):
            batch = batch.to(device)
            with autocast():
                out = model(batch.x, batch.edge_index)
            
            # Calculate train accuracy if train nodes are in this batch
            if hasattr(batch, 'train_mask') and batch.train_mask.sum() > 0:
                pred = out[batch.train_mask].argmax(dim=1)
                train_correct += pred.eq(batch.y[batch.train_mask]).sum().item()
                train_total += batch.train_mask.sum().item()
            
            # Calculate validation accuracy
            pred = out[batch.val_mask].argmax(dim=1)
            val_correct += pred.eq(batch.y[batch.val_mask]).sum().item()
            val_total += batch.val_mask.sum().item()
            
            # Clean up memory
            del out, batch, pred
            # Only clear cache periodically to avoid slowdowns
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Get test accuracy
        print("Evaluating test set...")
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            with autocast():
                out = model(batch.x, batch.edge_index)
            
            pred = out[batch.test_mask].argmax(dim=1)
            test_correct += pred.eq(batch.y[batch.test_mask]).sum().item()
            test_total += batch.test_mask.sum().item()
            
            # Clean up memory
            del out, batch, pred
            # Only clear cache periodically to avoid slowdowns
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        # Calculate final metrics
        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        test_acc = test_correct / max(test_total, 1)
        
        return train_acc, val_acc, test_acc
    else:
        # Full-graph evaluation (Cora/Citeseer/PubMed)
        with autocast():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Calculate accuracy for train/val/test
            train_correct = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item()
            train_acc = train_correct / data.train_mask.sum().item()
            
            val_correct = pred[data.val_mask].eq(data.y[data.val_mask]).sum().item()
            val_acc = val_correct / data.val_mask.sum().item()
            
            test_correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
            test_acc = test_correct / data.test_mask.sum().item()
        
        return train_acc, val_acc, test_acc
