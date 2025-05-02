import torch
import torch.nn.functional as F
import threading

def train(model, data, optimizer):
    """Train the model for one epoch."""
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    out = model(data.x, data.edge_index)
    
    # Compute loss
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, data):
    """Evaluate the model on train, validation, and test sets."""
    model.eval()
    logits = model(data.x, data.edge_index)

    # Subdivide the three mask-based accuracies into separate tasks
    masks = [data.train_mask, data.val_mask, data.test_mask]
    accs = [None] * len(masks)

    def _compute(idx, mask):
        pred = logits[mask].argmax(dim=1)
        accs[idx] = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

    # Spawn one thread per mask
    threads = []
    for i, mask in enumerate(masks):
        t = threading.Thread(target=_compute, args=(i, mask))
        t.start()
        threads.append(t)

    # Wait for all three to finish
    for t in threads:
        t.join()

    # Returns [train_acc, val_acc, test_acc] as before
    return accs
