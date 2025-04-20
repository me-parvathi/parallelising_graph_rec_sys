import time, csv, torch
from torch.nn.parallel import DistributedDataParallel as DDP

def make_optimizer(model):
    return torch.optim.Adam(
        [{'params': model.module.lin1.parameters(), 'weight_decay': .005},
         {'params': model.module.lin2.parameters(), 'weight_decay': 0.0}],
        lr=0.01)

def accuracy(logits, labels):  # util for test()
    pred = logits.argmax(dim=1)
    return (pred.eq(labels).sum().item() / labels.size(0))

def train_epoch(model, data, opt):
    model.train(); opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(
        model(data.x, data.edge_index, True)[data.train_mask],
        data.y[data.train_mask])
    loss.backward(); opt.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    logits = model(data.x, data.edge_index, False)
    accs = [accuracy(logits[m], data.y[m]) for m in
            [data.train_mask, data.val_mask, data.test_mask]]
    return accs

def run(rank, world_size, args):
    # … (init process‑group if world_size>1, set device, etc.)
    from data import get_cora
    from model import Net

    data, in_dim, out_dim = get_cora(args.data_root, args.device)
    net = Net(in_dim, 64, out_dim).to(args.device)
    model = DDP(net, device_ids=[args.device]) if world_size>1 else net
    opt   = make_optimizer(model)

    history, patience, best_val = [], args.patience, 0
    for epoch in range(1, args.epochs+1):
        tic = time.time()
        loss = train_epoch(model, data, opt)
        tr, val, tst = test(model, data)
        epoch_t = time.time()-tic
        peak_mb = torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0
        history.append((epoch, loss, tr, val, tst, epoch_t, peak_mb))

        # early‑stop
        best_val, patience = (val,0) if val>best_val else (best_val,patience+1)
        if patience>=args.patience: break

    if rank==0:                          # only once!
        with open(args.log_csv,'w',newline='') as f:
            csv.writer(f).writerows([('epoch','loss','train','val','test','t(s)','MB')] + history)
