import argparse, os, torch, torch.multiprocessing as mp
from engine import run

def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs',   type=int, default=1000)
    p.add_argument('--patience', type=int, default=100)
    p.add_argument('--log-csv',  default='appnp_training_log.csv')
    p.add_argument('--data-root',default=os.environ.get('SCRATCH','.')+'/Cora')
    p.add_argument('--backend',  default='nccl')      # gloo if no GPU
    p.add_argument('--device',   default='cuda:0' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

if __name__ == '__main__':
    args = parse()
    n_gpus = torch.cuda.device_count()
    world  = n_gpus if 'WORLD_SIZE' not in os.environ else int(os.environ['WORLD_SIZE'])
    if world>1:
        mp.spawn(run, args=(world,args), nprocs=world, join=True)
    else:
        run(0,1,args)
