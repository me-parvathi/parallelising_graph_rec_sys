# Graph-Based Recommendation System

This project explores scalable and parallelized implementations of graph-based recommendation models: **PinSage** and **PPNP**.

## How to Run
(TBD)
On CUDA machine there is only like stupid 4GB of data
so we use /scratch

Run the below to get the installations working
```
# Example for pip cache
mv ~/.cache/pip /scratch/$USER/pip_cache
ln -s /scratch/$USER/pip_cache ~/.cache/pip

# Example for .local (where pip installs user packages)
mv ~/.local /scratch/$USER/local
ln -s /scratch/$USER/local ~/.local
```

```
python main.py --epochs 120 --patience 75 \
               --data-root ./data/Planetoid \
               --log-csv run1.csv
```


## Running the Code

### Basic Usage
1.Without Benchmarking (Basic Training):
``` python main.py --dataset Cora ```

2.With Benchmarking (Full Profiling):
``` python main.py --dataset Cora --profile --track_memory --track_time ```

This will run the training with:
Profiling enabled (--profile)
Memory tracking (--track_memory)
Detailed timing metrics (--track_time)

3.With Distributed Training (DDP):
```
# Without benchmarking
python main.py --dataset Cora --ddp

# With benchmarking
python main.py --dataset Cora --ddp --profile --track_memory --track_time
```
Additional useful options you can add:
--epochs N: Set number of epochs (default is 200)
--hidden N: Set hidden layer size (default is 64)
--lr X: Set learning rate (default is 0.01)
--dropout X: Set dropout rate (default is 0.5)
--patience N: Set early stopping patience (default is 100)

For example, a full benchmarking run with custom parameters:
```python main.py --dataset Cora --epochs 100 --hidden 128 --lr 0.005 --profile --track_memory --track_time```

The benchmarking output will include:
A CSV file (appnp_training_log.csv) with detailed metrics
A plot (training_plot.png) showing loss and validation accuracy
Console output with:
Per-epoch metrics
Final benchmark summary
Memory usage statistics
Timing statistics



To document speedups, you can now:
Run ./run_comparison.sh for a quick comparison
Run python benchmark_parallel.py for comprehensive benchmarks
Generate detailed reports with python benchmark_parallel.py --report