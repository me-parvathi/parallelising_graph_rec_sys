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

