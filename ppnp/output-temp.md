
# Single GPU
python main.py --dataset Cora --epochs 20

=== Final Benchmark Summary ===
Best Validation Accuracy: 0.6920
Avg Epoch Time: 4.0421 sec
Peak GPU Memory: 50.86 MB
Peak CPU Memory: 868.00 MB
Avg Train Time: 2.0282 sec
Avg Eval Time: 2.0138 sec
Peak GPU Memory (Reserved): 62.00 MB
Peak CPU Memory (Virtual): 17488.57 MB
Avg Forward Time: 1.4129 sec
Avg Backward Time: 0.4037 sec
Avg Optimizer Time: 0.2018 sec

# Multiple GPUs
python main.py --dataset Cora --epochs 20 --ddp
=== Final Benchmark Summary ===
Best Validation Accuracy: 0.5000
Avg Epoch Time: 4.0442 sec
Peak GPU Memory: 51.21 MB
Peak CPU Memory: 1034.43 MB
Avg Train Time: 2.0297 sec
Avg Eval Time: 2.0146 sec
Peak GPU Memory (Reserved): 62.00 MB
Peak CPU Memory (Virtual): 17979.16 MB
Avg Forward Time: 1.4136 sec
Avg Backward Time: 0.4039 sec
Avg Optimizer Time: 0.2019 sec