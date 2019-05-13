This is a preliminary implementation of the paper:

Haddadpour, F.,  Kamani, M.M., Mahdavi, M., & Cadambe, V.
"Trading Redundancy for Communication: Speeding up Distributed SGD for Non-convex Optimization."
International Conference on Machine Learning. 2019.


You can download each dataset using:
```cli
python generate_cifar_tfrecords.py --data-dir=./cifar10 --dataset cifar10
```

Then you can run RI-SGD using this script:

```cli
python main.py --data-dir=./cifar10 \
                --num-gpus=8 \
                --train-steps=45000 \
                --variable-strategy GPU \
                --job-dir=./log/ri-sgd/cifar10-ri-redun25-step50 \
                --run-type multi \
                --redundancy=0.25  \
                --sync-step=50 \
                --dataset cifar10 \
                --eval-batch-size=128
```
```cli
python main.py --data-dir=./cifar10 \
                --num-gpus=8 \
                --train-steps=45000 \
                --variable-strategy GPU \
                --job-dir=./log/ri-sgd/cifar10-ri-sync \
                --run-type sync \
                --redundancy=0.0  \
                --dataset cifar10 \
                --eval-batch-size=128
```
where redundancy is equal to $`\mu`$ in paper and sync-step is equal to $\tau$ in paper.
