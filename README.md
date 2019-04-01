You can download each dataset using:

python generate_cifar_tfrecords.py --data-dir=./cifar10 --dataset cifar10

Then you can run RI-SGD using this script:

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

python main.py --data-dir=./cifar10 \
                --num-gpus=8 \
                --train-steps=45000 \
                --variable-strategy GPU \
                --job-dir=./log/ri-sgd/cifar10-ri-sync \
                --run-type sync \
                --redundancy=0.0  \
                --dataset cifar10 \
                --eval-batch-size=128

where redundancy is equal to $mu$ in paper and sync-step is equal to $tau$ in paper.
