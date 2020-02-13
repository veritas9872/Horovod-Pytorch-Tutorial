set -ex
horovodrun -np 2 -H localhost:2 python train/train_model.py