set -ex
N=2
horovodrun -np $N -H localhost:$N python train/train_model.py