echo "====== start of raft-sintel ======"

python -u train.py \
    --name raft-sintel \
    --stage sintel \
    --validation sintel \
    --restore_ckpt checkpoints/raft-things.pth \
    --gpus 0 1 \
    --num_steps 100000 \
    --batch_size 6 \
    --lr 0.000125 \
    --image_size 368 768 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --aug_tx 2.0 \
    --aug_ty 2.0 \
    --aug_rot 2.0 \
    --aug_smin 0.95 \
    --aug_smax 1.0


echo "====== start of raft-kitti ======"

python -u train.py \
    --name raft-kitti \
    --stage kitti \
    --validation kitti \
    --restore_ckpt raft-sintel.pth \
    --gpus 0 1 \
    --num_steps 50000 \
    --batch_size 6 \
    --lr 0.0001 \
    --image_size 288 960 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --aug_tx 2.0 \
    --aug_ty 2.0 \
    --aug_rot 2.0 \
    --aug_smin 0.95 \
    --aug_smax 1.0

