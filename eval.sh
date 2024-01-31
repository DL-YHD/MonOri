# validation sets
# CUDA_VISIBLE_DEVICES=0 python tools/plain_train_net.py --batch_size 16 --config runs/monori.yaml --ckpt ./output/exp/model_final.pth --eval  

# test sets
python tools/plain_train_net.py --batch_size 16 --config runs/monori.yaml --ckpt ./output/exp/model_moderate_best_soft.pth --eval  --test
