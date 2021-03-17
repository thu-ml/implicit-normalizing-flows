CUDA_VISIBLE_DEVICES=0 python train_tabular.py --nblocks 20 --vnorms '222222' --dims '128-128-128-128' \
    --save 'experiments/tabular_(power_block20,128*4,c99,sin)_bf' --act 'sin' --data 'power' --batchsize 1000 --coeff 0.99 --nepochs 10000 --epsf 1e-5
