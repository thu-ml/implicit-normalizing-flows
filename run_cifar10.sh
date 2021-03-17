CUDA_VISIBLE_DEVICES=0 python train_img.py --data cifar10 --actnorm True \
    --nblocks '2-2-2' --idim '512' --act 'swish' --kernels '3-1-3' --vnorms '2222' --fc-end False --preact True \
    --save 'experiments/cifar10(blocks_2*3(512,k313)_swish_nofc_preact_10term' --coeff 0.9 --n-exact-terms 10
