if [ $# -eq 3 ]; then
    experiment_name='noname'
elif [ $# -eq 4 ]; then
    experiment_name=$4
fi

if [ "$1" = "vggss_10k" ] || [ "$1" = "vggss_144k" ]; then
    train_path="/path/to/VGGSound/"
elif [ "$1" = "flickr_10k" ] || [ "$1" = "flickr_144k" ]; then
    train_path="/path/to/flickr_trainset/"
fi

if [ "$2" = "vggss" ] || [ "$2" = "all" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss.json"
elif [ "$2" = "vggss_heard" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss_heard_test.json"
elif [ "$2" = "vggss_unheard" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss_unheard_test.json"
elif [ "$2" = "flickr" ]; then
    test_data_path="/path/to/Flickr/test/"
    test_gt_path="/path/to/Flickr/test/Annotations/"
elif [ "$2" = "ms3" ]; then
    test_data_path="/path/to/AVSBench/"
    test_gt_path="metadata/ms3_meta_data.csv"
elif [ "$2" = "s4" ]; then
    test_data_path="/path/to/AVSBench/"
    test_gt_path="metadata/s4_meta_data.csv"
fi

if [ "$1" = "vggss_10k" ] || [ "$1" = "flickr_10k" ]; then
    python train_slot.py  \
        --train_data_path $train_path \
        --test_data_path $test_data_path \
        --test_gt_path $test_gt_path \
        --trainset $1 \
        --testset $2 \
        --epochs 100 \
        --warmup -1 \
        --batch_size 256 \
        --init_lr 0.00005 \
        --weight_decay 0.01 \
        --alpha 0.4 \
        --lam1 0.1 \
        --lam2 0.1 \
        --lam3 100.0 \
        --tau 0.03 \
        --infer_sharpening 0.1 \
        --num_slots 2 \
        --iters 5 \
        --reciprocal_k 20 \
        --mask_ratio 0.1 \
        --aud_length 5.0 \
        --workers 8 \
        --gpu $3 \
        --wandb true \
        --hard_aud true \
        --hard_img true \
        --rand_aud false \
        --experiment_name $experiment_name \

elif [ "$1" = "vggss_144k" ] || [ "$1" = "flickr_144k" ]; then
    python train_slot.py  \
        --train_data_path $train_path \
        --test_data_path $test_data_path \
        --test_gt_path $test_gt_path \
        --trainset $1 \
        --testset $2 \
        --epochs 50 \
        --warmup -1 \
        --batch_size 256 \
        --init_lr 0.00005 \
        --weight_decay 0.01 \
        --alpha 0.4 \
        --lam1 0.1 \
        --lam2 0.1 \
        --lam3 100.0 \
        --tau 0.03 \
        --infer_sharpening 1.0 \
        --num_slots 2 \
        --iters 5 \
        --aud_length 5.0 \
        --reciprocal_k 20 \
        --mask_ratio 0.1 \
        --workers 8 \
        --gpu $3 \
        --wandb false \
        --hard_aud true \
        --hard_img true \
        --rand_aud false \
        --experiment_name $experiment_name \

fi
