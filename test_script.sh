if [ "$1" = "vggss" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss.json"
elif [ "$1" = "vggss_heard" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss_heard_test.json"
elif [ "$1" = "vggss_unheard" ]; then
    test_data_path="/path/to/VGGSound/"
    test_gt_path="metadata/vggss_unheard_test.json"
elif [ "$1" = "flickr" ]; then
    test_data_path="/path/to/Flickr/test/"
    test_gt_path="/path/to/Flickr/test/Annotations/"
elif [ "$1" = "ms3" ]; then
    test_data_path="/path/to/AVSBench/"
    test_gt_path="metadata/ms3_meta_data.csv"
elif [ "$1" = "s4" ]; then
    test_data_path="/path/to/AVSBench/"
    test_gt_path="metadata/s4_meta_data.csv"
fi

python test_model.py  \
    --test_data_path $test_data_path \
    --test_gt_path $test_gt_path \
    --testset $1 \
    --batch_size 1 \
    --alpha 0.4 \
    --infer_sharpening 0.1 \
    --num_slots 2 \
    --iters 5 \
    --aud_length 5.0 \
    --workers 8 \
    --gpu $2 \
    --wandb false \
    --experiment_name $3 \
    --model_dir ./checkpoints \
    
