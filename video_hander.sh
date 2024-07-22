export  CUDA_VISIBLE_DEVICES=6
#conda activate /mnt/workspace/qinshiyang/miniconda3/envs/down_load_data
cd /mnt/workspace/qinshiyang/common_metrics_on_video_quality


###
#json_file_path
# [
#     {
#     "model_output_video": "1.mp4",
#     "gt_video" : "gt_1.mp4"
# }
# ,
#  {
#     "model_output_video": "2.mp4",
#     "gt_video" : "gt_2.mp4"
# }

# ]
python -m video_hander \
    --json_file_path /mnt/workspace/qinshiyang/hallo/test_out_metric/liveportrait/vfhq_liveportrait.json \
    --gt_key driving_audio_path \
    --model_out_key output_video_path \
    --result_json  /mnt/workspace/qinshiyang/hallo/hallo_metric/metric_results/vfhq_liveportrait_fvd.json \

# python -m video_hander \
#     --model_output_video_folder /mnt/workspace/qinshiyang/hallo/test_out_metric/hallo_vfhq/videos \
#     --gt_video_folder /mnt/workspace/qinshiyang/hallo/hallo_metric/vfhq_test_videos_final \
#     --result_json  /mnt/workspace/qinshiyang/hallo/hallo_metric/metric_results/vfhq_hallo_fvd.json \