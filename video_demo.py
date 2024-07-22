import torch
import cv2
import numpy as np
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import json

def video_to_tensor(video_path, size=(64, 64), max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频包含 {frame_count} 帧。")
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
        frame = torch.tensor(frame).permute(2, 0, 1)  # Convert to CHW format
        frames.append(frame)
    cap.release()
    
    # If the number of frames is less than max_frames, pad with zeros
    if len(frames) < max_frames:
        padding_frames = [torch.zeros(3, *size)] * (max_frames - len(frames))
        frames.extend(padding_frames)
    
    return torch.stack(frames)

def calculate_video_metrics(video_path1, video_path2, device=torch.device("cuda"), size=(512, 512), max_frames=240):
    video1 = video_to_tensor(video_path1, size=size, max_frames=max_frames).unsqueeze(0)
    video2 = video_to_tensor(video_path2, size=size, max_frames=max_frames).unsqueeze(0)
    
    result = {}
    result['fvd_styleganv'] = calculate_fvd(video1, video2, device, method='styleganv')
    #看需要，取消注释
    #result['fvd_videogpt'] = calculate_fvd(video1, video2, device, method='videogpt')
    #result['ssim'] = calculate_ssim(video1, video2)
    #result['psnr'] = calculate_psnr(video1, video2)
    result['lpips'] = calculate_lpips(video1, video2, device)
    
    return result

out_data = []

#json_file_path = "/mnt/workspace/qinshiyang/AniPortrait_custom/utils/pick_vid_test/gf_model_out_without3dproj.json"
json_file_path = "/mnt/workspace/qinshiyang/AniPortrait_custom/utils/stage2_score/test_out_video/hallo_fps30/hallo_a2v_fps30.json"
with open(json_file_path, 'r') as file:
    # 加载JSON文件中的数据到一个字典
    data = json.load(file)

#{"ref_img_save_path":ref_img_save_path,"gt_video_save_path":gt_video_save_path}

all_fvd = 0.0
all_lpips = 0.0
index = 0
#for item in data:
for key in data.keys():

    index +=1
    #source_video_path = item["output_video"]
    #modelout_save_path = item["gt_video"]

    source_video_path = data[key]["output_video_path"]
    modelout_save_path = data[key]["driving_audio_path"]

    result = calculate_video_metrics(source_video_path, modelout_save_path , max_frames=150)



    fvd = list(result['fvd_styleganv']['value'].values())[-1]

    print(fvd)
    
    all_fvd +=fvd
    lpips = list(result["lpips"]["value"].values())
    
    lpips_item_mean = np.mean(np.array(lpips))
    all_lpips+=lpips_item_mean


    #out_data[key] = {"fvd":fvd,"lpips_item_mean":lpips_item_mean}
    out_data.append({"fvd":fvd,"lpips_item_mean":lpips_item_mean})

all_lpips = all_lpips/index

all_fvd = all_fvd/index
print(f"all_fvd:{all_fvd}" , f"all_lpips:{all_lpips}")
    
out_json= '/mnt/workspace/qinshiyang/AniPortrait_custom/utils/stage2_score/test_out_video/hallo_fps30/merit_model_hallo_a2v_0708.json'

with open(out_json, 'w') as json_file:
    json.dump(out_data,json_file, indent=4)

print(f"结果已保存到{out_json}")
