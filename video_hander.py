import torch
import cv2
import numpy as np
import json
from calculate_fvd import calculate_fvd
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim
from calculate_lpips import calculate_lpips
import argparse
import os

def video_to_tensor(video_path, size=(512, 512),max_frames=None):
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

def calculate_video_metrics(video_path1, video_path2, device=torch.device("cuda"), size=(512, 512), max_frames=None):

    cap1 = cv2.VideoCapture(video_path1)
    frame_count1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    cap1.release()
    cap2 = cv2.VideoCapture(video_path1)
    frame_count2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    cap2.release()

    max_frames = min(frame_count1,frame_count2)

    if max_frames:
        max_frames = max_frames

    

    #print(f"视频包含 {frame_count} 帧。")

    video1 = video_to_tensor(video_path1, size=size, max_frames=max_frames).unsqueeze(0)
    video2 = video_to_tensor(video_path2, size=size, max_frames=max_frames).unsqueeze(0)
    
    result = {}
    result['fvd_styleganv'] = calculate_fvd(video1, video2, device, method='styleganv')
    #result['fvd_videogpt'] = calculate_fvd(video1, video2, device, method='videogpt')
    #result['ssim'] = calculate_ssim(video1, video2)
    #result['psnr'] = calculate_psnr(video1, video2)
    result['lpips'] = calculate_lpips(video1, video2, device)
    
    return result

def process_from_json(json_file_path,args):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    out_data = []
    all_fvd = 0.0
    all_lpips = 0.0
    index = 0

    for item in data:
        source_video_path = item[args.gt_key]
        modelout_save_path = item[args.model_out_key]

        result = calculate_video_metrics(source_video_path, modelout_save_path)

        fvd = list(result['fvd_styleganv']['value'].values())[-1]

        lpips = list(result["lpips"]["value"].values())
        lpips_item_mean = np.mean(np.array(lpips))


        print({"fvd": fvd, "lpips_item_mean": lpips_item_mean})



        all_fvd += fvd
        all_lpips += lpips_item_mean

        index += 1

    all_lpips = all_lpips / index
    all_fvd = all_fvd / index

    out_data.append({"fvd": all_fvd, "lpips_item_mean": all_lpips})

    print(f"all_fvd: {all_fvd}, all_lpips: {all_lpips}")

    return out_data

def process_from_folders(model_output_video_folder, gt_video_folder):
    out_data = []
    all_fvd = 0.0
    all_lpips = 0.0
    index = 0

    for video_file in os.listdir(model_output_video_folder):
        video_file_path = os.path.join(model_output_video_folder, video_file)

        if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
            source_video_path = video_file_path
            modelout_save_path = os.path.join(gt_video_folder, video_file)

            result = calculate_video_metrics(source_video_path, modelout_save_path)

            fvd = list(result['fvd_styleganv']['value'].values())[-1]

            lpips = list(result["lpips"]["value"].values())
            lpips_item_mean = np.mean(np.array(lpips))

            #out_data.append({"fvd": fvd, "lpips_item_mean": lpips_item_mean})
            all_fvd += fvd
            all_lpips += lpips_item_mean

            index += 1

    all_lpips = all_lpips / index
    all_fvd = all_fvd / index

    out_data.append({"fvd": all_fvd, "lpips_item_mean": all_lpips})

    print(f"all_fvd: {all_fvd}, all_lpips: {all_lpips}")

    return out_data

def main():
    parser = argparse.ArgumentParser(description="Calculate video metrics")
    parser.add_argument("--json_file_path", type=str, help="Path to the JSON file containing video paths")
    parser.add_argument("--gt_key", type=str, help="Path to the JSON file containing video paths")
    parser.add_argument("--model_out_key", type=str, help="Path to the JSON file containing video paths")
    parser.add_argument("--model_output_video_folder", type=str, help="Path to the folder containing model output videos")
    parser.add_argument("--gt_video_folder", type=str, help="Path to the folder containing ground truth videos")
    parser.add_argument("--result_json", type=str,required=True
                        , help="Path to result_json")


    args = parser.parse_args()

    if args.json_file_path:
        out_data = process_from_json(args.json_file_path,args)
    elif args.model_output_video_folder and args.gt_video_folder:
        out_data = process_from_folders(args.model_output_video_folder, args.gt_video_folder)
    else:
        print("Please provide either a JSON file path or both model output and ground truth video folders")
        return

    out_json = args.result_json
    os.makedirs(os.path.dirname(out_json),exist_ok=True)
    with open(out_json, 'w') as json_file:
        json.dump(out_data, json_file, indent=4)

    print(f"结果已保存到 {out_json}")

if __name__ == "__main__":
    main()