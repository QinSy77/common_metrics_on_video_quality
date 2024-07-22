import matplotlib.pyplot as plt
import json


def read_json_to_dict(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

# 示例用法
file_path = '/mnt/workspace/qinshiyang/common_metrics_on_video_quality/video_metrics_result_0627.json'
result_dict = read_json_to_dict(file_path)

# 提取帧数和对应的FVD值
frames = list(result_dict["lpips"]["value"].keys())
fvd_values = list(result_dict["lpips"]["value"].values())

# 将帧数转换为整数列表
frames = [int(frame) for frame in frames]

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(frames, fvd_values, marker='o', linestyle='-', color='b', label='lpips Value')

# 设置图表标题和标签
plt.title('lpips Value Over Different Frame Counts')
plt.xlabel('Frame Count')
plt.ylabel('lpips Value')
plt.legend()

# 显示网格
plt.grid(True)
plt.savefig("/mnt/workspace/qinshiyang/common_metrics_on_video_quality/lpips.png",dpi=500)

# 显示图表
#plt.show()
