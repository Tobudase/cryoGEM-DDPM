import os
import mrcfile
import numpy as np
import cv2
from tqdm import tqdm
import shutil
from glob import glob

#  配置路径 
mrc_dir = r"C:\Users\ROG\Downloads\archive\archive\10025\data\14sep05c_averaged_196"
output_img_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\image"
output_mask_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\mask"

patch_size = 128
stride = 128  # 步长

max_mrcs = 10

os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_mask_dir, exist_ok=True)

counter = 0

#  遍历前 max_mrcs 个 mrc
mrc_files = sorted([f for f in os.listdir(mrc_dir) if f.endswith('.mrc')])[:max_mrcs]

for file in tqdm(mrc_files, desc=f"Processing {max_mrcs} MRC files"):
    mrc_path = os.path.join(mrc_dir, file)
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data
        if data.ndim == 3:
            data = data[0]  

        # 归一化到 [0, 255]
        data = np.nan_to_num(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data = (data * 255).astype(np.uint8)

        h, w = data.shape
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = data[y:y+patch_size, x:x+patch_size]
                filename = f"{counter:05d}.png"
                out_path = os.path.join(output_img_dir, filename)
                cv2.imwrite(out_path, patch)
                shutil.copy(out_path, os.path.join(output_mask_dir, filename))  # 复制为假标签
                counter += 1

print(f" 处理完成，生成 {counter} 个 patch")
print(f" 图像保存至: {output_img_dir}")
print(f" 标签保存至: {output_mask_dir}")