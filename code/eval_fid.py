import subprocess
import os

# 项目路径 
real_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\image"
fake_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\outputs\generated\images"
device = "cuda"  

def run_fid(real_dir, fake_dir, device):
    print(" 正在评估 FID...")
    print(f"真实图像路径：{real_dir}")
    print(f"生成图像路径：{fake_dir}")
    print(f"使用设备：{device}")

    command = [
        'python', '-m', 'pytorch_fid',
        real_dir,
        fake_dir,
        '--device', device
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n 评估完成，结果如下：")
        print(result.stdout)
        # 保存结果到文件
        result_file = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\fid_result.txt"
        with open(result_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)
        print(f"\n 已保存至: {result_file}")
    else:
        print(" FID 计算失败：")
        print(result.stderr)

if __name__ == "__main__":
    run_fid(real_dir, fake_dir, device)
