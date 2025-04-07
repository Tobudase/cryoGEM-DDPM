# cryoGEM-DDPM

基于 DDPM（Denoising Diffusion Probabilistic Model）的 Cryo-EM 模拟图像生成项目。

本项目用于基于 CryoGEM 数据集和 EMPIAR-10025 原始数据，训练一个去噪扩散概率模型（DDPM），并通过 FID（Fréchet Inception Distance）指标评估图像质量。

---

##  项目结构

```
cryoGEM_DDPM_Project/
├── code/                    # 主代码目录
│   ├── train_ddpm_cryo.py  # 主训练/测试脚本
│   ├── prepare_empiar_patches.py  # MRC 图像切片脚本
│   ├── utils.py            # 数据加载、保存工具函数
│   └── DDPM_model.py       # DDPM 与判别器模型结构
├── dataset/                # 图像与标签目录
│   └── train/
│       ├── image/          # 输入图像（patches）
│       └── mask/           # 标签图像（可为假标签）
├── outputs/
│   └── generated/          # 生成的图像与标签输出
├── results/
│   └── checkpoints/        # 保存训练好的模型权重
└── README.md               # 项目说明
```

---

##  快速开始

### 1. 安装依赖环境

```bash
conda create -n cryogem python=3.9 -y
conda activate cryogem
pip install -r requirements.txt
```

或手动安装核心依赖：

```bash
pip install torch torchvision tqdm numpy opencv-python einops pytorch-fid mrcfile
```


### 2. 数据准备

使用 `prepare_empiar_patches.py` 处理 EMPIAR-10025 `.mrc` 文件为 patch 图像：

```bash
python code/prepare_empiar_patches.py
```

你可以在文件中修改：
```python
patch_size = 128
stride = 128
max_mrcs = 10
```

---

### 3. 模型训练

编辑 `train_ddpm_cryo.py` 中 `training=True`，运行：

```bash
python code/train_ddpm_cryo.py
```

你可以控制训练时长，例如：
```python
args.epochs = 10
args.batch_size = 8
args.image_size = 128
args.lr = 2e-4
```

---

### 4. 模型测试与图像生成

将 `training=False`，运行生成图像：

```bash
python code/train_ddpm_cryo.py
```

可在 `outputs/generated/images/` 查看生成图。

---

### 5. FID 评估

确保 `pytorch_fid` 已安装，运行评估脚本：

```bash
python code/eval_fid.py
```

并修改脚本内路径：
```python
real_dir = "C:/Users/ROG/Downloads/cryoGEM_DDPM_Project/dataset/train/image"
fake_dir = "C:/Users/ROG/Downloads/cryoGEM_DDPM_Project/outputs/generated/images"
```

---

##  模型架构说明

本项目基于 UNet + Transformer 结构，结合时间嵌入（timestep embedding）进行条件生成训练。整体模型架构为：

```
Input (image+label) --> UNet 编码器 --> Transformer blocks --> UNet 解码器 --> 去噪预测
```

---

##  预训练模型下载

我们已将训练好的 DDPM 模型上传至 Google Drive，便于复现与评估：

- 🔗 [下载 ddpm_latest.pth.tar](https://drive.google.com/file/d/1_XksaqyySbOX-RiXSlVA-a9rxK1hsIxz/view?usp=drive_link)

---

##  示例结果（FID）

- 输入 patch: 128x128
- 训练图像数量: ~2000
- 噪声步数: 200
- 生成数量: 2048
- FID 分数: **14.25**（合理，模型生成质量与训练数据基本一致）

---

##  参考资料

- CryoGEM: [https://arxiv.org/abs/2312.02235](https://arxiv.org/abs/2312.02235)
- Retree: [https://github.com/AAleka/retree](https://github.com/AAleka/retree)
- DDPM: [Ho et al., 2020](https://arxiv.org/abs/2006.11239)

---


