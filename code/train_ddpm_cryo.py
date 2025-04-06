import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_images, save_checkpoint, DDPMDataset, MaskDataset
from DDPM_model import DDPM, Discriminator
torch.manual_seed(1)



 # 执行扩散过程
class Diffusion:
    #初始化参数、噪声日程
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.fixed_noise = torch.randn(1, 1, img_size, img_size).to(device)

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
    
    #生成自定义的β调度
    def prepare_linear_noise_schedule(self):
        return torch.abs(torch.cos(torch.linspace(0, torch.pi / 2, self.noise_steps)) * self.beta_end -
                         (self.beta_end - self.beta_start))
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    #对输入图像加噪
    def noise_images(self, x, t):  # self, x, labels, t
        # labels = labels[:x.shape[0]]
        # labels = labels.expand(x.shape[0], *labels.shape[1:])
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(x)
        # eps[labels > 0] = x[labels > 0]
        img = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        # img[labels > 0] = x[labels > 0]
        return img, eps
    
    #随机采样时间步
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    #用训练好的模型从随机噪声生成图像
    def sample(self, model, labels, n):  # self, model, images, labels, n
        global counter
        # images = images[:n]
        labels = labels[:n]
        # labels = labels.expand(n, *labels.shape[1:])
        model.eval()
        with torch.no_grad():
            x = torch.randn((labels.shape[0], 1, self.img_size, self.img_size)).to(self.device)
            # x[labels > 0] = labels[labels > 0]  #
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(labels.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                    # noise[labels > 0] = labels[labels > 0]  #
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # x[labels > 0] = labels[labels > 0]  #
                # xs = (((x.clamp(-1, 1) + 1) / 2) * 255).type(torch.uint8)
                # save_images(xs, os.path.join("results/steps", f"{i}_fake.png"))
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        labels = (labels.clamp(-1, 1) + 1) / 2
        labels = (labels * 255).type(torch.uint8)

        # images = (images.clamp(-1, 1) + 1) / 2
        # images = (images * 255).type(torch.uint8)
        base_output_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\outputs\generated"
        images_dir = os.path.join(base_output_dir, "images")
        labels_dir = os.path.join(base_output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        save_images(x, os.path.join(images_dir, f"{counter:05d}_fake.png"))
        save_images(labels, os.path.join(labels_dir, f"{counter:05d}_label.png"))
 
 
    #根据给定的标签（mask）和训练好的模型，生成模拟图像并保存。
    def generate(self, model, labels, n, counter):  # self, model, images, labels, n, counter
        # images = images[:n]
        labels = labels[:n]
        # labels = labels.expand(n, *labels.shape[1:])
        model.eval()
        with torch.no_grad():
            x = torch.randn((labels.shape[0], 1, self.img_size, self.img_size)).to(self.device)
            # xs[labels > 0] = images[labels > 0]  # images[labels > 0]
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(labels.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                    # noise[labels > 0] = images[labels > 0]  # images[labels > 0]
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # x[labels > 0] = images[labels > 0]  # images[labels > 0]

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        # labels[labels > 0] = images[labels > 0]
        labels = (labels.clamp(-1, 1) + 1) / 2
        labels = (labels * 255).type(torch.uint8)

        base_output_dir = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\outputs\generated"
        os.makedirs(os.path.join(base_output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(base_output_dir, "labels"), exist_ok=True)

        for i, (img, lab) in enumerate(zip(x, labels)):
            save_images(img, os.path.join(base_output_dir, "images", f"{counter + i:05d}_image.png"))
            save_images(lab, os.path.join(base_output_dir, "labels", f"{counter + i:05d}_label.png"))

        counter += len(x)
        return counter
        

# 训练Discriminator区分真实图与伪图
def train_classifier(args):
    global counter
    device = args.device
    # 设置路径
    real_paths = glob(os.path.join(args.generated_path, "real", "*.png"))
    fake_paths = glob(os.path.join(args.generated_path, "fake", "*.png"))
    os.makedirs(os.path.join(args.generated_path, "good"), exist_ok=True)
    os.makedirs(os.path.join(args.generated_path, "bad"), exist_ok=True)

    dataset = MaskDataset(data_paths=real_paths, label_paths=fake_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    disc = Discriminator(in_channels=1).to(device)
    optimizer = optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "disc3.pth.tar"), disc, optimizer, args.lr,
        )
        n1 = 0
        n2 = 0
        pbar = tqdm(dataloader)
        for i, (_, fake) in enumerate(pbar):
            fake = fake.to(device)

            D_fake = disc(fake)

            if torch.mean(D_fake).item() > 0.1:
                fake = (fake.clamp(-1, 1) + 1) / 2
                fake = (fake * 255).type(torch.uint8)
                save_images(fake, os.path.join(args.generated_path, "good", f"{i:05d}_image.png"))
                n1 += 1
            else:
                fake = (fake.clamp(-1, 1) + 1) / 2
                fake = (fake * 255).type(torch.uint8)
                save_images(fake, os.path.join(args.generated_path, "bad", f"{i:05d}_image.png"))
                n2 += 1

            pbar.set_postfix(n1=n1, n2=n2, mean=torch.mean(D_fake).item())

    mse = nn.MSELoss()
    min_avg_loss = float("inf")

    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        for i, (real, fake) in enumerate(pbar):
            real = real.to(device)
            fake = fake.to(device)

            D_real = disc(real)
            D_fake = disc(fake)

            D_loss_real = mse(D_real, torch.ones_like(D_real) - 0.001*torch.randn_like(D_real))
            D_loss_fake = mse(D_fake, torch.zeros_like(D_fake))

            loss = D_loss_real + D_loss_fake

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix(epoch=epoch, AVG_LOSS=avg_loss / (i + 1), MIN_LOSS=min_avg_loss)

            if i % ((len(dataloader) - 1) // 2) == 0 and i != 0:
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)
            save_checkpoint(disc, optimizer, filename=os.path.join(args.checkpoints, f"disc{epoch}.pth.tar"))


# 用训练好的DDPM生成图像
def test(args):
    global counter
    device = args.device
    data_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\image\*.png")[:args.num_iters * args.batch_size]
    label_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\mask\*.png")[:args.num_iters * args.batch_size]

    dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pbar = tqdm(dataloader)

    ddpm = DDPM(img_channels=1, time_dim=args.emb_dim).to(device)
    diffusion = Diffusion(noise_steps=500, img_size=args.image_size, device=device)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoint, "ddpm_latest.pth.tar"), ddpm, None, None,
        )

    # 只处理 num_iters 批次
    for i, (images, labels) in enumerate(pbar):
        if i >= args.num_iters:
            break  

        labels = labels.to(device)
        counter = diffusion.generate(ddpm, labels, args.batch_size, counter)


# 训练DDPM模型
def train(args):
    global counter
    device = args.device
    data_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\image\*.png")
    label_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\mask\*.png")
    dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ddpm = DDPM(img_channels=1, time_dim=args.emb_dim).to(device)
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoint, "ddpm_latest.pth.tar"), ddpm, optimizer, args.lr,
        )

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    min_avg_loss = float("inf")

    diffusion = Diffusion(noise_steps=200, img_size=args.image_size, device=device)

    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0
        count2 = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            x_t, noise = diffusion.noise_images(images, t)

            for rep in range(5):
                if rep == 1:
                    count1 += 1
                predicted_noise = ddpm(x_t, t, labels)
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if loss < min_avg_loss:
                    if rep == 4:
                        count2 += 1
                    break

            avg_loss += loss.item()
            pbar.set_postfix(epoch=epoch, AVG_MSE=avg_loss / (i+1), count1=count1, count2=count2, MIN_MSE=min_avg_loss)

            if i % ((len(dataloader)-1)//2) == 0 and i != 0:
                # images = (images.clamp(-1, 1) + 1) / 2
                # images = (images * 255).type(torch.uint8)

                # save_images(images, os.path.join("results", f"experiment 5 seg2img/{counter}_real.png"))

                diffusion.sample(ddpm, labels, n=8)
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)
            
            latest_path = os.path.join(args.checkpoints, "ddpm_latest.pth.tar")
            save_checkpoint(ddpm, optimizer, filename=latest_path)

            print(f"[✓] Saved latest checkpoint to {latest_path} | Loss = {min_avg_loss:.6f}")

# 控制是否进入训练/测试模式。
if __name__ == '__main__':
    training = False
    counter = 18869
    if training:
        print(">>> 正在进入训练模式...")
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.load_model = False
        args.epochs = 10
        args.batch_size = 8
        args.emb_dim = 256 
        args.image_size = 128
        args.num_workers = 4
        args.checkpoints = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\results\checkpoints"
        data_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\image\*.png")
        label_paths = glob(r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\dataset\train\mask\*.png")
        args.device = "cuda"
        args.lr = 2e-4
        train(args)
        # train_classifier(args)
    else:
        print(">>> 正在进入测试模式...")
        test_parser = argparse.ArgumentParser()
        test_args = test_parser.parse_args()
        test_args.load_model = True
        test_args.emb_dim = 256
        test_args.num_iters = 25
        test_args.batch_size = 8
        test_args.image_size = 128
        test_args.num_workers = 4
        test_args.checkpoint = r"C:\Users\ROG\Downloads\cryoGEM_DDPM_Project\results\checkpoints"
        test_args.device = "cuda"
        test(test_args)