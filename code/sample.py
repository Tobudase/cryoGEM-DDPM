import os
import argparse
import torch
from tqdm import tqdm
from glob import glob
from utils import load_checkpoint, save_images, DDPMDataset
from DDPM_model import DDPM
from torch.utils.data import DataLoader

torch.manual_seed(1)

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=64, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_linear_noise_schedule(self):
        return torch.abs(torch.cos(torch.linspace(0, torch.pi / 2, self.noise_steps)) * self.beta_end -
                         (self.beta_end - self.beta_start))

    def generate(self, model, labels, n, save_dir):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            labels = labels.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                input_tensor = torch.cat([x, labels], dim=1)
                predicted_noise = model(input_tensor, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        os.makedirs(save_dir, exist_ok=True)
        for i, img in enumerate(x):
            save_images(img, os.path.join(save_dir, f"sample_{i}.png"))

        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/ddpm25.pth.tar')
    parser.add_argument('--save_dir', type=str, default='../outputs/generated')
    parser.add_argument('--dataset_dir', type=str, default='../datasets/segmentation/mask')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device
    label_paths = sorted(glob(os.path.join(args.dataset_dir, '*.png')))
    dataset = DDPMDataset(data_paths=label_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = DDPM(img_channels=4, time_dim=args.emb_dim).to(device)
    print("=> Loading checkpoint")
    load_checkpoint(args.checkpoint, model, None, None)

    diffusion = Diffusion(img_size=args.image_size, device=device)

    for i, (_, labels) in enumerate(dataloader):
        diffusion.generate(model, labels, n=labels.size(0), save_dir=args.save_dir)
        break  # 一次采样一个 batch