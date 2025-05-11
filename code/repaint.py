from datasets import load_dataset
from PIL import Image
import torch
from diffusers import UNet2DModel, DDPMScheduler
from torchvision import transforms
import PIL.Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import List


import os

curr_dir = os.getcwd()
print(curr_dir)
output_dir = curr_dir + "/celeba-validation-images"  # adjust path as you like
os.makedirs(output_dir, exist_ok=True)

# 1. Load the validation split
ds = load_dataset("korexyz/celeba-hq-256x256", split="validation")

# 2b. Shuffle and select indices 0–39
random40 = ds.shuffle().select(range(50))

for i, example in enumerate(random40):
    img = example["image"]
    # if it’s a numpy array, convert to PIL
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img.save(os.path.join(output_dir, f"img_{i:02d}.jpg"))


sample_to_pil = transforms.Compose(
    [
        transforms.Lambda(lambda t: t.squeeze(0)),  # CHW to HWC
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
        transforms.Lambda(lambda t: (t + 1) * 127.5),  # [-1, 1] to [0, 255]
        transforms.Lambda(lambda t: torch.clamp(t, 0, 255)),
        transforms.Lambda(lambda t: t.cpu().detach().numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ]
)


class RePaintScheduler:
    def __init__(
        self,
        ddpm_scheduler: DDPMScheduler,
        resample_steps: int = 10,
        jump_length: int = 10,
    ):
        self.ddpm = ddpm_scheduler
        self.resample_steps = resample_steps
        self.jump_length = jump_length
        self.timesteps = list(range(ddpm_scheduler.num_train_timesteps - 1, -1, -1))

    def q_sample_known(self, x0: torch.Tensor, t: int) -> torch.Tensor:
        alpha_bar = self.ddpm.alphas_cumprod[t].clamp(min=1e-5)
        noise = torch.randn_like(x0)
        return (alpha_bar.sqrt() * x0) + ((1 - alpha_bar).sqrt() * noise)

    def reverse_denoise(
        self, x_t: torch.Tensor, t: int, unet: UNet2DModel
    ) -> torch.Tensor:
        beta_t = self.ddpm.betas[t]
        alpha_t = self.ddpm.alphas[t]
        alpha_bar = self.ddpm.alphas_cumprod[t].clamp(min=1e-5)

        eps_pred = unet(x_t, t).sample
        mu = (1.0 / alpha_t.sqrt()) * (
            x_t - ((beta_t / (1 - alpha_bar).sqrt()) * eps_pred)
        )

        if t > 0:
            return mu + beta_t.sqrt() * torch.randn_like(x_t)
        return mu

    def forward_jump(self, x_prev: torch.Tensor, t: int) -> torch.Tensor:
        t_next = min(t + self.jump_length, 999)
        alpha_bar = self.ddpm.alphas_cumprod[t].clamp(min=1e-5)
        # print(f"\n t_next: {t_next}")
        alpha_bar_next = self.ddpm.alphas_cumprod[t_next].clamp(min=1e-5)
        ratio = (alpha_bar_next / alpha_bar).clamp(min=1e-5, max=1.0)
        noise = torch.randn_like(x_prev)
        return ratio.sqrt() * x_prev + (1 - ratio).sqrt() * noise


@torch.no_grad()
def repaint(
    unet: UNet2DModel, scheduler: RePaintScheduler, x0: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    RePaint inpainting driver.
    Args:
      unet: pretrained UNet2DModel for noise prediction.
      scheduler: configured RePaintScheduler instance.
      x0: original image [-1,1], shape [B,C,H,W].
      mask: binary mask (1=inpaint, 0=keep), same shape.
    Returns:
      Inpainted image tensor.
    """
    device = x0.device
    x = torch.randn_like(x0).to(device)  # start from noise

    for t in tqdm(scheduler.timesteps, desc="RePaint"):
        if t % 10 == 0:
            for u in range(scheduler.resample_steps):
                if u < scheduler.resample_steps - 1:
                    x = scheduler.forward_jump(x, t)
                    for i in range(scheduler.jump_length):
                        x = scheduler.reverse_denoise(
                            x, t + scheduler.jump_length - 1 - i, unet
                        )
        x_known = scheduler.q_sample_known(x0, t)
        x_unknown = scheduler.reverse_denoise(x, t, unet)
        x = mask * x_known + (1 - mask) * x_unknown

        if t % 100 == 0:
            print("saving intermediate result")
            result = x
            result = result.squeeze(0).cpu()
            result = (result + 1) / 2
            result = transforms.ToPILImage()(result.clamp(0, 1))
            result.save(f"result_{t}.png")

    return x


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "google/ddpm-celebahq-256"

    imgs_dir = curr_dir + "/celeba-validation-images"
    generated_dir = curr_dir + "/generated"
    os.makedirs(generated_dir, exist_ok=True)

    model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(device).eval()
    ddpm_scheduler = DDPMScheduler.from_pretrained(model_id)
    scheduler = RePaintScheduler(
        ddpm_scheduler=ddpm_scheduler,
        resample_steps=10,
        jump_length=5,
    )

    data_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ]
    )

    mask_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: t.unsqueeze(0)),
        ]
    )

    mask = PIL.Image.open(curr_dir + "/" + "half.png")
    mask = mask_transform(mask).to(device)
    imgs = os.listdir(imgs_dir)
    imgs.sort()  # ensures img_00.jpg, img_01.jpg, …

    # 3. Iterate over the first 20
    for img_file in imgs[:20]:
        path = os.path.join(imgs_dir, img_file)
        image = PIL.Image.open(path)

        image = data_transform(image).to(device)

        with torch.no_grad():
            result = repaint(model, scheduler, image, mask)

        result = result.squeeze(0).cpu()
        result = (result + 1) / 2
        result = transforms.ToPILImage()(result.clamp(0, 1))
        result.save(generated_dir + "/" + img_file)

        plt.imshow(result)
        plt.axis("off")
        plt.show()
