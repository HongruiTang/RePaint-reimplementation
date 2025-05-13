import torch
import PIL.Image
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler

import os

curr_dir = os.getcwd()
imgs_dir = curr_dir + "/data/celeba-validation-samples"
generated_dir = curr_dir + "/generated"
os.makedirs(generated_dir, exist_ok=True)
print("Results saved to", generated_dir)

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
            result.save(generated_dir + "/" + f"result_{t}.png")

    return x


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda" 
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"
    print("Using", device)
    model_id = "google/ddpm-celebahq-256"

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

    mask = PIL.Image.open("./data/mask" + "/" + "every_2nd_line.png")
    mask = mask_transform(mask).to(device)
    imgs = os.listdir(imgs_dir)
    imgs.sort()  # ensures img_00.jpg, img_01.jpg, …

    for img_file in imgs:
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
