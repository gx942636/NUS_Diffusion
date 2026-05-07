import math
import torch
import yaml
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
import scipy.io as scio
from .condition_methods import get_conditioning_method
from .measurements import get_noise, get_operator
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger

import os
from pathlib import Path


def load_yaml(file_path: str) -> dict:
    possible_paths = [
        file_path,
        os.path.join(os.getcwd(), file_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'yaml_configs',
                     'super_resolution_config.yaml'),
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            with open(abs_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            return config

    raise FileNotFoundError(f"在所有可能路径中都未找到配置文件: {file_path}")


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            image_size_1,
            image_size_2,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            data_path=None,
            data_consistency=False,
            thresholding=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size_1 = image_size_1
        self.image_size_2 = image_size_2
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.data_path = data_path
        self.data_consistency = data_consistency
        self.thresholding = thresholding
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func1 = nn.L1Loss(reduction='sum').to(device)
            self.loss_func = nn.L1Loss(reduction='none').to(device)
        elif self.loss_type == 'l2':
            self.loss_func1 = nn.MSELoss(reduction='sum').to(device)
            self.loss_func = nn.MSELoss(reduction='none').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))  # 补充因子数组
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))  # 补充因子累乘积数组
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))  # 补充因子累乘积数组的前一项数组

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))  # 补充因子累乘积的平方根数组
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))  # 1减去补充因子累乘积的平方根数组
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))  # 1减去补充因子累乘积的对数数组
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))  # 补充因子累乘积的倒数的平方根数组
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))  # 补充因子累乘积的倒数减去1的平方根数组

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)  # 计算后验方差
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))  # 将后验方差数组的对数进行裁剪
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))  # 后验均值的系数1
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))  # 后验均值的系数2

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
                         x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None, nus_fid=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t + 1]]).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            noise, x_spec = self.denoise_fn(nus_fid, condition_x, x, noise_level)
            x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        else:
            x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon, x_spec

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None, nus_fid=None):
        model_mean, model_log_variance, pred_xstart, x_spec = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, nus_fid=nus_fid)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), x_spec, pred_xstart

    @torch.no_grad()
    def p_sample_loop(self, x_in, nus_fid, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))
        if not self.conditional:
            shape = x_in.shape
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                # data consistency
                img = torch.complex(img[:, 0], img[:, 1]).unsqueeze(1)
                img = torch.fft.ifft2(img, dim=[-2, -1])
                noise_fid = torch.fft.ifft2(torch.complex(x_in[:, 0], x_in[:, 1]).unsqueeze(1), dim=[-2, -1])
                img[torch.nonzero(nus_fid, as_tuple=True)] = (img[torch.nonzero(nus_fid, as_tuple=True)]
                                                              + 1e6 * nus_fid[
                                                                  torch.nonzero(nus_fid, as_tuple=True)]) / (1 + 1e6)
                img = torch.concat((torch.fft.fft2(img, dim=[-2, -1]).real, torch.fft.fft2(img, dim=[-2, -1]).imag),
                                   dim=1)
                img = self.p_sample(img, i)

                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            x_nus = nus_fid
            shape = x.shape
            # img = torch.randn(shape, device=device)
            img = torch.fft.fft2(nus_fid, dim=(-2, -1))
            img = torch.concat((img.real, img.imag), dim=1)
            ret_img = x

            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step',
                          total=self.num_timesteps):
                if i < self.num_timesteps and i > 300:
                    continue

                # data consistency + L1
                if self.data_consistency:
                    if self.data_path['real_data_path'] is not None:
                        ori_spec = scio.loadmat(f"{self.data_path['real_data_path']}.mat")['spec']
                        phase_input, pad_shape, pad_height, pad_width = block_data_ddpm(ori_spec, 64)
                        img = torch.complex(img[:, 0], img[:, 1])
                        img = reconstruct_data_ddpm(img, pad_shape, pad_height, pad_width)
                        img = torch.fft.ifft2(img, dim=(-2, -1))
                        nus_fid = reconstruct_data_ddpm(x_nus.squeeze(1), pad_shape, pad_height, pad_width)
                        img[torch.nonzero(nus_fid, as_tuple=True)] = (1 * img[torch.nonzero(nus_fid, as_tuple=True)]
                                                                      + 1e3 * nus_fid[
                                                                          torch.nonzero(nus_fid, as_tuple=True)]) / (
                                                                             1 + 1e3)

                        # 填零加窗处理
                        current_h, current_w = img.shape[-2], img.shape[-1]
                        target_h, target_w = 128, 128
                        fid_padded = F.pad(img, (0, target_w - current_w, 0, target_h - current_h), mode='constant',
                                           value=0)
                        img = apply_window(fid_padded, window_type='gaussian')
                        img = img[:current_h, :current_w]

                        # L1稀疏软阈值约束（在选定域中执行）
                        sparse_domain = 'freq'  # 'time' 或 'freq'
                        lam_l1 = 0.009  # L1稀疏约束强度，可调
                        if sparse_domain == 'freq':
                            # 在频域中稀疏（适合NMR谱图）
                            freq_c = torch.fft.fft2(img, dim=(-2, -1))
                            mag = torch.abs(freq_c)
                            phase = torch.angle(freq_c)
                            mag_thr = torch.clamp(mag - lam_l1, min=0.0)
                            freq_sparse = mag_thr * torch.exp(1j * phase)
                            fid_dc = torch.fft.ifft2(freq_sparse, dim=(-2, -1))
                        else:
                            # 在时域中稀疏（适合FID稀疏重建）
                            mag = torch.abs(fid_dc)
                            phase = torch.angle(fid_dc)
                            mag_thr = torch.clamp(mag - lam_l1, min=0.0)
                            fid_dc = mag_thr * torch.exp(1j * phase)

                        img = torch.fft.fft2(fid_dc, dim=(-2, -1))
                        img, _, _, _ = block_data_ddpm(img, 64)
                        img = img.unsqueeze(1)
                    else:
                        img = torch.complex(img[:, 0], img[:, 1]).unsqueeze(1)
                        img = torch.fft.ifft2(img, dim=(-2, -1))
                        img[torch.nonzero(nus_fid, as_tuple=True)] = (img[torch.nonzero(nus_fid, as_tuple=True)]
                                                                      + 1e3 * nus_fid[
                                                                          torch.nonzero(nus_fid, as_tuple=True)]) / (
                                                                             1 + 1e3)
                        img = torch.fft.fft2(img, dim=(-2,-1))
                    img = torch.concat((img.real, img.imag), dim=1)
                    img, x_spec, _ = self.p_sample(img, i, condition_x=x, nus_fid=nus_fid)

                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)

        if continous:
            return ret_img, x_spec
        else:
            return ret_img, x_spec


    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):  # sample 函数直接生成输入观测值，并调用 p_sample_loop 进行采样
        image_size_1 = self.image_size_1
        image_size_2 = self.image_size_2
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size_1, image_size_2), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, nus_fid,
                         continous=False):  # super_resolution函数接受外部传入的输入观测值和噪声掩码，并调用p_sample_loop进行采样
        return self.p_sample_loop(x_in, nus_fid, continous)


    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
                continuous_sqrt_alpha_cumprod * x_start +
                (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise
        )



    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # # 最初的
            x_recon, x_spec = self.denoise_fn(x_in['LR'], x_in['SR'], x_noisy, continuous_sqrt_alpha_cumprod)
            # # 改1：输入融入SR
            # x_recon = self.denoise_fn(x_in['SR'], x_noisy, continuous_sqrt_alpha_cumprod)

        loss1 = self.loss_func1(noise, x_recon)  # 原始求loss，直接L1loss
        loss_real = self.loss_func(noise[:, 0], x_recon[:, 0])  # 第一次修改loss， 加入掩码因子
        loss_imag = self.loss_func(noise[:, 1], x_recon[:, 1])
        loss_real = loss_real * (x_in['WR'].squeeze(1))
        loss_imag = loss_imag * (x_in['WR'].squeeze(1))
        loss4 = self.loss_func1(x_spec, x_in['HR'])
        loss_real2 = self.loss_func(x_spec[:, 0], x_in['HR'][:, 0])
        loss_imag2 = self.loss_func(x_spec[:, 1], x_in['HR'][:, 1])
        loss_real2 = loss_real2 * (x_in['WR'].squeeze(1))
        loss_imag2 = loss_imag2 * (x_in['WR'].squeeze(1))
        loss2 = (loss_real2 + loss_imag2).sum()
        loss3 = (loss_real + loss_imag).sum()
        loss = 1.5 * loss1 + 6 * loss3 + 0.5 * loss4 + 2 * loss2

        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        # return (
        #     _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
        #     - pred_xstart
        # ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (
                self.sqrt_recip_alphas_cumprod[t] * x_t - pred_xstart) / self.sqrt_recipm1_alphas_cumprod[t]

    @torch.no_grad()
    def ddim_sample(
            self,
            x,
            t,
            clip_denoised=True,
            condition_x=None,
            eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        model_mean, model_log_variance, x_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, x_recon)
        # alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        alpha_bar = self.alphas_cumprod[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
                x_recon * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        # nonzero_mask = (
        #     (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        # )  # no noise when t == 0
        if t != 0:
            nonzero_mask = 1
        else:
            nonzero_mask = 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": x_recon}

    def ddim_sample_loop_progressive(
            self,
            x_in,
            noise=None,
            clip_denoised=True,
            device=None,
            eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(self.denoise_fn.parameters()).device
        shape = x_in.shape
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]
        sample_inter = (1 | (self.num_timesteps // 10))

        ret_img = x_in
        for i in tqdm(indices):
            # t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.ddim_sample(
                    x=img,
                    t=i,
                    clip_denoised=clip_denoised,
                    condition_x=x_in,
                    eta=eta,
                )
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, out["sample"]], dim=0)
                yield ret_img
                img = out["sample"]

    def ddim_sample_loop(
            self,
            x_in,
            noise=None,
            clip_denoised=True,
            device=None,
            eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
                x_in=x_in,
                noise=noise,
                clip_denoised=clip_denoised,
                device=device,
                eta=eta,
        ):
            final = sample
        return final


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def pad_data_ddpm(data, block_size):

    height, width = data.shape


    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size


    if isinstance(data, torch.Tensor):
        padded_data = F.pad(data, (0, pad_width, 0, pad_height), mode='constant', value=0)
    elif isinstance(data, np.ndarray):
        padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    return padded_data, pad_height, pad_width

def block_data_ddpm(data, block_size):

    padded_data, pad_height, pad_width = pad_data_ddpm(data, block_size)


    height, width = padded_data.shape


    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size


    if isinstance(data, torch.Tensor):
        blocks = torch.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype).to(data.device)
    elif isinstance(data, np.ndarray):
        blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size),
                             dtype=padded_data.dtype)


    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[idx] = block
            idx += 1

    return blocks, padded_data.shape, pad_height, pad_width

def reconstruct_data_ddpm(blocks, pad_shape, pad_height, pad_width):

    height, width = pad_shape


    num_blocks, block_height, block_width = blocks.shape


    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width


    reconstructed_data = torch.empty((height, width), dtype=blocks.dtype).to(blocks.device)


    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i * block_height:(i + 1) * block_height,j * block_width:(j + 1) * block_width] = block
            idx += 1


    final_data = reconstructed_data[:height if pad_height == 0 else -pad_height, :width if pad_width == 0 else -pad_width]

    return final_data


# 加窗处理
def apply_window(tensor, window_type='hann'):
    """对2D张量应用窗函数"""
    h, w = tensor.shape[-2], tensor.shape[-1]

    # 创建窗函数
    if window_type == 'hann':
        # 汉宁窗
        win_h = torch.hann_window(h, device=tensor.device).unsqueeze(-1)
        win_w = torch.hann_window(w, device=tensor.device).unsqueeze(0)
        window = win_h @ win_w  # 外积得到2D窗
    elif window_type == 'hamming':
        # 海明窗
        win_h = torch.hamming_window(h, device=tensor.device).unsqueeze(-1)
        win_w = torch.hamming_window(w, device=tensor.device).unsqueeze(0)
        window = win_h @ win_w
    elif window_type == 'blackman':
        # 布莱克曼窗
        win_h = torch.blackman_window(h, device=tensor.device).unsqueeze(-1)
        win_w = torch.blackman_window(w, device=tensor.device).unsqueeze(0)
        window = win_h @ win_w
    elif window_type == 'gaussian':
        # 高斯窗，标准差很大，分布很宽
        # sigma 越大 → 高斯分布越平坦 → 窗函数越接近全1矩阵 → 影响越小
        # linspace 范围越小 → 输入到指数函数的数值越小 → 指数结果越接近1 → 窗函数越接近全1矩阵 → 影响越小
        sigma_h = h * 5.0  # 很大的标准差
        sigma_w = w * 5.0
        x_h = torch.linspace(-0.5, 0.5, h, device=tensor.device)
        x_w = torch.linspace(-0.5, 0.5, w, device=tensor.device)
        win_h = torch.exp(-(x_h ** 2) / (2 * sigma_h ** 2)).unsqueeze(-1)
        win_w = torch.exp(-(x_w ** 2) / (2 * sigma_w ** 2)).unsqueeze(0)
        window = win_h @ win_w
    else:
        raise ValueError(f"不支持的窗类型: {window_type}")

    return tensor * window