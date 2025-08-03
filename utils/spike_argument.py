import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import interp1d

class SpikeAugmentation:
    def __init__(self, amp_prob=0.7, temp_prob=0.6, noise_prob=0.5, coll_prob=0.4, drop_prob=0.3, waveform_bank=None):
        self.amp_prob = amp_prob
        self.temp_prob = temp_prob
        self.noise_prob = noise_prob
        self.coll_prob = coll_prob
        self.waveform_bank = waveform_bank
        self.drop_prob = drop_prob
        self.cropAugmentation = CropAugmentation()
        self.noise = TorchSmartNoise()
        self.jitter = Jitter()


    def __call__(self, waveform, seed=42, drop=False, noise=True, noise_prob=0.2):
        """
        waveform: torch.Tensor of shape [C, T]
        Returns an augmented waveform of the same shape
        """
        self.noise_prob = noise_prob

        # if np.random.rand() < self.noise_prob:
        #     waveform = self.add_noise(waveform)

        if np.random.rand() < self.coll_prob and self.waveform_bank is not None:
            waveform = self.collision(waveform)

        if np.random.rand() < self.amp_prob:
            waveform = self.amplitude_jitter(waveform)

        if np.random.rand() < self.temp_prob:
            waveform = self.jitter(waveform)

        if np.random.rand() < self.noise_prob and noise:
            waveform = self.noise(waveform)

        waveform, (start, end) = self.cropAugmentation(waveform)

        if (np.random.rand() < self.drop_prob) and drop:
            waveform = self.drop_channel(waveform)

        return waveform, (start, end)

    def drop_channel(self, waveform):
        num_channels = waveform.shape[0]
        drop_idx = np.random.choice(num_channels, size=np.random.randint(1, num_channels + 1), replace=False)
        waveform[drop_idx, :] = 0
        return waveform

    def amplitude_jitter(self, waveform):
        scale = np.random.uniform(0.8, 1.2)
        amp_jit = np.array([scale for i in range(waveform.shape[0])])
        return waveform * amp_jit[:, None]

    def temporal_jitter(self, waveform):
        c, t = waveform.shape
        # Step 1: upsample 8x using linear interpolation
        t_new = np.linspace(0, t - 1, num=t * 8)
        waveform_np = waveform.cpu().numpy()
        upsampled = np.zeros((c, t * 8))
        for i in range(c):
            interp_fn = interp1d(np.arange(t), waveform_np[i], kind='linear')
            upsampled[i] = interp_fn(t_new)

        # Step 2: downsample with random offset 1~8
        offset = np.random.randint(1, 9)
        downsampled = upsampled[:, offset-1::8][:, :t]

        # Step 3: shift +2 or -2
        shift = np.random.choice([-2, 0, 2])
        if shift > 0:
            shifted = np.pad(downsampled, ((0, 0), (shift, 0)), mode='constant')[:, :-shift]
        elif shift < 0:
            shifted = np.pad(downsampled, ((0, 0), (0, -shift)), mode='constant')[:, -shift:]
        else:
            shifted = downsampled

        return torch.tensor(shifted, dtype=waveform.dtype)

    def add_noise(self, waveform):
        c, t = waveform.shape
        noise = torch.randn_like(waveform)
        # 模拟简单 spatiotemporal covariance（可以用真实 noise 统计矩阵替代）
        noise = F.gaussian_blur(noise.unsqueeze(0), (1, 5)).squeeze(0)
        return waveform + 0.05 * noise  # 控制 noise 强度

    def collision(self, waveform):
        idx = np.random.randint(0, self.waveform_bank.shape[0])
        coll_waveform = self.waveform_bank[idx]  # [C, T]
        coll_waveform = torch.tensor(coll_waveform, dtype=waveform.dtype)
        coll_waveform = coll_waveform.to(waveform.device)

        scale = np.random.uniform(0.2, 1.0)
        shift = (2 * np.random.binomial(1, 0.5) - 1) * np.random.randint(5, 60)

        coll_waveform = coll_waveform * scale
        # if shift > 0:
        #     coll_waveform = F.pad(coll_waveform, (shift, 0))[:, :waveform.shape[1]]
        # elif shift < 0:
        #     coll_waveform = F.pad(coll_waveform, (0, -shift))[:, -shift:]

        coll_waveform = self.shift_chans(coll_waveform, shift)
        return waveform + coll_waveform

    def gaussian_blur_1d(self, waveform, kernel_size=5, sigma=1.0):
        """
        对 spike waveform 做时域高斯模糊。
        Args:
            waveform: [B, C, T] 或 [C, T] 的 spike 张量
            kernel_size: 高斯核大小（建议为奇数）
            sigma: 标准差，越大模糊越强
        Returns:
            模糊后的 waveform，形状不变
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # 转为 [1, C, T]

        B, C, T = waveform.shape
        device = waveform.device

        # 构建高斯核
        k = (kernel_size - 1) // 2
        x = torch.arange(-k, k + 1, dtype=torch.float32, device=device)
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, -1)  # [1, 1, K]

        # 对每个通道做1D卷积
        kernel = kernel.repeat(C, 1, 1)  # [C, 1, K]
        waveform_blurred = F.conv1d(waveform, kernel, padding=k, groups=C)  # [B, C, T]

        if waveform_blurred.shape[0] == 1:
            return waveform_blurred.squeeze(0)  # 恢复为 [C, T] if 原来是2D
        return waveform_blurred

    def add_gaussian_noise(self, waveform, std=0.05):
        noise = torch.randn_like(waveform) * std
        return waveform + noise

    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), "constant")
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), "constant")
        if int(shift_) == shift_:
            ceil = int(shift_)
            temp = (
                np.roll(curr_wf_pos, ceil, axis=1)[:, :-int_shift]
                if shift_ > 0
                else np.roll(curr_wf_neg, ceil, axis=1)[:, int_shift:]
            )
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos, ceil, axis=1) * (shift_ - floor))[
                    :, :-ceil
                ] + (np.roll(curr_wf_pos, floor, axis=1) * (ceil - shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg, ceil, axis=1) * (shift_ - floor))[
                    :, -floor:
                ] + (np.roll(curr_wf_neg, floor, axis=1) * (ceil - shift_))[:, -floor:]
        wf_final = temp

        return wf_final

class CropAugmentation:
    def __init__(self, crop_size=11, prob=1.0):
        """
        crop_size: number of channels to retain
        """
        self.crop_size = crop_size
        self.prob = prob

    def __call__(self, waveform):
        """
        waveform: torch.Tensor of shape [C, T]
        returns: cropped waveform of shape [crop_size, T]
        """
        if np.random.rand() > self.prob:
            return waveform  # No augmentation

        C, T = waveform.shape
        if C <= self.crop_size:
            return waveform  # No cropping needed

        # Step 1: find max amp channel
        # max_ch = torch.argmax(waveform.abs().max(dim=1).values).item()
        max_ch = math.floor(C / 2)

        # Step 2: choose center channel
        half_crop = self.crop_size // 2

        if np.random.rand() < 0.5:
            # case 1: center on max channel
            center = max_ch
        else:
            # case 2: center on non-max but includes max
            choices = [c for c in range(half_crop, C - half_crop) if c != max_ch and abs(c - max_ch) <= half_crop]
            if not choices:
                center = max_ch  # fallback
            else:
                center = np.random.choice(choices)

        # Step 3: extract crop
        start = center - half_crop
        end = start + self.crop_size
        if start < 0:
            start = 0
            end = self.crop_size
        if end > C:
            end = C
            start = C - self.crop_size

        cropped = waveform[start : end, :]

        return cropped.clone(), (start, end)


class TorchSmartNoise(object):
    """Add spatio‑temporal correlated noise using precomputed sqrt‑covariance matrices."""

    def __init__(
        self,
        spatial_cov='./resources/dataset/ds/covariances/spatial_cov.npy',   # [C_total, C_total]
        temporal_cov='./resources/dataset/ds/covariances/temporal_cov.npy',  # [T, T]
        noise_scale: float = 1.0,
        device='cpu',
    ):
        """
        Args:
            sqrt_spatial_cov: 预先计算好的空间协方差矩阵平方根，shape=(C_total, C_total)
            sqrt_temporal_cov: 预先计算好的时间协方差矩阵平方根，shape=(T, T)
            noise_scale: 缩放系数
            normalize: 是否将最终 noisy waveform 归一化到 [0,1]
            gpu: 使用的 cuda 设备 id
            p: 添加噪声的概率
        """
        self.spatial_cov = np.load(spatial_cov)
        self.temporal_cov = np.load(temporal_cov)
        self.device = device

        with torch.no_grad():
            U_s, S_s, _ = np.linalg.svd(self.spatial_cov)
            self.sqrt_spatial_cov = torch.from_numpy(U_s @ np.diag(np.sqrt(S_s))).float().to(device)

            U_t, S_t, _ = np.linalg.svd(self.temporal_cov)
            self.sqrt_temporal_cov = torch.from_numpy(U_t @ np.diag(np.sqrt(S_t))).float().to(device)

            self.noise_scale = float(noise_scale)

    def __call__(self, wf):
        """
        sample can be:
          - Tensor [C, T]
          - Tuple (wf: Tensor[C, T], chan_nums: array[C])
        返回与输入同形状的带噪声 waveform
        """

        # 确保在 cuda
        with torch.no_grad():
            wf = wf.to(self.device)
            C, T = wf.shape
            assert T == self.sqrt_temporal_cov.shape[0], "时间维度不匹配"
            assert self.sqrt_spatial_cov.shape[0] >= C, "空间协方差维度不够"

            # 生成标准正态噪声 [C, T]
            noise = torch.randn((C, T), device=wf.device)

            # noise_full: [C_total, T]
            C_total = self.sqrt_spatial_cov.shape[0]
            noise_full = torch.zeros((C_total, T), device=wf.device)
            noise_full[:C, :] = noise
            noise = noise_full

            # 空间相关： sqrt_spatial_cov @ noise  => [C_total, T]
            spatial_noise = self.sqrt_spatial_cov @ noise

            # 时间相关： spatial_noise @ sqrt_temporal_cov  => [C_total, T]
            spatiotemporal_noise = spatial_noise @ self.sqrt_temporal_cov

            # 如果有 chan_nums，则取回原始的 C 行；否则直接裁到 [C, T]
            spatiotemporal_noise = spatiotemporal_noise[:C, :]

            print(spatiotemporal_noise)

            # 缩放并添加
            noisy_wf = wf + self.noise_scale * spatiotemporal_noise

            print(noisy_wf)
            return noisy_wf.to('cpu')


class Jitter(object):
    """Temporally jitter the waveform through upsampling, random offset downsampling, and shift"""

    def __init__(self, up_factor=8, shift=2):
        """
        Args:
            up_factor (int): Upsampling factor (e.g., 8).
            shift (int): Max temporal shift after downsampling (in samples).
        """
        assert isinstance(up_factor, int) and up_factor > 0, "up_factor must be positive int"
        assert isinstance(shift, int) and shift >= 0, "shift must be non-negative int"
        self.up_factor = up_factor
        self.shift = shift

    def __call__(self, sample):
        """
        Args:
            sample: waveform (torch.Tensor) or tuple/list (waveform, chan_nums[, chan_locs])
                    waveform shape should be [C, T]

        Returns:
            Jittered waveform in same structure as input (tuple or just waveform)
        """
        chan_locs = None
        if isinstance(sample, (tuple, list)):
            if len(sample) == 2:
                waveform, chan_nums = sample
            elif len(sample) == 3:
                waveform, chan_nums, chan_locs = sample
            else:
                raise ValueError("sample tuple must have 2 or 3 elements")
        else:
            waveform = sample
            chan_nums = None

        if not isinstance(waveform, torch.Tensor):
            raise TypeError("Expected waveform to be a torch.Tensor")

        c, t = waveform.shape
        waveform_np = waveform.cpu().numpy()

        # Step 1: upsample with linear interpolation
        t_new = np.linspace(0, t - 1, num=t * self.up_factor)
        upsampled = np.zeros((c, t * self.up_factor), dtype=np.float32)
        for i in range(c):
            interp_fn = interp1d(np.arange(t), waveform_np[i], kind='linear')
            upsampled[i] = interp_fn(t_new)

        # Step 2: random offset downsampling
        offset = np.random.randint(0, self.up_factor)
        downsampled = upsampled[:, offset::self.up_factor]
        downsampled = downsampled[:, :t]  # Ensure shape stays [C, T]

        shifted = self.shift_chans(downsampled, 2)

        jittered_tensor = torch.tensor(shifted, dtype=waveform.dtype)

        # Return in same structure as input
        if chan_nums is not None and chan_locs is not None:
            return [jittered_tensor, chan_nums, chan_locs]
        elif chan_nums is not None:
            return [jittered_tensor, chan_nums]
        else:
            return jittered_tensor


    def shift_chans(self, wf, shift_):
        # use template feat_channel shifts to interpolate shift of all spikes on all other chans
        int_shift = int(math.ceil(shift_)) if shift_ >= 0 else -int(math.floor(shift_))
        curr_wf_pos = np.pad(wf, ((0, 0), (0, int_shift)), "constant")
        curr_wf_neg = np.pad(wf, ((0, 0), (int_shift, 0)), "constant")
        if int(shift_) == shift_:
            ceil = int(shift_)
            temp = (
                np.roll(curr_wf_pos, ceil, axis=1)[:, :-int_shift]
                if shift_ > 0
                else np.roll(curr_wf_neg, ceil, axis=1)[:, int_shift:]
            )
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            if shift_ > 0:
                temp = (np.roll(curr_wf_pos, ceil, axis=1) * (shift_ - floor))[
                    :, :-ceil
                ] + (np.roll(curr_wf_pos, floor, axis=1) * (ceil - shift_))[:, :-ceil]
            else:
                temp = (np.roll(curr_wf_neg, ceil, axis=1) * (shift_ - floor))[
                    :, -floor:
                ] + (np.roll(curr_wf_neg, floor, axis=1) * (ceil - shift_))[:, -floor:]
        wf_final = temp

        return wf_final