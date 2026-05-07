import numpy as np
import torch
import model2D2 as Model
import argparse
import core.logger as Logger
import core.metrics as Metrics
import os
import random
import scipy.io as scio
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_256_256_val.json',
                    help='JSON file for configuration')
parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                    help='Run either train(training) or val(generation)', default='val')
parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
parser.add_argument('-debug', '-d', action='store_true')
parser.add_argument('-enable_wandb', action='store_true')
parser.add_argument('-log_wandb_ckpt', action='store_true')
parser.add_argument('-log_eval', action='store_true')
args = parser.parse_args(args=[])
opt = Logger.parse(args)
opt = Logger.dict_to_nonedict(opt)

# 基础路径
base_path = "./data/Yfgj_i_slices/label_spec"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def process_xulie(xulie):
    print(f"\n===== 正在处理 xulie = {xulie} =====")

    dynamic_real_data_path = f"{base_path}{xulie}"
    opt['datasets']['test']['real_data_path'] = dynamic_real_data_path

    data_path = f"{opt['datasets']['test']['real_data_path']}.mat"
    if not os.path.isfile(data_path):
        print(f"[跳过] xulie {xulie} 对应的 .mat 文件不存在: {data_path}")
        return None  # 跳过当前 xulie
    sample_rate = opt['datasets']['test']['sample_rate']
    ori_spec = scio.loadmat(data_path)['spec']
    N1 = ori_spec.shape[0]
    N2 = ori_spec.shape[1]

    threshold = 0.01
    mask = np.abs(ori_spec) < np.max(np.abs(ori_spec), axis=1, keepdims=True)[0] * threshold
    ori_spec[mask] = 0

    phase_input, pad_shape, pad_height, pad_width = block_data(ori_spec, 64)
    xx = np.fft.ifft2(ori_spec)

    random_integer = random.randint(1, 1000)
    print(f"xulie {xulie} 使用随机 mask 编号: {random_integer}")
    Mask = scio.loadmat(f"./data2D/0.08_2D_mask-{N1}-{N2}/Mask_{random_integer}.mat")['Mask']
    U = Mask

    NUS_FID = np.multiply(U, xx)
    NOISE_input = np.fft.fft2(NUS_FID)
    NUS_FID, _, _, _ = block_data(NUS_FID, 64)
    NOISE_input, _, _, _ = block_data(NOISE_input, 64)

    GT_label = np.fft.fft2(xx)
    GT_label, _, _, _ = block_data(GT_label, 64)
    block_num = phase_input.shape[0]

    max_amp = np.max(np.abs(NOISE_input))
    NUS_FID_tensor = torch.complex(
        torch.from_numpy(NUS_FID.real).float(),
        torch.from_numpy(NUS_FID.imag).float()
    ) / (max_amp * 11)
    NOISE_input_tensor = torch.complex(
        torch.from_numpy(NOISE_input.real).float(),
        torch.from_numpy(NOISE_input.imag).float()
    ) / (max_amp)
    GT_label_tensor = torch.complex(
        torch.from_numpy(GT_label.real).float(),
        torch.from_numpy(GT_label.imag).float()
    ) / (max_amp * 12.5)

    LR = NUS_FID_tensor.unsqueeze(1)
    SR = torch.cat((NOISE_input_tensor.real.unsqueeze(1), NOISE_input_tensor.imag.unsqueeze(1)), dim=1)
    HR = torch.cat((GT_label_tensor.real.unsqueeze(1), GT_label_tensor.imag.unsqueeze(1)), dim=1)

    val_data = {'LR': LR, 'HR': HR, 'SR': SR}

    diffusion = Model.create_model(opt)
    result_path = '{}/{}'.format(opt['path']['results'], 100)
    os.makedirs(result_path, exist_ok=True)
    diffusion.set_new_noise_schedule(opt['model']['beta_schedule']['val'], schedule_phase='val')

    val_data = diffusion.set_device(val_data)
    diffusion.feed_data(val_data)
    diffusion.test(continous=False)
    visuals = diffusion.get_current_visuals()

    try:
        fake_img = reconstruct_data(
            torch.complex(visuals['INF'][0:block_num, 0], visuals['INF'][0:block_num, 1]).float().cpu().numpy(),
            pad_shape, pad_height, pad_width
        )
        sr_img = reconstruct_data(
            torch.complex(visuals['SR'][-block_num:, 0], visuals['SR'][-block_num:, 1]).float().cpu().numpy(),
            pad_shape, pad_height, pad_width
        )
        sr_spec = reconstruct_data(
            torch.complex(visuals['SR_Spec'][-block_num:, 0], visuals['SR_Spec'][-block_num:, 1]).float().cpu().numpy(),
            pad_shape, pad_height, pad_width
        )
        hr_img = reconstruct_data(
            torch.complex(visuals['HR'][0:block_num, 0], visuals['HR'][0:block_num, 1]).float().cpu().numpy(),
            pad_shape, pad_height, pad_width
        )
        lr_img = reconstruct_data(
            visuals['LR'][0:block_num].squeeze(1).float().cpu().numpy(),
            pad_shape, pad_height, pad_width
        )
    except NameError:
        fake_img = torch.complex(visuals['INF'][0, 0], visuals['INF'][0, 1]).squeeze().float().cpu().numpy()
        sr_img = torch.complex(visuals['SR'][-1, 0], visuals['SR'][-1, 1]).squeeze().float().cpu().numpy()
        sr_spec = torch.complex(visuals['SR_Spec'][-1, 0], visuals['SR_Spec'][-1, 1]).squeeze().float().cpu().numpy()
        hr_img = torch.complex(visuals['HR'][0, 0], visuals['HR'][0, 1]).squeeze().float().cpu().numpy()
        lr_img = visuals['LR'][0].squeeze().float().cpu().numpy()

    Metrics.save_contour2D(
        nus_spec=fake_img,
        recon_spec=sr_img,
        label_spec=hr_img,
        sr_spec=sr_spec,
        xulie=xulie,
        save_path='{}/{}_{}_hr.png'.format(result_path, 200, xulie)
    )

    plt.show()

    return None

def pad_data(data, block_size):
    height, width = data.shape
    pad_height = (block_size - height % block_size) % block_size
    pad_width = (block_size - width % block_size) % block_size
    padded_data = np.pad(data, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return padded_data, pad_height, pad_width


def block_data(data, block_size):
    padded_data, pad_height, pad_width = pad_data(data, block_size)
    height, width = padded_data.shape
    num_blocks_vertical = height // block_size
    num_blocks_horizontal = width // block_size
    blocks = np.empty((num_blocks_vertical * num_blocks_horizontal, block_size, block_size), dtype=padded_data.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = padded_data[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            blocks[idx] = block
            idx += 1
    return blocks, padded_data.shape, pad_height, pad_width


def reconstruct_data(blocks, pad_shape, pad_height, pad_width):
    height, width = pad_shape
    num_blocks, block_height, block_width = blocks.shape
    num_blocks_vertical = height // block_height
    num_blocks_horizontal = width // block_width
    reconstructed_data = np.empty((height, width), dtype=blocks.dtype)
    idx = 0
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            block = blocks[idx]
            reconstructed_data[i * block_height:(i + 1) * block_height, j * block_width:(j + 1) * block_width] = block
            idx += 1
    final_data = reconstructed_data[
                 :height - (pad_height if pad_height != 0 else 0),
                 :width - (pad_width if pad_width != 0 else 0)
                 ]
    return final_data


if __name__ == '__main__':

    peak_position = [382, 388, 393, 394, 397, 399, 402, 403, 406, 408,
                     411, 412, 414, 415, 416, 417, 420, 421, 424, 428,
                     429, 430, 432, 433, 435, 437, 439, 441, 442, 443,
                     444, 445, 447, 452, 453, 456, 458, 462]
    # for xulie in peak_position:
    for xulie in range(0, 522):
        try:
            process_xulie(xulie)
        except Exception as e:
            print(f"处理 xulie={xulie} 时出错: {e}")