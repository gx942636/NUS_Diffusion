import nmrglue as ng
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def compute_rlne(x_true, x_pred):
    x_pred = np.array(x_pred)
    x_true = np.array(x_true)
    l2_error = np.linalg.norm(x_pred - x_true, ord=2)
    l2_true = np.linalg.norm(x_true, ord=2)
    rlne = l2_error / l2_true
    return rlne


def projection_3d(data, projection_type='skyline', iz1=1, izn=512, peak_position=None, recon=False):
    '''
    Perform projection of 3D data along the third dimension.
    '''
    if projection_type == 'skyline':
        peak_position = [391, 395, 397, 399, 400, 401, 402, 404, 409, 410,
                         411, 412, 413, 415, 416, 418, 419, 421, 422, 424,
                         425, 427, 428, 430, 431, 432, 435, 438, 440, 444,
                         447, 448, 458, 461, 463, 465, 468]

        # out = np.max(data[:, iz1 - 1:izn - 1, :], 1) / np.max(np.max(data[:, iz1 - 1:izn - 1, :], 1))
        out = np.max(data[:, peak_position, :], 1) / np.max(np.max(data[:, peak_position, :], 1))
        # if not recon:
        #     out = np.max(data[:, iz1 - 1:izn - 1, :], 1) / np.max(np.max(data[:, iz1 - 1:izn - 1, :], 1))
        # else:
        #     out = np.max(data[:, peak_position, :], 1) / np.max(np.max(data[:, peak_position, :], 1))

        return out

    elif projection_type == 'sum':
        out = np.abs(np.sum(data[:, :, iz1 - 1:izn - 1], 2))
        return out

def iz1_izn(data_type):
    if data_type == 'PSRP':
        N1 = 32
        N2 = 32
        iz1 = 80
        izn = 180
        ppm1 = np.linspace(103.97273003, 133.59290332, N1)  # N15 32
        ppm2 = np.linspace(168.243315, 185.48343621, N2)  # CO  32

    elif data_type == 'A3DK08':
        N1 = 64
        N2 = 40
        # 观察间接维投影发现，直接维峰主要是在60-180之间
        iz1 = 70
        izn = 180
        ppm1 = np.linspace(164.54, 189.07, N1)  # C13  64
        ppm2 = np.linspace(101.83, 134.74, N2)  # N15  40

        peak_position = [327, 331, 339, 344, 345, 352, 353, 358, 359, 361,
                         362, 363, 366, 367, 369, 370, 375, 377, 384, 385,
                         386, 388, 389, 391, 394, 395, 398, 399, 401, 402,
                         403, 405, 407, 409, 412, 415, 417, 419, 422, 424,
                         434]

    elif data_type == 'specCnoesy':
        N1 = 150
        N2 = 64
        iz1 = 190
        izn = 370
        ppm1 = np.linspace(-2.69277800, 12.3113880, N1);  # 1H   150
        ppm2 = np.linspace(71.67634631, 9.71591854, N2);  # 13C  64

    elif data_type == 'mousethia':
        N1 = 55
        N2 = 40
        iz1 = 290
        izn = 330
        ppm1 = np.linspace(15.552939, 75.25997947, N1)  # CACB 55
        ppm2 = np.linspace(101.26288858, 135.83123601, N2)  # N15  40

    elif data_type == 'Ecolinird':
        N1 = 60
        N2 = 36
        iz1 = 100
        izn = 200
        ppm1 = np.linspace(11.64680295, 81.64694662, N1)  # CACB  60
        ppm2 = np.linspace(99.03853462, 134.62348861, N2)  # N15   36


    elif data_type == 'Altg':
        N1 = 64
        N2 = 128
        iz1 = 120
        izn = 210
        ppm1 = np.linspace(167.62, 187.64, N1)  # C13  32
        ppm2 = np.linspace(104.26, 133.89, N2)  # N15  40
        peak_position = [382, 388, 393, 394, 397, 399, 402, 403, 406, 408,
                         411, 412, 414, 415, 416, 417, 420, 421, 424, 428,
                         429, 430, 432, 433, 435, 437, 439, 441, 442, 443,
                         444, 445, 447, 452, 453, 456, 458, 462]

    elif data_type == 'Yorp':
        N1 = 32
        N2 = 50
        iz1 = 140
        izn = 190
        ppm1 = np.linspace(12.46400863, 82.46398819, N1)  # CACB  34
        ppm2 = np.linspace(106.81524902, 130.81472222, N2)  # N15  34

    elif data_type == 'Yfgj':
        N1 = 34
        N2 = 34
        iz1 = 130
        izn = 220
        ppm1 = np.linspace(41.25573204, 71.25703962, N1)  # C13  34
        ppm2 = np.linspace(103.7399968, 132.74115, N2)  # N15  34
        peak_position = [391, 395, 397, 399, 400, 401, 402, 404, 409, 410,
                         411, 412, 413, 415, 416, 418, 419, 421, 422, 424,
                         425, 427, 428, 430, 431, 432, 435, 438, 440, 444,
                         447, 448, 458, 461, 463, 465, 468]

    elif data_type == 'Ykvr':
        N1 = 40
        N2 = 45
        iz1 = 100
        izn = 140
        ppm1 = np.linspace(102.04888518, 134.9338859, N1)  # N15  40
        ppm2 = np.linspace(-1.82879304, 11.52294876, N2)  # HN  45

    elif data_type == 'hncacb_6240' or data_type == 'hncacb_6240_1':
        N1 = 32
        N2 = 64
        # iz1 = 150
        # izn = 440
        iz1 = 100
        izn = 440

    return iz1, izn, peak_position


if __name__ == '__main__':
    data_type = 'Yfgj'  # 'Altg' 'A3DK08' 'mousethia' 'Ecolinird' 'PSRP' 'hncacb_6240'  'Yfgj'
    model_type = 'Res-Conformer'  # Res-Conformer Dense-CNN JTF-Net  SMILE
    sample_rates = [15]  # 8, 10, 15, 20, 25 | 5,10,15,20,25,30,40,50,60,70,80,90
    iz1, izn, peak_position = iz1_izn(data_type)
    # if data_type == 'mousethia' or data_type == 'Ecolinird':
    #     dic, data_label = ng.pipe.read(f"F:/NMRPipe/ReadNMRData/{data_type}/{data_type}_abs_3D_pad.ft")
    # else:
    #     # dic, data_label = ng.pipe.read(f"F:/NMRPipe/ReadNMRData/{data_type}/{data_type}_3D_pad_tmp.ft")

    dic, data_label = ng.pipe.read(f'./{data_type}_label.ft')
    dic, data_recon = ng.pipe.read(f'./{data_type}_recon.ft')
    # if data_type in['Ecolinird','mousethia']:
    #     data_label = np.abs(data_label)

    if data_type == 'hncacb_6240' or data_type == 'hncacb_6240_1':
        data_label = np.abs(data_label.transpose(1, 0, 2))
    data_label1 = data_label
    print(data_label.shape)
    # data_label = np.fft.fftshift(data_label)
    data_label = projection_3d(data_label, projection_type='skyline', iz1=iz1, izn=izn, peak_position=peak_position)
    data_label = np.fft.fftshift(data_label)

    # data_recon = np.fft.fftshift(data_recon)
    data_recon = projection_3d(data_recon, projection_type='skyline', iz1=iz1, izn=izn, peak_position=peak_position,recon=False)
    data_recon = np.fft.fftshift(data_recon)

    # 要用下面两行找到，在重建之前的哪些label切片才能组成真正的谱图(不少峰)，因为重建之后，我们用了np.fft.fftshift，所以显示的结果与重建之前的是不对应的
    # 使用matlab在直接维进行寻峰，基本就能确定是重建之前的哪些label切片组成的真正谱图
    # A3DK08的直接维度切片是320-440  Altg的直接维度切片是350-480
    # data_recon = projection_3d(data_label1, projection_type='skyline', iz1=320, izn=440, peak_position=peak_position,recon=True)
    # data_recon = np.fft.fftshift(data_recon)
    print(data_label.shape)
    plt.figure()
    level = 8
    plt.subplot(1, 2, 1)
    plt.contour(data_label, cmap='viridis', levels=level)
    plt.title('Label')
    plt.subplot(1, 2, 2)
    plt.contour(data_recon, cmap='viridis', levels=level)
    plt.title('Recon')
    plt.show()

    label_spec = data_label
    recon_spec = data_recon
    scio.savemat('../label_spec.mat', {'label_spec': label_spec})
    scio.savemat('../recon_spec.mat', {'recon_spec': recon_spec})

    # for mask_idx in range(1, 20):
    #     for sample_rate in sample_rates:
    #         if model_type == 'Admm_pnp':
    #             if data_type == 'mousethia' or data_type == 'Ecolinird':
    #                 dic, data_recon = ng.pipe.read(
    #                     f"F:/NMRPipe/ReadNMRData/{data_type}/recon_results/{model_type}/{data_type}_k17_unsort_5-20/{sample_rate}/{data_type}_{sample_rate}_{mask_idx}_abs_pad.ft")
    #             else:
    #                 dic, data_recon = ng.pipe.read(
    #                     f"F:/NMRPipe/ReadNMRData/{data_type}/recon_results/{model_type}/{data_type}_k17_unsort_5-20/{sample_rate}/{data_type}_{sample_rate}_{mask_idx}_pad.ft")
    #         else:
    #             if data_type == 'mousethia' or data_type == 'Ecolinird':
    #                 dic, data_recon = ng.pipe.read(
    #                     f"F:/NMRPipe/ReadNMRData/{data_type}/recon_results/{model_type}/{data_type}/{sample_rate}/{data_type}_{sample_rate}_{mask_idx}_abs_pad.ft")
    #             else:
    #                 dic, data_recon = ng.pipe.read(
    #                     f"F:/NMRPipe/ReadNMRData/{data_type}/recon_results/{model_type}/{data_type}/{sample_rate}/{data_type}_{sample_rate}_{mask_idx}_pad.ft")
    #         if data_type == 'hncacb_6240' or data_type == 'hncacb_6240_1':
    #             data_recon = data_recon.transpose(1, 0, 2)
    #
    #         spec_recon = projection_3d(data_recon, projection_type='skyline', iz1=iz1, izn=izn)
    #         print(mask_idx, 'RLNE', compute_rlne(spec_origin, spec_recon))
    #         plt.figure()
    #         plt.subplot(1, 2, 1)
    #         plt.contour(spec_origin, cmap='viridis', levels=12)
    #         plt.title('Label')
    #         plt.subplot(1, 2, 2)
    #         plt.contour(spec_recon, cmap='viridis', levels=12)
    #         plt.title(f'Reconstruction: NUS Rate={sample_rate}  RLNE={compute_rlne(spec_origin, spec_recon):.4f}')
    #         plt.show()