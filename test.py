from datasets.dataset_msrs import MSRS_dataset, Generator
from utils.common import YCbCr2RGB, tensor2img, save_img_single
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import argparse
import os
import torch
import torch.nn as nn
from models.model import Fusion  ##########change model here##########

# MSRS RoadScene TNO M3FD FMB MRI_CT MRI_PET MRI_SPECT
def test_fusion(args):
    save_dir = '/18851096398/SFMFusion/two_stage_2_3/MSRS_size=128_weight=1/epoch_14'
    os.makedirs(save_dir, exist_ok=True)
    fusion_model_path = '/18851096398/SFMFusion/two_stage_2_3/MSRS_size=128_weight=1/epoch_14.pth'

    net = Fusion(in_chans=1,
                 out_chans=1,
                 embed_dim=32, ##########change here##########
                 depths=(2, 2, 2),  ##########change here##########
                 mlp_ratio=2.,
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ).cuda()

    net.load_state_dict(torch.load(fusion_model_path))

    net.eval()
    print('net load done!')
    
    test_dataset = MSRS_dataset(base_dir=args.root_path, split="test",
                                transform=transforms.Compose(
                                    [Generator()]))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,   
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_bar):
            # 获取原始图像和尺寸
            image_vis_y_batch, image_ir_batch = sampled_batch['image_vis_y'], sampled_batch['image_ir']
            original_size = image_vis_y_batch.shape[-2:]  # (H, W)

            # 确保图像尺寸是8的倍数，裁剪多余部分
            def crop_to_multiple_of_8(tensor):
                H, W = tensor.shape[-2:]
                new_H = H - (H % 8)  # 裁剪到8的倍数
                new_W = W - (W % 8)  # 裁剪到8的倍数
                return tensor[..., :new_H, :new_W]  # 裁剪多余的部分

            # 裁剪可见光和红外图像到8的倍数
            image_vis_y_batch = crop_to_multiple_of_8(image_vis_y_batch).cuda()
            image_ir_batch = crop_to_multiple_of_8(image_ir_batch).cuda()

            # 同样裁剪色度分量
            image_vis_cb, image_vis_cr = sampled_batch['image_vis_cb'], sampled_batch['image_vis_cr']
            image_vis_cb, image_vis_cr = image_vis_cb.cuda(), image_vis_cr.cuda()

            # 执行模型推理
            fuse_y,y,ir = net(image_vis_y_batch, image_ir_batch)
            # fuse_y = net(image_vis_y_batch, image_ir_batch)

            # 插值回原始尺寸
            fuse_y_resized = torch.nn.functional.interpolate(fuse_y, size=original_size, mode='bilinear',
                                                             align_corners=False)

            # 将 YCbCr 转换为 RGB 图像
            fuse_img = YCbCr2RGB(fuse_y_resized, image_vis_cb, image_vis_cr)
            #fuse_img = fuse_y_resized

            # 保存图像
            name = sampled_batch['name']
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                single_fuse_img = tensor2img(fuse_img[k])
                save_img_single(single_fuse_img, save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='/18851096398/SFMFusion/data/MSRS')  ##########change here##########
    args = parser.parse_args()
    test_fusion(args)

    print("Test Done!")
