import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer_resize import trainer_msrs #change here
import torch.nn as nn
from models.model import Fusion  ##########change model here##########
# from thop import profile  # new code

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/18851096398/SFMFusion/data/MSRS')  ##########change here##########
parser.add_argument('--output', type=str, default='/18851096398/SFMFusion/two_stage_2_3')  ##########change here##########
parser.add_argument('--dataset_name', type=str, default='MSRS', help='train on this dataset')
parser.add_argument('--epochs', type=int, default=15)  ##########change here##########
parser.add_argument('--interval', type=int, default=5)  ##########change here##########
parser.add_argument('--img_size', type=int, default=128)  ##########change here##########
parser.add_argument('--batch_size', type=int, default=20, help='batch_size per gpu') ##########change here##########
parser.add_argument('--num_workers', type=int, default=16, help='num_workers')   ##########change here##########
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')     ##########change here##########
parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight_decay')
parser.add_argument('--warmup', type=int, default=1, help='If activated, warp up')
parser.add_argument('--warmup_period', type=int, default=800, help='Warm up iterations')
parser.add_argument('--loss_int_weight', type=int, default=1)
parser.add_argument('--loss_grad_weight', type=int, default=1)  ##########change here##########
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
args = parser.parse_args()

if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset_name
    args.exp = dataset_name + '_size=' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_weight=' + str(args.loss_grad_weight)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    net = Fusion(in_chans=1,
                 out_chans=1,
                 embed_dim=32, ##########change here##########
                 depths=(2, 2, 2),  ##########change here##########
                 mlp_ratio=2.,
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ).cuda()

    net.load_state_dict(torch.load('/18851096398/SFMFusion/two_stage_1/MSRS_size=128_weight=1/epoch_4.pth'))
    # new code
    # ir = torch.randn(1, 1, 1024, 768).cuda()
    # vi = torch.randn(1, 1, 1024, 768).cuda()
    # flops, params = profile(net, inputs=(ir, vi))
    # print(f"FLOPs: {flops / 1e9} G (十亿次浮点运算)")
    # print(f"Params: {params / 1e6} M (百万个参数)")
    #print(net)
    # 生成配置文件
    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    trainer = {'MSRS': trainer_msrs}
    trainer[dataset_name](args, net, snapshot_path)
