import torch
import torch.nn as nn
import torch.nn.functional as F


class Int_loss(nn.Module):
    def __init__(self):
        super(Int_loss, self).__init__()

    def forward(self, image_y, image_ir, fuse_y):
        intensity_max = torch.max(image_y, image_ir)
        loss_intensity = F.l1_loss(fuse_y, intensity_max)

        return loss_intensity


class Grad_loss(nn.Module):
    def __init__(self):
        super(Grad_loss, self).__init__()
        self.sobel = Sobel_xy()

    def forward(self, image_y, image_ir, fuse_y):
        image_y_grad = self.sobel(image_y)
        image_ir_grad = self.sobel(image_ir)
        grad_max = torch.max(image_y_grad, image_ir_grad)
        fuse_y_grad = self.sobel(fuse_y)
        loss_grad = F.l1_loss(grad_max, fuse_y_grad)

        return loss_grad


class Sobel_xy(nn.Module):
    def __init__(self):
        super(Sobel_xy, self).__init__()
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_y = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).cuda()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).cuda()

    def forward(self, x):
        sobel_x = F.conv2d(x, self.weight_x, padding=1)
        sobel_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.abs(sobel_x) + torch.abs(sobel_y)
