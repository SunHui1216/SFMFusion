import torch
from PIL import Image
import numpy as np
import cv2


def RGB2YCbCr(rgb_image):  # input: rgb_image: [3, H, W]
    R = rgb_image[0:1]  # [1, H, W]
    G = rgb_image[1:2]  # [1, H, W]
    B = rgb_image[2:3]  # [1, H, W]
    Y = 0.299 * R + 0.587 * G + 0.114 * B  # [1, H, W]
    Cb = (B - Y) * 0.564 + 0.5  # [1, H, W]
    Cr = (R - Y) * 0.713 + 0.5  # [1, H, W]

    Y = Y.clamp(0.0, 1.0)  # Clamp values to be between 0 and 1
    Cb = Cb.clamp(0.0, 1.0).detach()  # Detach from computation graph
    Cr = Cr.clamp(0.0, 1.0).detach()

    return Y, Cb, Cr  # output: Y, Cb, Cr: [1, H, W]


def YCbCr2RGB(Y, Cb, Cr):  # input:Y, Cb, Cr:[B, 1, H, W]
    YCbCr = torch.cat([Y, Cb, Cr], dim=1)
    B, C, H, W = YCbCr.shape
    im_flat = YCbCr.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [0.0, -0.344, 1.773], [1.403, -0.714, 0.0]]
                       ).to(Y.device)
    bias = torch.tensor([0.0, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    rgb_image = temp.reshape(B, H, W, C).transpose(1, 3).transpose(2, 3)
    rgb_image = rgb_image.clamp(0, 1.0)
    return rgb_image  # output:rgb_image:[B, 3, H, W]


# tensor to array to PIL,used in test.py
def tensor2img(img):
    img = img.cpu().float().numpy()
    # print(img.shape)
    img = np.transpose(img, (1, 2, 0)) * 255.0  # [C,H,W] to [H,W,C]  0,1 to 0,255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    # img = np.squeeze(img,axis=2)
    # img = Image.fromarray(img,mode = 'L')

    # for ir
    # img = np.squeeze(img,axis=2)
    # img = Image.fromarray(img,mode = 'L')
    return img


# save PIL,used in test.py
def save_img_single(img, name):
    img.save(name)


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


