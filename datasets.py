import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from PIL import Image


def make_dataset(root, name, spilt):
    dir = os.path.join(root, name, spilt)
    rgb = []
    depth = []
    gt = []
    d_name = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if fname[-6:-4] == 'or':
                color_path = os.path.join(root, fname)
                rgb.append(color_path)
            elif fname[-6:-4] == 'th':
                depth_path = os.path.join(root, fname)
                depth.append(depth_path)
            elif fname[-6:-4] == 'gt':
                gt_path = os.path.join(root, fname)
                gt.append(gt_path)
        d_name.append(name)
    return rgb, depth, gt, d_name


def pil_loader(path, ratio=1.):
    return transforms.ToTensor()(Image.open(path)) / ratio


class CompletionDataset(data.Dataset):
    def __init__(self, root, datasets='NYUDepth', spilt='train', data_len=0, loader=pil_loader):
        assert data_len % 3 == 0
        self.datasets = datasets
        datasets_name = [self.datasets]
        data_len_eval = data_len // 3
        self.rgb = []
        self.depth = []
        self.gt = []
        self.d_name = []
        for i in range(len(datasets_name)):
            rgb, depth, gt, d_name = make_dataset(root, datasets_name[i], spilt)
            if data_len > 0:
                self.rgb += rgb[:int(data_len_eval)]
                self.depth += depth[:int(data_len_eval)]
                self.gt += gt[:int(data_len_eval)]
                self.d_name += d_name[:int(data_len_eval)]
            else:
                self.rgb += rgb
                self.depth += depth
                self.gt += gt
                self.d_name += d_name
        norm_size = [192, 288]
        kitti_size = [256, 1216]
        if spilt == 'train':
            size_diml = int(192 * np.random.uniform(1.0, 1.2))
            self.diml_t_gt = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.diml_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.diml_t_depth = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomErasing(p=0.9, scale=(0.05, 0.4), value=0, inplace=True),
            )

            size_nyu = int(240 * np.random.uniform(1.0, 1.2))
            self.nyu_t_gt = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.nyu_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.nyu_t_depth = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomErasing(p=0.9, scale=(0.05, 0.4), value=0, inplace=True),
            )

            size_sun = int(216 * np.random.uniform(1.0, 1.2))
            self.sun_t_gt = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.sun_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.sun_t_depth = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomErasing(p=0.9, scale=(0.05, 0.4), value=0, inplace=True),
            )

            size_kitti = int(kitti_size[1] * np.random.uniform(1.0, 1.05))
            self.kitti_t_gt = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.kitti_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.kitti_t_depth = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.RandomErasing(p=0.9, scale=(0.05, 0.4), value=0, inplace=True),
            )

        else:
            size_diml = 192
            self.diml_t_gt = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.diml_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.diml_t_depth = torch.nn.Sequential(
                transforms.Resize(size_diml),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )

            size_nyu = 240
            self.nyu_t_gt = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.nyu_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.nyu_t_depth = torch.nn.Sequential(
                transforms.Resize(size_nyu),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )

            size_sun = 216
            self.sun_t_gt = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.sun_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.sun_t_depth = torch.nn.Sequential(
                transforms.Resize(size_sun),
                transforms.CenterCrop(norm_size),
                transforms.ConvertImageDtype(torch.float),
            )
            size_kitti = int(kitti_size[1] * np.random.uniform(1.0, 1.2))
            self.kitti_t_gt = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
            )
            self.kitti_t_rgb = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            )
            self.kitti_t_depth = torch.nn.Sequential(
                transforms.Resize(size_kitti),
                transforms.CenterCrop(kitti_size),
                transforms.ConvertImageDtype(torch.float),
            )
        self.loader = loader

    def __getitem__(self, index):
        with torch.no_grad():
            ret = {}
            if self.d_name[index] == 'DIML':
                ret['rgb'] = self.diml_t_rgb(self.loader(self.rgb[index]))
                ret['gt'] = self.diml_t_gt(self.loader(self.gt[index], 1000))
                ret['depth'] = self.diml_t_depth(self.loader(self.depth[index], 1000))

            elif self.d_name[index] == 'NYUDepth':
                ret['rgb'] = self.nyu_t_rgb(self.loader(self.rgb[index]))
                ret['gt'] = self.nyu_t_gt(self.loader(self.gt[index], 0.1))
                ret['depth'] = self.nyu_t_depth(self.loader(self.depth[index], 0.1))

            elif self.d_name[index] == 'SUNRGBD':
                ret['rgb'] = self.sun_t_rgb(self.loader(self.rgb[index]))
                ret['gt'] = self.sun_t_gt(self.loader(self.gt[index], 6553.5))
                ret['depth'] = self.sun_t_depth(self.loader(self.depth[index], 6553.5))

            elif self.d_name[index] == 'Kitti':
                ret['rgb'] = self.kitti_t_rgb(self.loader(self.rgb[index]))
                ret['gt'] = self.kitti_t_gt(self.loader(self.gt[index], 2560))
                ret['depth'] = self.kitti_t_depth(self.loader(self.depth[index], 2560))

            ret['mask'] = transforms.ConvertImageDtype(torch.float)((ret['depth'] <= 1e-3))
            return ret

    def __len__(self):
        return len(self.rgb)
