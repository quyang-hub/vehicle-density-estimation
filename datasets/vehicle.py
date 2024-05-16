from PIL import Image
import torch.utils.data as data
import os
import torch
import torchvision.transforms.functional as F
from torchvision import datasets,transforms
import random
import numpy as np
import scipy.io as sio
import scipy.io
from skimage import io


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    i = random.randint(0, res_h)
    res_w = im_w - crop_w
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map

    # 快速创建离散映射
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h * im_width + p_w).to(torch.int64)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        # 256
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)
        target = np.ones(len(keypoints))

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(), st_size, torch.from_numpy(target.copy()).float()


class Trancos(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train',image_list=None):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")
        self.im_list = []
        for im in image_list:
            temp = os.path.join(self.root_path, 'images', im)
            self.im_list.append(temp)
        print('number of img [{}]: {}'.format(method, len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        name = os.path.basename(img_path).split('.')[0]
        gd_path = os.path.join(self.root_path, 'images', '{}.txt'.format(name))
        keypoints = []
        with open(gd_path) as f:
            for line in f:
                x, y = line.split()
                x, y = int(x) - 1, int(y) - 1
                keypoints.append((x, y))
        X, mask = self.load_example(name,keypoints)
        X = X * mask
        if self.method == 'train':
            return self.train_transform(X, keypoints)
        elif self.method == 'val':
            X = Image.fromarray(np.uint8(X))
            wd, ht = X.size
            st_size = 1.0 * min(wd, ht)
            if st_size < self.c_size:
                rr = 1.0 * self.c_size / st_size
                wd = round(wd * rr)
                ht = round(ht * rr)
                st_size = 1.0 * min(wd, ht)
                X = X.resize((wd, ht), Image.BICUBIC)
            X = self.trans(X)
            density = density_map(
                (X.shape[1], X.shape[2]),
                keypoints,
                1e3*np.ones((len(keypoints), 2)),
                out_shape=None)
            return X, len(keypoints), name, density

    def train_transform(self, img, keypoints):
        img = Image.fromarray(np.uint8(img))
        wd, ht = img.size
        st_size = int(1.0 * min(wd, ht))
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = np.array(list(keypoints)) * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = np.array(keypoints) - np.array([j, i])
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])
        keypoints = np.array(keypoints)
        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)
        target = np.ones(len(keypoints))

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float(), st_size, torch.from_numpy(target.copy()).float()


    def load_example(self, img_f,img_centers):
        X = io.imread(os.path.join(self.root_path, 'images', '{}.jpg'.format(img_f)))
        mask = scipy.io.loadmat(os.path.join(self.root_path, 'images', '{}mask.mat'.format(img_f)))['BW']
        mask = mask[:, :, np.newaxis].astype('float32')
        return X, mask

def density_map(shape, centers, gammas, out_shape=None):
    if out_shape is None:
        D = np.zeros(shape)
    else:
        D = np.zeros(out_shape)
    for i, (x, y) in enumerate(centers):
        D += gauss2d(shape, (x, y), gammas[i], out_shape=out_shape)
    return D

def gauss2d(shape, center, gamma, out_shape=None):
    H, W = shape
    if out_shape is None:
        Ho = H
        Wo = W
    else:
        Ho, Wo = out_shape
    x, y = np.array(range(Wo)), np.array(range(Ho))
    x, y = np.meshgrid(x, y)
    x, y = x.astype(float)/Wo, y.astype(float)/Ho
    x0, y0 = float(center[0])/W, float(center[1])/H
    G = np.exp(-(1/2)*(((x - x0)*gamma[0])**2 + ((y - y0)*gamma[1])**2))  # 以(x0, y0)为中心的高斯核
    return G/np.sum(G)
