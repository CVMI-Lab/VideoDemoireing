import numpy as np
import torch
import argparse
import cv2, os, glob
import torch.utils.data as data
import torchvision.transforms as transforms
import random
from PIL import Image
from PIL import ImageFile

class data_loader(data.Dataset):

    def __init__(self, args, image_list, mode='train'):
        self.image_list = image_list
        self.args = args
        self.mode = mode
        self.loader = args.LOADER

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        image_in_gt = self.image_list[index]
        number = int(image_in_gt[4:9])
        image_in = 'src_%05d' % number + '.png'
        if self.mode == 'train':
            path_tar = self.args.TRAIN_DATASET + '/target/' + image_in_gt
            path_src = self.args.TRAIN_DATASET + '/source/' + image_in

            #import pdb; pdb.set_trace()

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i//2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i+1)//2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_in = 'src_%05d' % number_tmp + '.png'
                path_aux = self.args.TRAIN_DATASET + 'source' + aux_in
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'pre_train':
                    path_aux = path_src
                path_src_auxs.append(path_aux)

            if self.loader == 'crop':
                x = random.randint(0, self.args.WIDTH - self.args.CROP_SIZE)
                y = random.randint(0, self.args.HEIGHT - self.args.CROP_SIZE)
                #import pdb; pdb.set_trace()
                labels = crop_loader(self.args.CROP_SIZE, x, y, [path_tar])
                moire_imgs = crop_loader(self.args.CROP_SIZE, x, y, [path_src])
                if self.args.NUM_AUX_FRAMES > 0:
                    if self.args.affine_aug:
                        moire_imgs_aux = crop_aug_loader(self.args.CROP_SIZE, x, y, path_src_auxs)  # if use the same x and y
                    else:
                        moire_imgs_aux = crop_loader(self.args.CROP_SIZE, x, y, path_src_auxs)  # if use the same x and y
            elif self.loader == 'reszie':
                labels = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_tar])
                moire_imgs = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_src])
                if self.args.NUM_AUX_FRAMES > 0:
                    moire_imgs_aux = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_src_auxs)
            elif self.loader == 'default':
                labels = default_loader([path_tar])
                moire_imgs = default_loader([path_src])
                if self.args.NUM_AUX_FRAMES > 0:
                    moire_imgs_aux = default_loader(path_src_auxs)

        elif self.mode == 'val':
            path_tar = self.args.TEST_DATASET + '/target/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/source/' + image_in

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES+1):
                if i % 2 == 0:
                    number_tmp = number + i//2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i+1)//2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_in = 'src_%05d' % number_tmp + '.png'
                path_aux = self.args.TEST_DATASET + 'source' + aux_in
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'pre_train':
                    path_aux = path_src
                path_src_auxs.append(path_aux)
            

            if self.loader == 'resize':
                labels = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_tar])
                moire_imgs = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, [path_src])
                if self.args.NUM_AUX_FRAMES > 0:
                    moire_imgs_aux = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_src_auxs)
            else:
                labels = default_loader([path_tar])
                moire_imgs = default_loader([path_src])
                if self.args.NUM_AUX_FRAMES > 0:
                    moire_imgs_aux = default_loader(path_src_auxs)

        else:
            print('Unrecognized mode! Please select either "train" or "val"')
            raise NotImplementedError

        data['in_img'] = moire_imgs
        data['label'] = labels
        data['number'] = number
        if self.args.NUM_AUX_FRAMES > 0:
            data['in_img_aux'] = moire_imgs_aux

        return data

    def __len__(self):
        return len(self.image_list)

def default_loader(path_set):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = default_toTensor(img)
         imgs.append(img)

    return imgs

def crop_loader(crop_size, x, y, path_set):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)

    return imgs
    
    
def crop_aug_loader(crop_size, x, y, path_set):
    imgs = []
    
    def transform_aug(img):
        h,w,c = img.shape
        M = np.float32([[1, 0, 0.02*w], [0, 1, 0.02*h]])
        dst = cv2.warpAffine(img, M, (w, h))
        return dst
    
    def rotation_aug(img):
        h,w,c = img.shape
        scale = random.uniform(0.98, 1.02)
        angle = random.uniform(2, -2)
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
        dst = cv2.warpAffine(img, M, (w, h))
        return dst
    
    
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img_arr = np.array(img)
         img_arr = transform_aug(img_arr)
         img_arr = rotation_aug(img_arr)
         #import pdb; pdb.set_trace()
         img = Image.fromarray(img_arr)
         img = img.crop((x, y, x + crop_size, y + crop_size))
         img = default_toTensor(img)
         imgs.append(img)

    return imgs
    

def resize_loader(resize_size_h, resize_size_w, path_set):
    imgs = []
    for path in path_set:
         img = Image.open(path).convert('RGB')
         img = img.resize((resize_size_w, resize_size_h),Image.BICUBIC)
         img = default_toTensor(img)
         imgs.append(img)

    return imgs

def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)
