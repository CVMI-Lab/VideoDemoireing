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
        self.frames_each_video = args.frames_each_video
        file_type = image_list[0].split('.')[-1]
        self.file_type = '.%s' % file_type

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        data = {}
        index = index 
        image_in_gt = self.image_list[index]
        video_number = image_in_gt.split('/')[-1][0:7]
        number = int(image_in_gt.split('/')[-1][7:12])
        image_in = video_number + '%05d' % number + self.file_type

        assert self.args.NUM_AUX_FRAMES > 0
       
        if self.mode == 'train':
            path_srcs = []
            path_tars = []
            path_src_auxs = []

            if self.args.use_temporal:
                # currently, this code can only use 2 output frames/ 2 branches
                for k in range(2):
                    if number > (self.frames_each_video - 2):
                        number = self.frames_each_video - 2
                    number_cur = number + k
                    path_tar = self.args.TRAIN_DATASET + '/target/' + video_number + '%05d' % number_cur + self.file_type
                    path_src = self.args.TRAIN_DATASET + '/source/' + video_number + '%05d' % number_cur + self.file_type

                    path_srcs.append(path_src)
                    path_tars.append(path_tar)

                    for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                        if i % 2 == 0:
                            number_tmp = number_cur + i // 2 * self.args.FRAME_INTERVAL
                        else:
                            number_tmp = number_cur - (i + 1) // 2 * self.args.FRAME_INTERVAL
                            if number_tmp < 0:
                                number_tmp = number_cur
                        aux_in = video_number + '%05d' % number_tmp + self.file_type
                        path_aux = self.args.TRAIN_DATASET + '/source/' + aux_in
                        if not os.path.isfile(path_aux):
                            path_aux = path_src
                        if self.args.MODE == 'single':
                            path_aux = path_src
                        path_src_auxs.append(path_aux)

            # elif self.args.use_flow:
            #     # currently, this code can only use 2 output frames/ 2 branches
            #     path_flows = []
            #     path_masks = []
            #     for k in range(2):
            #         if number > (self.frames_each_video - 2):
            #             number = self.frames_each_video - 2
            #         number_cur = number + k
            #         path_tar = self.args.TRAIN_DATASET + '/target/' + video_number + '%05d' % number_cur + self.file_type
            #         path_src = self.args.TRAIN_DATASET + '/source/' + video_number + '%05d' % number_cur + self.file_type
            #         # flow between two frames (branch 1 and branch 2)
            #         path_flow = self.args.flow_path + video_number + '%05d' % number_cur + '.npz'
            #         path_mask = self.args.flow_path + video_number + '%05d' % number_cur + '.png'
            #
            #         if not os.path.isfile(path_flow):
            #             path_flow = self.args.flow_path + video_number + '%05d' % number + '.npz'
            #         if not os.path.isfile(path_mask):
            #             path_mask = self.args.flow_path + video_number + '%05d' % number + '.png'
            #
            #         path_srcs.append(path_src)
            #         path_tars.append(path_tar)
            #         path_flows.append(path_flow)
            #         path_masks.append(path_mask)
            #
            #         for i in range(1, self.args.NUM_AUX_FRAMES + 1):
            #             if i % 2 == 0:
            #                 number_tmp = number_cur + i // 2 * self.args.FRAME_INTERVAL
            #             else:
            #                 number_tmp = number_cur - (i + 1) // 2 * self.args.FRAME_INTERVAL
            #                 if number_tmp < 0:
            #                     number_tmp = number_cur
            #             aux_in = video_number + '%05d' % number_tmp + self.frames_each_video
            #             path_aux = self.args.TRAIN_DATASET + '/source/' + aux_in
            #             if not os.path.isfile(path_aux):
            #                 path_aux = path_src
            #             if self.args.MODE == 'single':
            #                 path_aux = path_src
            #             path_src_auxs.append(path_aux)

            else:
                path_tar = self.args.TRAIN_DATASET + '/target/' + image_in_gt
                path_src = self.args.TRAIN_DATASET + '/source/' + image_in

                path_srcs.append(path_src)
                path_tars.append(path_tar)

                for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                    if i % 2 == 0:
                        number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                    else:
                        number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                        if number_tmp < 0:
                            number_tmp = number
                    aux_in = video_number + '%05d' % number_tmp + self.file_type
                    path_aux = self.args.TRAIN_DATASET + '/source/' + aux_in
                    if not os.path.isfile(path_aux):
                        path_aux = path_src
                    if self.args.MODE == 'single':
                        path_aux = path_src
                    path_src_auxs.append(path_aux)
                
            if self.loader == 'crop':
                x = random.randint(0, self.args.WIDTH - self.args.CROP_SIZE)
                y = random.randint(0, self.args.HEIGHT - self.args.CROP_SIZE)
                labels = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_tars)
                moire_imgs = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_srcs)
                moire_imgs_aux = crop_loader(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_src_auxs)
                # if self.args.use_flow:
                #     flows = crop_loader_flow(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_flows)
                #     masks = crop_loader_mask(self.args.CROP_SIZE, self.args.CROP_SIZE, x, y, path_masks)

            elif self.loader == 'reszie':
                labels = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_tars)
                moire_imgs = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_srcs)
                moire_imgs_aux = resize_loader(self.args.RESIZE_H, self.args.RESIZE_W, path_src_auxs)

            elif self.loader == 'default':
                labels = default_loader(path_tars)
                moire_imgs = default_loader(path_srcs)
                moire_imgs_aux = default_loader(path_src_auxs)

        elif self.mode == 'val':
            path_tar = self.args.TEST_DATASET + '/target/' + image_in_gt
            path_src = self.args.TEST_DATASET + '/source/' + image_in

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_in = video_number + '%05d' % number_tmp + self.file_type
                path_aux = self.args.TEST_DATASET + '/source/' + aux_in
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'single':
                    path_aux = path_src
                path_src_auxs.append(path_aux)

            labels = default_loader([path_tar])
            moire_imgs = default_loader([path_src])
            moire_imgs_aux = default_loader(path_src_auxs)

        elif self.mode == 'test':
            path_src = self.args.TEST_DATASET + '/source/' + image_in

            path_src_auxs = []
            for i in range(1, self.args.NUM_AUX_FRAMES + 1):
                if i % 2 == 0:
                    number_tmp = number + i // 2 * self.args.FRAME_INTERVAL
                else:
                    number_tmp = number - (i + 1) // 2 * self.args.FRAME_INTERVAL
                    if number_tmp < 0:
                        number_tmp = number
                aux_in = video_number + '%05d' % number_tmp + self.file_type
                path_aux = self.args.TEST_DATASET + '/source/' + aux_in
                if not os.path.isfile(path_aux):
                    path_aux = path_src
                if self.args.MODE == 'single':
                    path_aux = path_src
                path_src_auxs.append(path_aux)

            moire_imgs = default_loader([path_src])
            if self.args.NUM_AUX_FRAMES > 0:
                moire_imgs_aux = default_loader(path_src_auxs)

        else:
            print('Unrecognized mode! Please select either "train" or "val" or "test"')
            raise NotImplementedError

        data['number'] = video_number + '%05d' % number
        data['in_img'] = moire_imgs
        data['in_img_aux'] = moire_imgs_aux

        if not self.mode == 'test':
            data['label'] = labels

        # if self.mode == 'train' and self.args.use_flow:
        #     data['flow'] = flows
        #     data['mask'] = masks

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


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def crop_loader(crop_size_x, crop_size_y, x, y, path_set, pad_size=100, pad=False):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        if pad:
            img = add_margin(img, pad_size, pad_size, pad_size, pad_size, (123, 117, 104))
        img = img.crop((x, y, x + crop_size_x, y + crop_size_y))
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def crop_loader_mask(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.crop((x, y, x + crop_size_x, y + crop_size_y))
        img = 1 - default_toTensor(img)
        imgs.append(img)

    return imgs
    
    
def crop_loader_flow(crop_size_x, crop_size_y, x, y, path_set):
    imgs = []
    for path in path_set:
        img = np.load(path)['flow']
        img = img[y:(y+crop_size_y), x:(x+crop_size_x), :]
        img = default_toTensor(img)
        imgs.append(img)
        
    return imgs


def resize_loader(resize_size_h, resize_size_w, path_set):
    imgs = []
    for path in path_set:
        img = Image.open(path).convert('RGB')
        img = img.resize((resize_size_w, resize_size_h), Image.BICUBIC)
        img = default_toTensor(img)
        imgs.append(img)

    return imgs


def default_toTensor(img):
    t_list = [transforms.ToTensor()]
    composed_transform = transforms.Compose(t_list)

    return composed_transform(img)
