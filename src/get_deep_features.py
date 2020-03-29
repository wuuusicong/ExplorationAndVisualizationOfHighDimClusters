from collections import defaultdict

import numpy as np
import os
import pickle

import torch
from PIL import Image
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.transforms import transforms


def default_loader(path):
    return Image.open(path).convert('RGB')


def default_filelist_reader(filelist):
    im_list = []
    with open(filelist, 'r') as rf:
        for line in rf.readlines():
            im_path = line.strip()
            im_list.append(im_path)
    return im_list


class ImageLabelFilelist(data.Dataset):
    def __init__(self,
                 root,
                 filelist,
                 transform=None,
                 filelist_reader=default_filelist_reader,
                 loader=default_loader,
                 return_paths=True):
        self.root = root
        self.im_list = filelist_reader(os.path.join(filelist))
        self.transform = transform
        self.loader = loader
        self.imgs = [(im_path.split('/')[1], int(im_path.split('/')[0])) for im_path in self.im_list]
        self.label_to_images = defaultdict(self.empty_list)
        for im, gt_label in self.imgs:
            self.label_to_images[gt_label] += [im]
        self.classes_centers = []
        self.return_paths = return_paths
        print('Data loader')
        print("\tRoot: %s" % root)
        print("\tList: %s" % filelist)
        # print("\tNumber of classes: %d" % (len(self.classes)))

    def empty_list(self):
        return []

    def __getitem__(self, index):
        im_path, gt_label = self.imgs[index]
        full_img_path = os.path.join(self.root, im_path)
        # full_img_path = self.root + '/' + im_path
        img = self.loader(full_img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, gt_label, full_img_path
        else:
            return img, gt_label

    def __len__(self):
        return len(self.imgs)


def loader_from_list(
        root,
        file_list,
        batch_size,
        new_size=None,
        height=128,
        width=128,
        crop=True,
        num_workers=4,
        shuffle=True,
        center_crop=False,
        return_paths=True,
        drop_last=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    if center_crop or True:
        transform_list = [transforms.CenterCrop((height, width))] + \
                         transform_list if crop else transform_list
    else:
        transform_list = [transforms.RandomCrop((height, width))] + \
                         transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list \
        if new_size is not None else transform_list
    # if not center_crop:
    #     transform_list = [transforms.RandomHorizontalFlip()] + transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageLabelFilelist(root,
                                 file_list,
                                 transform,
                                 return_paths=return_paths)
    loader = DataLoader(dataset,
                        batch_size,
                        shuffle=shuffle,
                        drop_last=drop_last,
                        num_workers=num_workers)
    return loader

loader = loader_from_list(
            root="../data/ImageNetCropped/animals_faces",
            file_list="../data/ImageNetCropped/animals_faces_list_train_partial.txt",
            batch_size=8,
            new_size=140,
            height=128,
            width=128,
            center_crop=True,
            crop=True,
            num_workers=1)


def _init_embedding():
    vgg = models.vgg16(pretrained=True)
    vgg.train(False)
    new_classifier = nn.Sequential(*(list(vgg.classifier.children())[:-1]))
    vgg.classifier = new_classifier
    if torch.cuda.is_available():
        vgg = vgg.to(torch.device('cuda'))
    paths_list = []
    embeddings_list = []
    labels_list = []
    for inputs, label, paths in loader:
        if torch.cuda.is_available():
            output = vgg(inputs.cuda())
        else:
            output = vgg(inputs)
        for idx, path in enumerate(paths):
            embeddings_list.append(output.data[idx].detach().cpu().numpy())
            labels_list.append(label.data[idx].item())
            paths_list.append(f"{path}\n")
            # self.im2vector[path] = output.data[idx]
    embeddings_array = np.array(embeddings_list)
    labels_array = np.array(labels_list)
    with open("vgg_animals_embeddings.pickle", "wb") as f:
        pickle.dump(embeddings_array, f)
    with open("vgg_animals_labels.pickle", "wb") as f:
        pickle.dump(labels_array, f)
    with open("vgg_data.txt", "w") as f:
        f.writelines(paths_list)


if __name__ == '__main__':
    _init_embedding()