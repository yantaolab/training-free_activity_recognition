# Script to encode the target dataset images using CLIP's image encoders
# We encode the validation and testing splits of each dataset independently
# We also encode few-shot support sets akin to TIP-Adapter for 5 shot configurations (k = 1, 2, 4, 8, 16) [refer Sec. 4.2 of paper]

import os
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import get_augmentation
import torch
from utils.Text_Prompt import *
import numpy as np

import torch.nn.functional as F
import random

class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)

class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


# feature dimensions for each model
feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

global args
global global_step

parser = argparse.ArgumentParser()
# number of augmentations to apply for averaging visual features
parser.add_argument('--config', '-cfg', default='configs/HRI10/HRI10_fewshot.yaml')
parser.add_argument('--log_time', default='')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'], args.log_time)

# dummy parameters for dataloader
random.seed(1)
torch.manual_seed(1)

config = DotMap(config)

k_shot = config.data.k_shot
model_name = config.network.arch
augment_epoch = config.data.augment_epoch
print('Current model: {} and k-shot: {}'.format(model_name, k_shot))

disp_name = model_name
if('/' in model_name):
    disp_name = model_name.replace('/', '')

device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

transform_val = get_augmentation(False, config)
transform_train = get_augmentation(True, config)

fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text).cuda()
model_image = torch.nn.DataParallel(model_image).cuda()
fusion_model = torch.nn.DataParallel(fusion_model).cuda()

if config.data.randaug.N > 0:
    transform_train = randAugment(transform_train, config)

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)

features_path = "./features"

val_features_path = features_path+"/{}_f_val_m{}.pt".format(config.data.dataset, disp_name)
val_targets_path = features_path+"/{}_t_val_m{}.pt".format(config.data.dataset, disp_name)

test_features_path = features_path+"/{}_f_test_m{}.pt".format(config.data.dataset, disp_name)
test_targets_path = features_path+"/{}_t_test_m{}.pt".format(config.data.dataset, disp_name)

train_features_path = features_path+"/{}_f_train_m{}_k{}.pt".format(config.data.dataset, disp_name, k_shot)
train_targets_path = features_path+"/{}_t_train_m{}_k{}.pt".format(config.data.dataset, disp_name, k_shot)

if(os.path.exists(train_features_path) and os.path.exists(train_targets_path)):
    load_train = True
else:
    load_train = False

if(os.path.exists(test_features_path) and os.path.exists(test_targets_path)):
    load_test = True
else:
    load_test = False

if(os.path.exists(val_features_path) and os.path.exists(val_targets_path)):
    load_val = True
else:
    load_val = False

# load few shot dataset
val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                           image_tmpl=config.data.image_tmpl,
                           transform=transform_val, random_shift=config.random_shift)
val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                        pin_memory=True, drop_last=True)

train_data = Action_DATASETS(config.data.train_list, config.data.label_list,
                             num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                             random_shift=config.data.random_shift,
                             transform=transform_train)
train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                          shuffle=False, pin_memory=False, drop_last=True)

test_data = val_data
test_loader = val_loader

if device == "cpu":
    model_text.float()
    model_image.float()
else:
    clip.model.convert_weights(
        model_text)  # Actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model_image)

if config.pretrain:
    if os.path.isfile(config.pretrain):
        print(("=> loading checkpoint '{}'".format(config.pretrain)))
        checkpoint = torch.load(config.pretrain)
        model.load_state_dict(checkpoint['model_state_dict'])
        fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'])
        del checkpoint
    else:
        print(("=> no checkpoint found at '{}'".format(config.pretrain)))

model.eval()
fusion_model.eval()

# ------------------------------------------saving val features------------------------------------------
print('start saving val image features')

if not load_val:
    val_features = []
    val_labels = []
    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            target = target.cuda()
            # encode image
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            val_features.append(image_features)
            val_labels.append(target)
    val_features = torch.cat(val_features)
    val_labels = torch.cat(val_labels)

    assert val_features.shape[0]==len(val_data) and val_features.shape[1]==feat_dims[model_name], 'val_features is not of shape nxC'
    assert val_labels.shape[0]==len(val_data), 'val_labels is not of shape n'

    print('Storing val features to: '+val_features_path+' and '+val_targets_path)

    # dim nxC
    torch.save(val_features, val_features_path)
    # dim n
    torch.save(val_labels, val_targets_path)

else:
    print('Loading val features from: '+val_features_path+' and '+val_targets_path)

    # dim nxC
    val_features = torch.load(val_features_path)
    # dim n
    val_labels = torch.load(val_targets_path)

    assert val_features.shape[0]==len(val_data) and val_features.shape[1]==feat_dims[model_name], 'val_features is not of shape nxC'
    assert val_labels.shape[0]==len(val_data), 'val_labels is not of shape n'


# ------------------------------------------saving test features------------------------------------------
print('start saving test image features')

if not load_test:
    test_features = []
    test_labels = []
    with torch.no_grad():
        for i, (image, target) in enumerate(tqdm(test_loader)):

            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image = image.cuda()
            target = target.cuda()

            # encode image
            image_input = image.to(device).view(-1, c, h, w)
            image_features = model.encode_image(image_input).view(b, t, -1)
            image_features = fusion_model(image_features)
            # L2 norm image embedding
            image_features /= image_features.norm(dim=-1, keepdim=True)

            test_features.append(image_features)
            test_labels.append(target)
    test_features = torch.cat(test_features)
    test_labels = torch.cat(test_labels)

    assert test_features.shape[0]==len(test_data) and test_features.shape[1]==feat_dims[model_name], 'test_features is not of shape nxC'
    assert test_labels.shape[0]==len(test_data), 'test_labels is not of shape n'

    print('Storing test features to: '+test_features_path+' and '+test_targets_path)

    # dim nxC
    torch.save(test_features, test_features_path)
    # dim n
    torch.save(test_labels, test_targets_path)

else:
    print('Loading test features from: '+test_features_path+' and '+test_targets_path)

    # dim nxC
    test_features = torch.load(test_features_path)
    # dim n
    test_labels = torch.load(test_targets_path)

    assert test_features.shape[0]==len(test_data) and test_features.shape[1]==feat_dims[model_name], 'test_features is not of shape nxC'
    assert test_labels.shape[0]==len(test_data), 'test_labels is not of shape n'

# ------------------------------------------saving few-shot support features------------------------------------------
print('start saving few-shot image features')

if not load_train:

    train_images_targets = []
    train_images_features_agg = []

    # take average of features over multiple augmentations for a more robust feature set
    # similar to averaging done in: https://github.com/gaopengcuhk/Tip-Adapter/blob/fcb06059457a3b74e44ddb0d5c96d2ea7e4c5957/utils.py#L46
    with torch.no_grad():
        for augment_idx in range(augment_epoch):
            train_images_features = []

            print('Augment time: {:} / {:}'.format(augment_idx, augment_epoch))
            for i, (image, target) in enumerate(tqdm(train_loader)):
                image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
                b, t, c, h, w = image.size()
                image = image.cuda()
                target = target.cuda()

                # encode image
                image_input = image.to(device).view(-1, c, h, w)
                image_features = model.encode_image(image_input).view(b, t, -1)
                image_features = fusion_model(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                train_images_features.append(image_features)

                if augment_idx == 0:
                    target = target.cuda()
                    train_images_targets.append(target)

            images_features_cat = torch.cat(train_images_features, dim=0).unsqueeze(0)
            train_images_features_agg.append(images_features_cat)

    # concatenate and take mean of features from multiple augment runs
    train_images_features_agg = torch.cat(train_images_features_agg, dim=0).mean(dim=0)
    # L2 normalise image embeddings from few shot dataset -- dim NKxC
    train_images_features_agg /= train_images_features_agg.norm(dim=-1, keepdim=True)
    # dim CxNK
    train_images_features_agg = train_images_features_agg.permute(1, 0)

    # convert all image labels to one hot labels -- dim NKxN
    train_images_targets = F.one_hot(torch.cat(train_images_targets, dim=0)).half()

    assert train_images_features_agg.shape[0]==feat_dims[model_name] and train_images_features_agg.shape[1]==k_shot*config.data.num_classes, 'train_images_features_agg is not of shape CxNK'
    assert train_images_targets.shape[0]==k_shot*config.data.num_classes and train_images_targets.shape[1]==config.data.num_classes, 'train_images_targets is not of shape NKxN'

    print('Storing train features to: '+train_features_path+' and '+train_targets_path)
    # dim CxNK
    torch.save(train_images_features_agg, train_features_path)
    # dim NKxN
    torch.save(train_images_targets, train_targets_path)

else:
    print('Loading train features from: '+train_features_path+' and '+train_targets_path)
    # dim CxNK
    train_images_features_agg = torch.load(train_features_path)
    # dim NKxN
    train_images_targets = torch.load(train_targets_path)

    assert train_images_features_agg.shape[0]==feat_dims[model_name] and train_images_features_agg.shape[1]==k_shot*config.data.num_classes, 'train_images_features_agg is not of shape CxNK'
    assert train_images_targets.shape[0]==k_shot*config.data.num_classes and train_images_targets.shape[1]==config.data.num_classes, 'train_images_targets is not of shape NKxN'
