# Script to generate text classifier weights using the target category names

import os
import clip
import random
import torch
import argparse
import torch.nn as nn
import yaml
from dotmap import DotMap
from utils.Augmentation import get_augmentation
from utils.Text_Prompt import *
from datasets import Action_DATASETS

random.seed(1)
torch.manual_seed(1)

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


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-cfg', default='configs/HRI10/HRI10_fewshot.yaml')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

config = DotMap(config)

# dummy parameters for dataloader

feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}

device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                               T=config.data.num_segments, dropout=config.network.drop_out,
                                               emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

transform_val = get_augmentation(False, config)

model_text = TextCLIP(model)
model_image = ImageCLIP(model)

model_text = torch.nn.DataParallel(model_text).cuda()
model_image = torch.nn.DataParallel(model_image).cuda()

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
        del checkpoint
    else:
        print(("=> no checkpoint found at '{}'".format(config.pretrain)))

val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                    image_tmpl=config.data.image_tmpl,
                    transform=transform_val, random_shift=config.random_shift)

classes, num_text_aug, text_dict = text_prompt(val_data)

input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

dataset = config.data.dataset
model_name = config.network.arch

disp_name = model_name
if('/' in model_name):
    disp_name = model_name.replace('/', '')

# load few shot dataset

print('Current dataset {}, model_name {}'.format(dataset, model_name))

wp = './features/{}_zeroshot_text_weights_m{}.pt'

if(os.path.exists(wp.format(dataset, disp_name))):
    load_text = True
else:
    load_text = False

if not load_text:
    model.eval()
    text_inputs = classes.to(device)
    text_features = model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    zeroshot_weights = text_features

    print('Storing zeroshot weights to: '+wp.format(dataset, disp_name))
    torch.save(zeroshot_weights, wp.format(dataset, disp_name))
else:
    print('Reading zeroshot weights from: '+wp.format(dataset, disp_name))
    zeroshot_weights = torch.load(wp.format(dataset, disp_name))

print(zeroshot_weights.shape)
assert zeroshot_weights.shape[1]==feat_dims[config.network.arch] and zeroshot_weights.shape[0]==config.data.num_classes * 3, 'zeroshot_weights are not of dim CxN'
