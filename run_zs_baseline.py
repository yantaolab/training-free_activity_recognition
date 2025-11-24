# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

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
from utils.Text_Prompt import *
import numpy as np

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

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='configs/HRI10/HRI10_fewshot.yaml')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)
    # wandb.init(project=config['network']['type'],
    #            name='{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
    #                                      config['data']['dataset']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                                   T=config.data.num_segments, dropout=config.network.drop_out,
                                                   emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32

    transform_val = get_augmentation(False, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    # wandb.watch(model)
    # wandb.watch(fusion_model)

    features_path = "./features/"
    model_name = config.network.arch

    disp_name = model_name
    if ('/' in model_name):
        disp_name = model_name.replace('/', '')

    test_features_path = features_path + "/{}_f_test_m{}.pt".format(config.data.dataset, disp_name)
    test_targets_path = features_path + "/{}_t_test_m{}.pt".format(config.data.dataset, disp_name)

    # dim nxC
    test_features = torch.load(test_features_path)
    # dim n
    test_labels = torch.load(test_targets_path)

    text_classifier_weights_path = os.path.join(features_path,
                                                "{}_zeroshot_text_weights_m{}.pt".format(config.data.dataset, disp_name))
    text_classifier_weights = torch.load(text_classifier_weights_path)


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

    # set label as 16 times, because the text is 16 sentences
    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                        image_tmpl=config.data.image_tmpl,
                        transform=transform_val, random_shift=config.random_shift)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=True, drop_last=True)

    classes, num_text_aug, text_dict = text_prompt(val_data)
    test_video_data_num = np.array(val_data.video_list).shape[0]

    ''''''
    # num = 0
    # corr_1 = 0
    # model.eval()
    # fusion_model.eval()
    # with torch.no_grad():
    #     for iii, (image, class_id) in enumerate(tqdm(val_loader)):
    #         image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
    #         b, t, c, h, w = image.size()
    #         class_id = class_id.to(device)
    #         image_input = image.to(device).view(-1, c, h, w)
    #         image_features = model.encode_image(image_input).view(b, t, -1)
    #         image_features = fusion_model(image_features)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #
    #         image_features = test_features[[iii], :]
    #
    #         similarity = (100.0 * image_features @ text_classifier_weights.T)
    #         similarity = similarity.view(b, num_text_aug, -1).softmax(dim=-1)
    #         similarity = similarity.mean(dim=1, keepdim=False)
    #         values_1, indices_1 = similarity.topk(1, dim=-1)
    #         num += b
    #         for i in range(b):
    #             if indices_1[i] == class_id[i]:
    #                 corr_1 += 1
    # top1 = float(corr_1) / num * 100
    # print(top1)



    import time

    total_time = 0.0
    t0 = time.perf_counter()

    logits = (100.0 * test_features @ text_classifier_weights.T)
    logits = logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
    logits = logits.mean(dim=1, keepdim=False)

    tipx_acc1, tipx_acc5 = accuracy(logits, test_labels, topk=(1, 2))
    B = test_labels.size(0)
    top1_pct = tipx_acc1 / B * 100.0
    top5_pct = tipx_acc5 / B * 100.0
    print(f"Top1={top1_pct:.2f}%, Top5={top5_pct:.2f}%")
    acc = cls_acc(logits, test_labels)
    print(f"Top1={acc:.2f}%")

    # TODO: 这里放你的代码
    elapsed = time.perf_counter() - t0
    total_time += elapsed
    count = test_video_data_num

    if count > 0:
        avg_time = total_time
        print(f"\n共处理 {count} 张图像")
        print(f"耗时：{avg_time:.4f} s ({avg_time * 1000:.2f} ms)")
    else:
        print("⚠️ 没有找到任何图像")



    # labels = test_labels
    # np_preds = torch.argmax(logits, dim=1).cpu().numpy()
    # np_labels = labels.cpu().numpy()
    # zs_acc = 100 * (np_preds == np_labels).sum() / np_labels.shape[0]
    # print('ZS Acc for Dataset: {}, Model: {} == '.format(config.data.dataset, config.network.arch), zs_acc)


if __name__ == '__main__':
    main()
