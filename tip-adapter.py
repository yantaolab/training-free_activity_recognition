import os
import torch
import clip
import torch.nn as nn
import random
from tqdm import tqdm
import argparse
from modules.Visual_Prompt import visual_prompt
from utils.Augmentation import *
from dotmap import DotMap
import yaml
from utils.Text_Prompt import *
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

class AdapterMLP(nn.Module):
    def __init__(
        self, in_dim, out_dim,
        hidden=1024, depth=2,
        act='gelu', norm='ln',
        p=0.0, residual=False,
        dtype=None, device=None
    ):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, depth - 1)):     # 前 depth-1 层：in→hidden→…→hidden
            layers.append(nn.Linear(d, hidden, bias=False))
            if norm == 'ln':
                layers.append(nn.LayerNorm(hidden))
            elif norm == 'bn':
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True))
            if p > 0:
                layers.append(nn.Dropout(p))
            d = hidden
        # 最后一层：hidden→out
        layers.append(nn.Linear(d, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

        # 残差只在 in_dim==out_dim 时启用
        self.use_residual = residual and in_dim == out_dim

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        if dtype is not None or device is not None:
            self.to(device=device, dtype=dtype)

    def forward(self, x):
        y = self.net(x)
        if self.use_residual:
            y = y + x
        return y


def search_hp(cache_keys, cache_values, test_features, test_labels, clip_weights, test_video_data_num, num_text_aug, adapter=None):

    search_scale = [50, 50]
    search_step = [200, 20]

    beta_list = [i * (search_scale[0] - 0.1) / search_step[0] + 0.1 for i in
                 range(search_step[0])]
    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in
                  range(search_step[1])]

    best_acc = 0
    best_beta, best_alpha = 0, 0

    for beta in beta_list:
        for alpha in alpha_list:
            if adapter:
                affinity = adapter(test_features)
            else:
                affinity = test_features @ cache_keys

            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * test_features @ clip_weights.T
            tip_logits = clip_logits + cache_logits * alpha

            tip_logits = tip_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
            tip_logits = tip_logits.mean(dim=1, keepdim=False)

            tip_acc1 = cls_acc(tip_logits, test_labels)

            if tip_acc1 > best_acc:
                print("New best setting, beta: {:.2f}, alpha: {:.2f}; accuracy: {:.2f}".format(beta, alpha, tip_acc1))
                best_acc = tip_acc1
                best_beta = beta
                best_alpha = alpha

    print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha

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

def run_tip_adapter(cache_keys, cache_values, test_features, test_labels, clip_weights, test_video_data_num, num_text_aug):

    clip_logits = 100. * test_features @ clip_weights.T
    # Tip-Adapter
    beta, alpha = 1, 1.17
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha

    tip_logits = tip_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
    tip_logits = tip_logits.mean(dim=1, keepdim=False)

    tip_acc1 = cls_acc(tip_logits, test_labels)

    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(tip_acc1))

    # Search Hyperparameters
    _ = search_hp(cache_keys, cache_values, test_features, test_labels, clip_weights, test_video_data_num, num_text_aug, adapter=None)


def copy_first_group_to_all_t(a: torch.Tensor, n_groups: int, allow_pad: bool = False):
    L = a.size(-1)
    if n_groups <= 0:
        raise ValueError("n_groups 必须为正整数")
    if L % n_groups != 0 and not allow_pad:
        raise ValueError(f"列数 {L} 不能被 n_groups={n_groups} 整除；可设 allow_pad=True")
    pad = (((L + n_groups - 1)//n_groups)*n_groups - L) if allow_pad else 0
    if pad:
        a = F.pad(a, (0, pad))

    group_len = a.size(-1) // n_groups
    x = a.reshape(*a.shape[:-1], n_groups, group_len)   # (..., G, group_len)
    first = x[..., 0, :]                                # 第1组
    x[...] = first.unsqueeze(-2)                        # 复制到所有组
    out = x.reshape_as(a)
    return out[..., :L] if pad else out


def run_tip_adapter_with_neg(
    cache_keys,            # [d, N_pos]
    cache_values,          # [N_pos, C] one-hot
    test_features,         # [B, d]
    test_labels,           # [B]
    clip_weights,          # [C, d]
    test_video_data_num: int,
    num_text_aug: int,
    # 下面是新增/可调参数
    beta: float = 1.0,                 # 正缓存温度（与原实现一致）
    alpha: float = 1.17,               # 正缓存融合系数（与原实现一致）
    use_neg: bool = True,              # 开关：是否启用负缓存
    gate_gamma=0.1,
    gate_temp=1.0,
    mu: float = 100.0,                  # 负缓存强度（每个负键行= -mu·e_c）
    k_neg: int = 5,                 # 负缓存 Top-K；None 表示不用 Top-K
    lambda_neg: float = 1.0            # 负缓存整体权重（建议 0.5~1.0）
):
    """
    约定与原版一致：
    - cache_keys 维度 [d, N]，因此亲和度 affinity = [B, N]
    - cache_values 维度 [N, C]，one-hot
    """

    # --- 原始 CLIP logits ---
    clip_logits = 100. * (test_features @ clip_weights.T)   # [B, C]

    # --- Tip-Adapter 正缓存 ---
    affinity = test_features @ cache_keys                   # [B, N_pos]
    # 与你原实现等价：exp(beta*(affinity-1)) @ cache_values
    cache_logits_pos = ((-1.0) * (beta - beta * affinity)).exp() @ cache_values  # [B, C]

    pre_logits = clip_logits + alpha * cache_logits_pos  # [B, C]
    p_pre = F.softmax(pre_logits / gate_temp, dim=1)  # [B, C]

    tip_logits = pre_logits.clone()

    # --- 负缓存：来自“其他类别”的支持集样本（One-vs-Rest）---
    if use_neg:
        beta_n = beta

        # 负值矩阵：每个键属于其 owner 类，对应列为 -mu，其它为 0
        owner = cache_values.argmax(dim=1)  # [N]
        V_neg = torch.zeros_like(cache_values)
        V_neg[torch.arange(cache_values.size(0)), owner] = mu  # [N, C]
        V_neg = copy_first_group_to_all_t(V_neg, num_text_aug)

        # 负键相似度（与正键相同）
        A_neg = affinity.clone()  # [B, N]

        # 可选：Top-K 近邻后再加权，去掉长尾噪声
        if k_neg is not None and 0 < k_neg <= A_neg.size(1):
            vals, idx = torch.topk(A_neg, k=k_neg, dim=1)
            W_local = ((-1.0) * (beta_n - beta_n * vals)).exp()  # [B, k]
            W_neg = torch.zeros_like(A_neg).scatter_(1, idx, W_local)  # [B, N]
        else:
            W_neg = ((-1.0) * (beta_n - beta_n * A_neg)).exp()

        # -------- 关键：软门控 --------
        # 为每个样本 b、每个键 n（其 owner=owner[n]），取该样本对 owner 类的前向概率 p_pre[b, owner[n]]
        # B, N = A_neg.size(0), A_neg.size(1)
        # p_owner = p_pre.gather(1, owner.unsqueeze(0).expand(B, N))  # [B, N]
        # gate = (1.0 - p_owner).clamp(min=0.0).pow(gate_gamma)  # [B, N]
        # W_neg = W_neg * gate  # 抑制自类负压

        cache_logits_neg =(W_neg @ V_neg) * lambda_neg  # [B, C]
        tip_logits = tip_logits + 100 * cache_logits_neg




    # --- 与原版一致的后处理：文本增强聚合 ---
    tip_logits = tip_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
    tip_logits = tip_logits.mean(dim=1, keepdim=False)

    tip_acc1 = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter(+neg) test accuracy: {:.2f}. ****\n".format(tip_acc1))

    return tip_logits, tip_acc1


def run_tip_adapter_F(cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, fusion_model, train_loader_F, test_video_data_num, num_text_aug, num_segments, batch_size):

    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=0.001, eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50 * len(train_loader_F))


    beta, alpha = 1, 1.17
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(50):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, 50))

        for i, (image, target) in enumerate(tqdm(train_loader_F)):
            image = image.view((-1, num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():
                # encode image
                image_input = image.to('cuda').view(-1, c, h, w)
                image_features = clip_model.encode_image(image_input).view(b, t, -1)
                image_features = fusion_model(image_features)
                image_features /= image_features.norm(dim=-1, keepdim=True)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features @ clip_weights.T
            tip_logits = clip_logits + cache_logits * alpha

            tip_logits = tip_logits.view(batch_size, num_text_aug, -1).softmax(dim=-1)
            tip_logits = tip_logits.mean(dim=1, keepdim=False)

            loss = F.cross_entropy(tip_logits, target)

            tip_acc1 = cls_acc(tip_logits, target)

            correct_samples += tip_acc1 / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        tip_top1, tip_top5 = 0., 0.
        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features @ clip_weights.T
        tip_logits = clip_logits + cache_logits * alpha

        tip_logits = tip_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
        tip_logits = tip_logits.mean(dim=1, keepdim=False)

        tip_acc1 = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(tip_acc1))
        if tip_acc1 > best_acc:
            best_acc = tip_acc1
            best_epoch = train_idx
            torch.save(adapter.weight, 'features' + "/best_F_" + str(1) + "shots.pt")
    
    adapter.weight = torch.load('features' + "/best_F_" + str(1) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(affinity, cache_values, test_features, test_labels, clip_weights, test_video_data_num, num_text_aug, adapter=adapter)


def main():

    # Load config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='configs/HRI10/HRI10_fewshot.yaml')
    parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    # dummy parameters for dataloader
    k_shot = config.data.k_shot

    feat_dims = {'RN50': 1024, 'ViT-B/16': 512, 'RN50x16': 768, 'RN101': 512, 'ViT-L/14': 768, 'ViT-B/32': 512}
    features_path = "./features"

    model_name = config.network.arch

    print('Current model: {}'.format(model_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout)  # Must set jit=False for training  ViT-B/32
    model.eval()

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    input_resolution = model.visual.input_resolution
    context_length = model.context_length
    vocab_size = model.vocab_size

    dataset = config.data.dataset
    model_name = config.network.arch
    num_segments = config.data.num_segments
    batch_size = config.data.batch_size

    disp_name = model_name
    if('/' in model_name):
        disp_name = model_name.replace('/', '')

    val_features_path = features_path+"/{}_f_val_m{}.pt".format(dataset, disp_name)
    val_targets_path = features_path+"/{}_t_val_m{}.pt".format(dataset, disp_name)

    test_features_path = features_path+"/{}_f_test_m{}.pt".format(dataset, disp_name)
    test_targets_path = features_path+"/{}_t_test_m{}.pt".format(dataset, disp_name)

    support_features_path = os.path.join(features_path+"/{}_f_train_m{}_k{}.pt".format(dataset, disp_name, k_shot))
    support_labels_path = os.path.join(features_path+"/{}_t_train_m{}_k{}.pt".format(dataset, disp_name, k_shot))

    text_classifier_weights_path = os.path.join(features_path, "{}_zeroshot_text_weights_m{}.pt".format(dataset, disp_name))

    # dim nxC
    val_features = torch.load(val_features_path)
    # dim n
    val_labels = torch.load(val_targets_path)

    # dim nxC
    test_features = torch.load(test_features_path)
    # dim n
    test_labels = torch.load(test_targets_path)

    # dim nxC
    support_features = torch.load(support_features_path)
    # dim n
    support_labels = torch.load(support_labels_path)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                               image_tmpl=config.data.image_tmpl,
                               transform=transform_val, random_shift=config.random_shift)

    # set label as 16 times, because the text is 16 sentences
    classes, num_text_aug, text_dict = text_prompt(val_data)
    support_labels = support_labels.repeat(1, num_text_aug)

    test_video_data_num = np.array(val_data.video_list).shape[0]

    text_classifier_weights = torch.load(text_classifier_weights_path)

    train_data = Action_DATASETS(config.data.train_list, config.data.label_list,
                                 num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                                 random_shift=config.data.random_shift,
                                 transform=transform_train)
    train_loader_F = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)

    # Mydata dataset
    random.seed(1)
    torch.manual_seed(1)


    import time
    total_time = 0.0
    t0 = time.perf_counter()

    # ------------------------------------------ Tip-Adapter ------------------------------------------
    run_tip_adapter(support_features, support_labels, test_features, test_labels, text_classifier_weights, test_video_data_num, num_text_aug)
    # run_tip_adapter_with_neg(support_features, support_labels, test_features, test_labels, text_classifier_weights,
    #                 test_video_data_num, num_text_aug)

    # ------------------------------------------ Tip-Adapter-F ------------------------------------------
    # run_tip_adapter_F(support_features, support_labels, test_features, test_labels, text_classifier_weights, model,
    #                   fusion_model, train_loader_F, test_video_data_num, num_text_aug, num_segments, batch_size)

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


if __name__ == '__main__':
    main()
