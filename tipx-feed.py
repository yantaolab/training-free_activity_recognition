# Main implementation of the TIP-X framework
# We use some parts of the TIP-Adapter codebase: https://github.com/gaopengcuhk/Tip-Adapter
# Refer Sec 3.2 of paper

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
from scipy.stats import wasserstein_distance
import torch.nn.functional as F

random.seed(1)
torch.manual_seed(1)


def compute_image_text_distributions(temp, train_images_features_agg, test_features, val_features,
                                     vanilla_zeroshot_weights):
    train_image_class_distribution = train_images_features_agg.T @ vanilla_zeroshot_weights.T
    train_image_class_distribution = nn.Softmax(dim=-1)(train_image_class_distribution / temp)

    test_image_class_distribution = test_features @ vanilla_zeroshot_weights.T
    test_image_class_distribution = nn.Softmax(dim=-1)(test_image_class_distribution / temp)

    val_image_class_distribution = val_features @ vanilla_zeroshot_weights.T
    val_image_class_distribution = nn.Softmax(dim=-1)(val_image_class_distribution / temp)

    return train_image_class_distribution, test_image_class_distribution, val_image_class_distribution


def get_cosine_similarity_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    cosine_sims = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0] // bs)):
        curr_batch = test_image_class_distribution[i * bs: (i + 1) * bs]

        # 计算余弦相似度
        cosine_sim = torch.mm(curr_batch, train_image_class_distribution.T)
        cosine_sims[i * bs: (i + 1) * bs, :] = cosine_sim

    return cosine_sims


def get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    kl_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0] // bs)):
        curr_batch = test_image_class_distribution[i * bs: (i + 1) * bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)
        q = train_image_class_distribution
        q_repeated = torch.cat([q] * bs)
        kl = repeated_batch * (repeated_batch.log() - q_repeated.log())
        kl = kl.sum(dim=-1)
        kl = kl.view(bs, -1)
        kl_divs_sim[i * bs: (i + 1) * bs, :] = kl

    return kl_divs_sim


def get_js_divergence_sims(train_image_class_distribution, test_image_class_distribution):
    bs = 100
    js_divs_sim = torch.zeros((test_image_class_distribution.shape[0], train_image_class_distribution.shape[0]))

    for i in tqdm(range(test_image_class_distribution.shape[0] // bs)):
        curr_batch = test_image_class_distribution[i * bs: (i + 1) * bs]
        repeated_batch = torch.repeat_interleave(curr_batch, train_image_class_distribution.shape[0], dim=0)

        q = train_image_class_distribution
        q_repeated = torch.cat([q] * bs)

        # 计算 M = (P + Q) / 2
        m = 0.5 * (repeated_batch + q_repeated)

        # 计算 KL(P || M) 和 KL(Q || M)
        kl_pm = repeated_batch * (repeated_batch.log() - m.log())
        kl_qm = q_repeated * (q_repeated.log() - m.log())

        js_div = 0.5 * kl_pm.sum(dim=-1) + 0.5 * kl_qm.sum(dim=-1)
        js_div = js_div.view(bs, -1)

        js_divs_sim[i * bs: (i + 1) * bs, :] = js_div

    return js_divs_sim


def get_kl_div_sims(args, test_features, val_features, train_features, clip_weights):
    train_image_class_distribution, test_image_class_distribution, val_image_class_distribution = compute_image_text_distributions(
        args.temperature, train_features, test_features, val_features, clip_weights)

    # train_kl_divs_sim = get_cosine_similarity_sims(train_image_class_distribution, train_image_class_distribution)
    # test_kl_divs_sim = get_cosine_similarity_sims(train_image_class_distribution, test_image_class_distribution)
    # val_kl_divs_sim = get_cosine_similarity_sims(train_image_class_distribution, val_image_class_distribution)

    train_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, train_image_class_distribution)
    test_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, test_image_class_distribution)
    val_kl_divs_sim = get_kl_divergence_sims(train_image_class_distribution, val_image_class_distribution)

    return train_kl_divs_sim, test_kl_divs_sim, val_kl_divs_sim

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def scale_(x, target):
    y = (x - x.min()) / (x.max() - x.min())
    y *= target.max() - target.min()
    y += target.min()

    return y

def _kl_to_affinity(kl_mat, mode="exp", tau_kl=1.0, ref_aff=None):
    """
    把 KL 距离矩阵转成“与缓存键的亲和度”。两种模式：
    - "exp": A = exp(-tau_kl * KL)  （最稳，推荐）
    - "scale": A = -scale_(KL, ref_aff)  （沿用你原来的 scale_ 映射，再取负号）
    """
    if mode == "exp":
        return torch.exp(-tau_kl * kl_mat)
    elif mode == "scale":
        assert ref_aff is not None, "scale 模式需要提供 ref_aff（如 new_knowledge/aff）"
        neg_affs = scale_(kl_mat, ref_aff)  # 你原来的映射
        return -neg_affs
    else:
        raise ValueError("mode must be 'exp' or 'scale'")

def copy_first_group_to_all_t(a: torch.Tensor, n_groups: int, allow_pad: bool = False):
    L = a.size(-1)
    if n_groups <= 0:
        raise ValueError("n_groups 必须为正整数")
    if L % n_groups != 0 and not allow_pad:
        raise ValueError(f"列数 {L} 不能被 n_groups={n_groups} 整除；可设 allow_pad=True")
    pad = (((L + n_groups - 1) // n_groups) * n_groups - L) if allow_pad else 0
    if pad:
        a = F.pad(a, (0, pad))

    group_len = a.size(-1) // n_groups
    x = a.reshape(*a.shape[:-1], n_groups, group_len)  # (..., G, group_len)
    first = x[..., 0, :]  # 第1组
    x[...] = first.unsqueeze(-2)  # 复制到所有组
    out = x.reshape_as(a)
    return out[..., :L] if pad else out

def hparam_search(val_features, val_labels, test_features, test_labels, train_images_features_agg, train_images_targets,
                  zeroshot_weights, val_kl_divs_sim, test_kl_divs_sim, test_video_data_num, num_text_aug, num_classes):
    search_scale = [50, 50, 30]
    search_step = [200, 20, 50]
    # train_images_targets = torch.where(train_images_targets == 1, torch.tensor(0.5, dtype=train_images_targets.dtype).cuda(), train_images_targets)

    alpha_list = [i * (search_scale[1] - 0.1) / search_step[1] + 0.1 for i in range(search_step[1])]
    beta_list = [i * (search_scale[0] - 1) / search_step[0] + 1 for i in range(search_step[0])]
    gamma_list = [i * (search_scale[2] - 0.1) / search_step[2] + 0.1 for i in range(search_step[2])]
    mu_list = [0.5, 1, 10, 100]

    # ---------- 验证集 ----------
    clip_logits_val = 100. * (val_features @ zeroshot_weights.T)  # [Bv, C]
    aff_val = val_features @ train_images_features_agg  # [Bv, N_cache] 供对照/可选
    best_tipx_acc = 0
    gate_gamma = 1.0
    gate_temp = 1.0
    k_neg = None  # Top-K 近邻后再加权，去掉长尾噪声

    for alpha in alpha_list:
        for beta in beta_list:
            n = 0.
            batch_idx = 0

            # 正缓存（TIP-Adapter）
            cache_pos_val = ((-1.0) * (beta - beta * aff_val)).exp() @ train_images_targets  # [Bv, C]

            neg_affs = scale_((val_kl_divs_sim).cuda(), aff_val)
            affinities = -neg_affs
            kl_logits = affinities.half() @ train_images_targets

            batch_idx += 1
            n += val_features.size(0)

            for gamma in gamma_list:
                # for gamma_n in gamma_list:
                for mu in mu_list:
                    tipx_top1, tipx_top5 = 0., 0.

                    # 前向证据分布
                    pre_logits_val = clip_logits_val + alpha * cache_pos_val + kl_logits * gamma
                    p_pre = F.softmax(pre_logits_val / gate_temp, dim=1)  # [B, C]
                    tipx_logits = pre_logits_val.clone()

                    # --- 负缓存：来自“其他类别”的支持集样本（One-vs-Rest）---
                    beta_n = beta
                    gamma_n = alpha

                    # 负值矩阵：每个键属于其 owner 类，对应列为 mu，其它为 0
                    owner = train_images_targets.argmax(dim=1)  # [N]
                    V_neg = torch.zeros_like(train_images_targets)
                    V_neg[torch.arange(train_images_targets.size(0)), owner] = mu  # [N, C]
                    V_neg = copy_first_group_to_all_t(V_neg, num_text_aug)

                    # 负键相似度（与正键相同）
                    A_neg = (aff_val).clone()  # [B, N]

                    # 可选：Top-K 近邻后再加权，去掉长尾噪声
                    if k_neg is not None and 0 < k_neg <= A_neg.size(1):
                        vals, idx = torch.topk(A_neg, k=k_neg, dim=1)
                        W_local = ((-1.0) * (beta_n - beta_n * vals)).exp()  # [B, k]
                        W_neg = torch.zeros_like(A_neg).scatter_(1, idx, W_local)  # [B, N]
                    else:
                        W_neg = ((-1.0) * (beta_n - beta_n * A_neg)).exp()

                    # -------- 关键：软门控 - -------
                    # 为每个样本 b、每个键 n（其 owner=owner[n]），取该样本对 owner 类的前向概率 p_pre[b, owner[n]]
                    B, N = A_neg.size(0), A_neg.size(1)
                    p_pre = p_pre.view(test_video_data_num, num_text_aug, -1).mean(dim=1)
                    p_owner = p_pre.gather(1, owner.unsqueeze(0).expand(B, N))  # [B, N]
                    gate = (1 - p_owner).clamp(min=0.0).pow(gate_gamma)  # [B, N]
                    W_neg = W_neg * gate  # 抑制自类负压

                    cache_logits_neg = (W_neg.half() @ V_neg) * 1.0  # [B, C]
                    tipx_logits = tipx_logits + gamma_n * cache_logits_neg

                    tipx_logits = tipx_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
                    tipx_logits = tipx_logits.mean(dim=1, keepdim=False)

                    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, val_labels, topk=(1, 5))
                    tipx_top1 += tipx_acc1
                    tipx_top5 += tipx_acc5
                    tipx_top1 = (tipx_top1 / n) * 100
                    tipx_top5 = (tipx_top5 / n) * 100

                    if tipx_top1 > best_tipx_acc:
                        best_tipx_acc = tipx_top1
                        best_alpha_tipx = alpha
                        best_gamma_tipx = gamma
                        best_beta_tipx = beta
                        best_gamma_n_tipx = gamma_n
                        best_mu_tipx = mu
                        print(best_tipx_acc, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_gamma_n_tipx, best_mu_tipx)


    n = test_features.size(0)

    clip_logits = 100. * (test_features @ zeroshot_weights.T)

    # 正缓存（TIP-Adapter）
    cache_pos_test = ((-1.0) * (best_beta_tipx - best_beta_tipx * aff_val)).exp() @ train_images_targets  # [Bv, C]

    neg_affs = scale_((test_kl_divs_sim).cuda(), aff_val)
    affinities = -neg_affs
    kl_logits = affinities.half() @ train_images_targets

    # 前向证据分布
    pre_logits_test = clip_logits + best_alpha_tipx * cache_pos_test + kl_logits * best_gamma_tipx
    p_pre = F.softmax(pre_logits_test / gate_temp, dim=1)  # [B, C]
    tipx_logits = pre_logits_test.clone()

    # --- 负缓存：来自“其他类别”的支持集样本（One-vs-Rest）---
    beta_n = best_beta_tipx
    gamma_n = best_gamma_n_tipx

    # 负值矩阵：每个键属于其 owner 类，对应列为 -mu，其它为 0
    owner = train_images_targets.argmax(dim=1)  # [N]
    V_neg = torch.zeros_like(train_images_targets)
    V_neg[torch.arange(train_images_targets.size(0)), owner] = best_mu_tipx  # [N, C]
    V_neg = copy_first_group_to_all_t(V_neg, num_text_aug)

    # 负键相似度（与正键相同）
    A_neg = (aff_val).clone()  # [B, N]

    # 可选：Top-K 近邻后再加权，去掉长尾噪声
    if k_neg is not None and 0 < k_neg <= A_neg.size(1):
        vals, idx = torch.topk(A_neg, k=k_neg, dim=1)
        W_local = ((-1.0) * (beta_n - beta_n * vals)).exp()  # [B, k]
        W_neg = torch.zeros_like(A_neg).scatter_(1, idx, W_local)  # [B, N]
    else:
        W_neg = ((-1.0) * (beta_n - beta_n * A_neg)).exp()

    # -------- 关键：软门控 - -------
    # 为每个样本 b、每个键 n（其 owner=owner[n]），取该样本对 owner 类的前向概率 p_pre[b, owner[n]]
    B, N = A_neg.size(0), A_neg.size(1)
    p_pre = p_pre.view(test_video_data_num, num_text_aug, -1).mean(dim=1)
    p_owner = p_pre.gather(1, owner.unsqueeze(0).expand(B, N))  # [B, N]
    gate = (1 - p_owner).clamp(min=0.0).pow(gate_gamma)  # [B, N]
    W_neg = W_neg * gate  # 抑制自类负压

    cache_logits_neg = (W_neg.half() @ V_neg) * 1.0  # [B, C]
    tipx_logits = tipx_logits + gamma_n * cache_logits_neg

    tipx_logits = tipx_logits.view(test_video_data_num, num_text_aug, -1).softmax(dim=-1)
    tipx_logits = tipx_logits.mean(dim=1, keepdim=False)

    tipx_top1, tipx_top5 = 0., 0.

    tipx_acc1, tipx_acc5 = accuracy(tipx_logits, test_labels, topk=(1, 5))
    tipx_top1 += tipx_acc1
    tipx_top5 += tipx_acc5
    tipx_top1 = (tipx_top1 / n) * 100
    tipx_top5 = (tipx_top5 / n) * 100


    return tipx_top1, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_gamma_n_tipx, best_mu_tipx


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='configs/HRI10/HRI10_fewshot.yaml')
    parser.add_argument('--temperature', type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = DotMap(config)

    # dummy parameters for dataloader
    k_shot = config.data.k_shot

    features_path = "./features"

    model_name = config.network.arch

    print('Current model: {} and k-shot: {}'.format(model_name, k_shot))

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
    num_classes = config.data.num_classes

    disp_name = model_name
    if ('/' in model_name):
        disp_name = model_name.replace('/', '')

    val_features_path = features_path + "/{}_f_val_m{}.pt".format(dataset, disp_name)
    val_targets_path = features_path + "/{}_t_val_m{}.pt".format(dataset, disp_name)

    test_features_path = features_path + "/{}_f_test_m{}.pt".format(dataset, disp_name)
    test_targets_path = features_path + "/{}_t_test_m{}.pt".format(dataset, disp_name)

    support_features_path = os.path.join(features_path + "/{}_f_train_m{}_k{}.pt".format(dataset, disp_name, k_shot))
    support_labels_path = os.path.join(features_path + "/{}_t_train_m{}_k{}.pt".format(dataset, disp_name, k_shot))

    text_classifier_weights_path = os.path.join(features_path,
                                                "{}_zeroshot_text_weights_m{}.pt".format(dataset, disp_name))

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

    import time

    total_time = 0.0
    t0 = time.perf_counter()

    train_kl_divs_sims, test_kl_divs_sims, val_kl_divs_sims = get_kl_div_sims(args, test_features, val_features,
                                                                              support_features, text_classifier_weights)

    tipx_acc, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_gamma_n_tipx, best_mu_tipx = hparam_search(
        val_features, val_labels, test_features, test_labels, support_features, support_labels, text_classifier_weights,
        val_kl_divs_sims, test_kl_divs_sims, test_video_data_num, num_text_aug, num_classes)

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

    print('--------------------------------------------')
    print('Best for Dataset: {}, Model: {}, alpha: {}, beta: {}, gamma: {}, gamma_n: {}, mu: {}, TIP-X Accuracy: {}'
          .format(dataset, model_name, best_alpha_tipx, best_beta_tipx, best_gamma_tipx, best_gamma_n_tipx, best_mu_tipx, tipx_acc))
    print('--------------------------------------------')
    print()
    print('----------------------------------------------------------------------------')
