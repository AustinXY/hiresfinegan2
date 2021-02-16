import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from prepare_data import IMIM
from torchsummary import summary


os.environ["CUDA_VISIBLE_DEVICES"]="1"

try:
    import wandb

except ImportError:
    wandb = None

from model import Generator, Discriminator, D_NET_BG
from dataset import MultiResolutionDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from non_leaking import augment, AdaptiveAugment

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif classname == 'MnetConv':
        nn.init.constant_(m.mask_conv.weight.data, 1)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )

    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def sample_c_code(batch, c_categories, device):
    return torch.empty(batch, 0).to(device)
    c_code = torch.zeros(batch, c_categories).to(device)
    cid = np.random.randint(c_categories, size=batch)
    for i in range(batch):
        c_code[i, cid[i]] = 1
    return c_code

def binarization_loss(mask):
    return torch.min(1-mask, mask).mean()

def train(args, loader, generator, netsD, g_optim, rf_opt, info_opt, g_ema, device):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module0 = netsD[0].module
        d_module1 = netsD[1].module
        d_module2 = netsD[2].module

    else:
        g_module = generator
        d_module0 = netsD[0]
        d_module1 = netsD[1]
        d_module2 = netsD[2]

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 256, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    criterion_class = nn.CrossEntropyLoss()
    criterion = nn.BCELoss(reduction='none')
    criterion_one = nn.BCELoss()

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, mask = next(loader)
        real_img = real_img.to(device)
        mask = mask.to(device)

        ############# train bg discriminator #############
        requires_grad(generator, True)
        requires_grad(netsD[0], True)
        requires_grad(netsD[1], False)
        requires_grad(netsD[2], False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        c_code = sample_c_code(args.batch, args.c_categories, device)

        # summary(generator, [[512], [200]])
        # sys.exit()

        image_li, _, _ = generator(noise, c_code)
        # raw_images = image_li[1]
        # fake_bg = raw_images[0]

        # real_logits = netsD[0](real_img, mask=mask)

        # rf_scores, _, fnl_masks = real_logits
        # weights_real = torch.ones_like(rf_scores)

        # invalid_patch = fnl_masks != 0.0
        # weights_real.masked_fill_(invalid_patch, 0.0)

        # real_labels = torch.ones_like(rf_scores)

        # errD_real_uncond = criterion(real_logits[0], real_labels)  # Real/Fake loss for 'real background' (on patch level)
        # errD_real_uncond = torch.mul(errD_real_uncond, weights_real)  # Masking output units which correspond to receptive fields which lie within the boundin box
        # errD_real_uncond = errD_real_uncond.mean()

        # errD_real_uncond_classi = criterion(real_logits[1], weights_real)  # Background/foreground classification loss
        # errD_real_uncond_classi = errD_real_uncond_classi.mean()

        # fake_logits = netsD[0](fake_bg, mask=None)

        # fake_labels = torch.zeros_like(fake_logits[0])

        # errD_fake_uncond = criterion(fake_logits[0], fake_labels)  # Real/Fake loss for 'fake background' (on patch level)
        # errD_fake_uncond = errD_fake_uncond.mean()

        # norm_fact_real = weights_real.sum()
        # norm_fact_fake = weights_real.shape[0]*weights_real.shape[1]*weights_real.shape[2]*weights_real.shape[3]

        # if norm_fact_real > 0:    # Normalizing the real/fake loss for background after accounting the number of masked members in the output.
        #     errD_real = errD_real_uncond * ((norm_fact_fake * 1.0) /(norm_fact_real * 1.0))
        # else:
        #     errD_real = errD_real_uncond

        # errD_fake = errD_fake_uncond
        # errD = ((errD_real + errD_fake) * args.bg_wt) + errD_real_uncond_classi

        # loss_dict["d0"] = errD

        # netsD[0].zero_grad()
        # errD.backward()
        # rf_opt[0].step()


        # ############# train child discriminator #############
        # requires_grad(generator, False)
        # requires_grad(netsD[0], False)
        # requires_grad(netsD[1], False)
        # requires_grad(netsD[2], True)

        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # c_code = sample_c_code(args.batch, args.c_categories, device)
        # image_li, _, _ = generator(noise, c_code)
        # fake_img = image_li[0]

        # if args.augment:
        #     real_img_aug, _ = augment(real_img, ada_aug_p)
        #     fake_img, _ = augment(fake_img, ada_aug_p)
        # else:
        #     real_img_aug = real_img

        # fake_pred = netsD[2](fake_img)[0]
        # real_pred = netsD[2](real_img_aug)[0]

        # d_loss = d_logistic_loss(real_pred, fake_pred)

        # loss_dict["d"] = d_loss
        # loss_dict["real_score"] = real_pred.mean()
        # loss_dict["fake_score"] = fake_pred.mean()

        # netsD[2].zero_grad()
        # d_loss.backward()
        # rf_opt[2].step()

        # if args.augment and args.augment_p == 0:
        #     ada_aug_p = ada_augment.tune(real_pred)
        #     r_t_stat = ada_augment.r_t_stat

        # d_regularize = i % args.d_reg_every == 0

        # if d_regularize:
        #     real_img.requires_grad = True
        #     real_pred = netsD[2](real_img)[0]
        #     r1_loss = d_r1_loss(real_pred, real_img)

        #     netsD[2].zero_grad()
        #     (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

        #     rf_opt[2].step()

        # loss_dict["r1"] = r1_loss


        # ############# train generator #############
        # requires_grad(generator, True)
        # requires_grad(netsD[0], False)
        # requires_grad(netsD[1], True)
        # requires_grad(netsD[2], True)

        # noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        # c_code = sample_c_code(args.batch, args.c_categories, device)
        # image_li, code_li, _ = generator(noise, c_code)
        # fake_img = image_li[0]
        # mkd_images = image_li[2]
        # masks = image_li[3]
        # p_code = code_li[1]
        # c_code = code_li[2]

        # if args.augment:
        #     fake_img, _ = augment(fake_img, ada_aug_p)

        # # background rf
        # output = netsD[0](fake_img)
        # real_labels = torch.ones_like(output[0])
        # g_bg_loss = criterion_one(output[0], real_labels) * args.bg_wt
        # errG_classi = criterion_one(output[1], real_labels) # Background/Foreground classification loss for the fake background image (on patch level)
        # g_bg_loss += errG_classi

        # loss_dict["g_bg"] = g_bg_loss

        # # child rf
        # fake_pred = netsD[2](fake_img)[0]
        # g_loss = g_nonsaturating_loss(fake_pred)

        # loss_dict["g"] = g_loss

        # # parent, child info
        # pred_p = netsD[1](mkd_images[1])[1]
        # p_info_loss = criterion_class(pred_p, torch.nonzero(p_code.long(), as_tuple=False)[:,1])

        # pred_c = netsD[2](mkd_images[2])[1]
        # c_info_loss = criterion_class(pred_c, torch.nonzero(c_code.long(), as_tuple=False)[:,1])

        # loss_dict["p_info"] = p_info_loss
        # loss_dict["c_info"] = c_info_loss

        # binary_loss = binarization_loss(masks[1]) * 0
        # # oob_loss = torch.sum(bg_mk * ch_mk, dim=(-1,-2)).mean() * 1e-2
        # ms = masks[1].size()
        # min_fg_cvg = 0.2 * ms[2] * ms[3]
        # fg_cvg_loss = F.relu(min_fg_cvg - torch.sum(masks[1], dim=(-1,-2))).mean() * 0

        # ms = masks[1].size()
        # min_bg_cvg = 0.2 * ms[2] * ms[3]
        # bg_cvg_loss = F.relu(min_bg_cvg - torch.sum(torch.ones_like(masks[1])-masks[1], dim=(-1,-2))).mean() * 0

        # loss_dict["bin"] = binary_loss
        # loss_dict["cvg"] = fg_cvg_loss + bg_cvg_loss

        # generator_loss = g_loss + g_bg_loss + p_info_loss + c_info_loss #+ binary_loss + fg_cvg_loss + bg_cvg_loss

        # generator.zero_grad()
        # # netsD[0].zero_grad()
        # netsD[1].zero_grad()
        # netsD[2].zero_grad()
        # generator_loss.backward()
        # g_optim.step()
        # info_opt[1].step()
        # info_opt[2].step()

        # g_regularize = i % args.g_reg_every == 0

        # if g_regularize:
        #     path_batch_size = max(1, args.batch // args.path_batch_shrink)
        #     noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
        #     c_code = sample_c_code(path_batch_size, args.c_categories, device)
        #     image_li, _, latents = generator(noise, c_code, return_latents=True)
        #     fake_img = image_li[0]

        #     path_loss, mean_path_length, path_lengths = g_path_regularize(
        #         fake_img, latents, mean_path_length
        #     )

        #     generator.zero_grad()
        #     weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

        #     if args.path_batch_shrink:
        #         weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        #     weighted_path_loss.backward()

        #     g_optim.step()

        #     mean_path_length_avg = (
        #         reduce_sum(mean_path_length).item() / get_world_size()
        #     )

        # loss_dict["path"] = path_loss
        # loss_dict["path_length"] = path_lengths.mean()

        # accumulate(g_ema, g_module, accum)

        # loss_reduced = reduce_loss_dict(loss_dict)

        # d_loss_val = loss_reduced["d"].mean().item()
        # d0_loss_val = loss_reduced["d0"].mean().item()
        # g_loss_val = loss_reduced["g"].mean().item()
        # g_bg_loss_val = loss_reduced["g_bg"].mean().item()
        # p_info_loss_val = loss_reduced["p_info"].mean().item()
        # c_info_loss_val = loss_reduced["c_info"].mean().item()
        # binary_loss_val = loss_reduced["bin"].mean().item()
        # cvg_loss_val = loss_reduced["cvg"].mean().item()
        # r1_val = loss_reduced["r1"].mean().item()
        # path_loss_val = loss_reduced["path"].mean().item()
        # real_score_val = loss_reduced["real_score"].mean().item()
        # fake_score_val = loss_reduced["fake_score"].mean().item()
        # path_length_val = loss_reduced["path_length"].mean().item()

        # if get_rank() == 0:
        #     pbar.set_description(
        #         (
        #             f"d: {d_loss_val:.4f}; d0: {d0_loss_val:.4f}; g: {g_loss_val:.4f}; g_bg: {g_bg_loss_val:.4f}; "
        #             f"p_info: {p_info_loss_val:.4f}; c_info: {c_info_loss_val:.4f}; "
        #             f"bin: {binary_loss_val:.4f}; cvg: {cvg_loss_val:.4f}; "
        #             f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f};  r1: {r1_val:.4f};"
        #             f"augment: {ada_aug_p:.4f}"
        #         )
        #     )

        #     if wandb and args.wandb:
        #         wandb.log(
        #             {
        #                 "Generator": g_loss_val,
        #                 "Discriminator": d_loss_val,
        #                 "BG_Generator": g_bg_loss_val,
        #                 "BG_Discriminator": d0_loss_val,
        #                 "p_info": p_info_loss_val,
        #                 "c_info": c_info_loss_val,
        #                 "Augment": ada_aug_p,
        #                 "Rt": r_t_stat,
        #                 "R1": r1_val,
        #                 "Path Length Regularization": path_loss_val,
        #                 "Mean Path Length": mean_path_length,
        #                 "Real Score": real_score_val,
        #                 "Fake Score": fake_score_val,
        #                 "Path Length": path_length_val,
        #             }
        #         )

        #     if i % 1000 == 0:
        #         with torch.no_grad():
        #             g_ema.eval()
        #             c_code = sample_c_code(args.n_sample, args.c_categories, device)
        #             image_li, _, _ = g_ema([sample_z], c_code)

        #             utils.save_image(
        #                 image_li[0],
        #                 f"sample/{str(i).zfill(6)}_0.png",
        #                 nrow=int(args.n_sample ** 0.5),
        #                 normalize=True,
        #                 range=(-1, 1),
        #             )

        #             for j in range(3):
        #                 utils.save_image(
        #                     image_li[1][j],
        #                     f"sample/{str(i).zfill(6)}_{str(1+j)}.png",
        #                     nrow=int(args.n_sample ** 0.5),
        #                     normalize=True,
        #                     range=(-1, 1),
        #                 )

        #             for j in range(3):
        #                 utils.save_image(
        #                     image_li[2][j],
        #                     f"sample/{str(i).zfill(6)}_{str(4+j)}.png",
        #                     nrow=int(args.n_sample ** 0.5),
        #                     normalize=True,
        #                     range=(-1, 1),
        #                 )

        #             for j in range(2):
        #                 utils.save_image(
        #                     image_li[3][j],
        #                     f"sample/{str(i).zfill(6)}_{str(7+j)}.png",
        #                     nrow=int(args.n_sample ** 0.5),
        #                     normalize=True,
        #                     range=(0, 1),
        #                 )

        #     if i % 10000 == 0:
        #         torch.save(
        #             {
        #                 "g": g_module.state_dict(),
        #                 "d0": d_module0.state_dict(),
        #                 "d1": d_module1.state_dict(),
        #                 "d2": d_module2.state_dict(),
        #                 "g_ema": g_ema.state_dict(),
        #                 "g_optim": g_optim.state_dict(),
        #                 "rf_optim2": rf_opt[2].state_dict(),
        #                 "info_optim1": info_opt[1].state_dict(),
        #                 "info_optim2": info_opt[2].state_dict(),
        #                 "args": args,
        #                 "ada_aug_p": ada_aug_p,
        #             },
        #             f"checkpoint/{str(i).zfill(6)}.pt",
        #         )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("path", type=str, help="path to the lmdb dataset")
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument(
        "--n_sample",
        type=int,
        default=8,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=128, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )
    parser.add_argument(
        "--b_categories",
        type=int,
        default=0,
        help="number of background categories",
    )
    parser.add_argument(
        "--p_categories",
        type=int,
        default=0,
        help="number of parent categories",
    )
    parser.add_argument(
        "--c_categories",
        type=int,
        default=0,
        help="number of child categories",
    )
    parser.add_argument(
        "--p_size",
        type=int,
        default=64,
        help="parent output size",
    )
    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8
    args.bg_wt = 1e-1

    args.start_iter = 0

    if args.n_sample == 0:
        args.n_sample = args.batch

    generator = Generator(
        args.size, args.latent, args.n_mlp, args.p_categories, args.c_categories,
        args.p_size, channel_multiplier=args.channel_multiplier
    ).to(device)
    # for name, param in generator.named_parameters():
    #     print (name, param.size())
    # sys.exit()
    g_ema = Generator(
        args.size, args.latent, args.n_mlp, args.p_categories, args.c_categories,
        args.p_size, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    depth = int(math.log(args.size, 2))-5
    netD0 = D_NET_BG(depth).to(device)
    netD0.apply(weights_init)

    netD1 = Discriminator(
        args.p_size, args.p_categories, channel_multiplier=args.channel_multiplier
    ).to(device)
    netD2 = Discriminator(
        args.size, args.c_categories, channel_multiplier=args.channel_multiplier
    ).to(device)
    netsD = [netD0, netD1, netD2]

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    d_optim0 = optim.Adam(
        netsD[0].parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_optim2 = optim.Adam(
        netsD[2].parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    rf_opt = [d_optim0, None, d_optim2]

    info_optim1 = optim.Adam(
        netsD[1].parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    info_optim2 = optim.Adam(
        netsD[2].pred_linear.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )
    info_opt = [None, info_optim1, info_optim2]

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
        except ValueError:
            pass

        generator.load_state_dict(ckpt["g"])
        netsD[0].load_state_dict(ckpt["d0"])
        netsD[1].load_state_dict(ckpt["d1"])
        netsD[2].load_state_dict(ckpt["d2"])
        g_ema.load_state_dict(ckpt["g_ema"])

        g_optim.load_state_dict(ckpt["g_optim"])
        rf_opt[0].load_state_dict(ckpt["d_optim0"])
        rf_opt[2].load_state_dict(ckpt["rf_optim"])

        info_opt[1].load_state_dict(ckpt["info_optim1"])
        info_opt[2].load_state_dict(ckpt["info_optim2"])

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        netsD[0] = nn.parallel.DistributedDataParallel(
            netsD[0],
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        netsD[1] = nn.parallel.DistributedDataParallel(
            netsD[1],
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        netsD[2] = nn.parallel.DistributedDataParallel(
            netsD[2],
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(args.path, transform, args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )


    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project="stylegan2 bg patchgan")

    train(args, loader, generator, netsD, g_optim, rf_opt, info_opt, g_ema, device)
