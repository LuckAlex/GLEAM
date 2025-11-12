import numpy as np
import torch
import torch.nn as nn

import copy
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from TransferAttack.transferattack.gradient.vmifgsm import VMIFGSM
import scipy.stats as st
import matplotlib.pyplot as plt
from utils import get_kernel
from OpenAttack.attack_assist.substitute.word import WordSubstitute, get_default_substitute
from copy import deepcopy
from scipy.ndimage import filters
from ot_loss import calculate_wasserstein_distance


from copy import deepcopy
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as F


def bspline_basis(u):
    p = torch.zeros(u.shape + (4,), dtype=u.dtype, device=u.device)
    p[..., 0] = (1 - u) ** 3 / 6
    p[..., 1] = (3 * u ** 3 - 6 * u ** 2 + 4) / 6
    p[..., 2] = (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
    p[..., 3] = u ** 3 / 6
    return p


def init_control_points(src_img, row_block_num, col_block_num, min_offset=-10, max_offset=10):
    """
    使用 Torch 初始化控制点和权重。
    :param src_img: 输入图像张量，形状为 (N, C, H, W)
    :param row_block_num: 网格的行数。
    :param col_block_num: 网格的列数。
    :param min_offset: 控制点的最小偏移。
    :param max_offset: 控制点的最大偏移。
    :return: 控制点张量和权重张量。
    """
    device = src_img.device
    src_h, src_w = src_img.shape[2:]
    grid_rows = row_block_num + 3  # B 样条在边界需要额外考虑
    grid_cols = col_block_num + 3

    # 初始化控制点 (x, y)，带有随机偏移
    control_points = torch.zeros((grid_rows, grid_cols, 2), dtype=torch.float32, device=device)
    control_points[..., 0] = torch.empty((grid_rows, grid_cols), device=device).uniform_(min_offset, max_offset)
    control_points[..., 1] = torch.empty((grid_rows, grid_cols), device=device).uniform_(min_offset, max_offset)

    # 初始化权重，默认设置为 1
    # weights = torch.ones((grid_rows, grid_cols), dtype=torch.float32, device=device).uniform_(min_offset, max_offset)
    weights = torch.ones((grid_rows, grid_cols), dtype=torch.float32, device=device)

    return control_points, weights


def nurbs_ffd_torch(src_img, row_block_num, col_block_num, control_points, weights, batch_size):
    """
    基于NURBS的自由形变，使用PyTorch实现。
    :param src_img: 输入图像张量，形状为 (N, C, H, W)
    :param row_block_num: 控制网格的行数。
    :param col_block_num: 控制网格的列数。
    :param control_points: 控制点张量，形状为 (grid_rows, grid_cols, 2)
    :param weights: 权重张量，形状为 (grid_rows, grid_cols)
    :return: 形变后的图像张量。
    """
    device = src_img.device
    _, _, src_h, src_w = src_img.shape

    delta_x = src_w / col_block_num
    delta_y = src_h / row_block_num
    grid_rows = row_block_num + 3
    grid_cols = col_block_num + 3

    # 生成像素坐标网格
    x_coords = torch.arange(0, src_w, device=device)
    y_coords = torch.arange(0, src_h, device=device)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
    x_grid = x_grid.float()  # 形状 (H, W)
    y_grid = y_grid.float()

    # 计算块索引和局部坐标
    x_block = x_grid / delta_x
    y_block = y_grid / delta_y
    i = torch.floor(y_block).long()
    j = torch.floor(x_block).long()
    u = x_block - j
    v = y_block - i

    # 有效性掩码
    valid_mask = (i >= 0) & (i <= grid_rows - 4) & (j >= 0) & (j <= grid_cols - 4)

    # 初始化形变场
    Tx = torch.zeros_like(x_grid)
    Ty = torch.zeros_like(y_grid)

    # 仅处理有效像素
    i_valid = i[valid_mask]
    j_valid = j[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]

    # 准备控制点索引
    m = torch.arange(0, 4, device=device).view(1, 4, 1)
    n = torch.arange(0, 4, device=device).view(1, 1, 4)
    control_point_y = i_valid.unsqueeze(-1).unsqueeze(-1) + m  # 形状 (N_valid, 4, 1)
    control_point_x = j_valid.unsqueeze(-1).unsqueeze(-1) + n  # 形状 (N_valid, 1, 4)
    control_point_indices = control_point_y * grid_cols + control_point_x  # 形状 (N_valid, 4, 4)

    # 扁平化控制点和权重
    control_points_flat = control_points.view(-1, 2)
    weights_flat = weights.view(-1)

    # 收集控制点和权重
    control_point_indices_flat = control_point_indices.view(-1)
    control_points_selected = control_points_flat[control_point_indices_flat]  # 形状 (N_valid * 16, 2)
    weights_selected = weights_flat[control_point_indices_flat]  # 形状 (N_valid * 16)

    # 重塑为 (N_valid, 4, 4, 2) 和 (N_valid, 4, 4)
    control_points_selected = control_points_selected.view(-1, 4, 4, 2)
    weights_selected = weights_selected.view(-1, 4, 4)

    # 计算不带权重的B样条基函数
    Bx = bspline_basis(u_valid)  # 形状 (N_valid, 4)
    By = bspline_basis(v_valid)  # 形状 (N_valid, 4)

    # 计算加权基函数
    weighted_basis = (By.unsqueeze(-1) * Bx.unsqueeze(-2)) * weights_selected  # 形状 (N_valid, 4, 4)

    # 计算归一化因子
    denominator = weighted_basis.sum(dim=(1, 2))  # 形状 (N_valid,)
    denominator[denominator == 0] = 1e-8  # 避免除以零

    # 计算NURBS基函数
    N = weighted_basis / denominator.unsqueeze(-1).unsqueeze(-1)  # 形状 (N_valid, 4, 4)

    # 计算Tx和Ty的分子
    Tx_numerator = (N * control_points_selected[..., 0]).sum(dim=(1, 2))
    Ty_numerator = (N * control_points_selected[..., 1]).sum(dim=(1, 2))

    # 计算形变
    Tx_valid = Tx_numerator
    Ty_valid = Ty_numerator

    # 将形变赋值给有效像素
    Tx[valid_mask] = Tx_valid
    Ty[valid_mask] = Ty_valid

    # 计算新的像素位置
    src_x = x_grid + Tx
    src_y = y_grid + Ty

    # 归一化坐标到 [-1, 1] 以适应 grid_sample
    src_x_norm = 2 * (src_x / (src_w - 1)) - 1
    src_y_norm = 2 * (src_y / (src_h - 1)) - 1
    grid = torch.stack((src_x_norm, src_y_norm), dim=-1)  # 形状 (H, W, 2)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # 形状 (N, H, W, 2)

    # 使用 grid_sample 进行图像采样
    dst_img = F.grid_sample(src_img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return dst_img


class Attacker():
    def __init__(self, model, img_attacker, txt_attacker):
        self.model = model
        self.img_attacker = img_attacker
        self.txt_attacker = txt_attacker

    def attack(self, imgs, txts, txt2img, device='cpu', max_length=30, scales=None, **kwargs):



        with torch.no_grad():
            txts_input = self.txt_attacker.tokenizer(txts, padding='max_length', truncation=True,
                                                     max_length=max_length, return_tensors="pt").to(device)
            txts_output = self.model.inference_text(txts_input)
            txt_supervisions = txts_output['text_feat']
        adv_imgs = self.img_attacker.txt_guided_attack(self.model, imgs, txt2img, device,
                                                       scales=scales, txt_embeds=txt_supervisions)

        with torch.no_grad():
            origin_img_output = self.model.inference_image(self.img_attacker.normalization(imgs))
            img_supervisions = origin_img_output['image_feat'][txt2img]
            adv_img_output = self.model.inference_image(self.img_attacker.normalization(adv_imgs))
            adv_img_supervisions = adv_img_output['image_feat'][txt2img]
        # adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions,
        #                                                adv_img_embeds=adv_img_supervisions)
        adv_txts = self.txt_attacker.img_guided_attack(self.model, txts, img_embeds=img_supervisions, adv_img_embeds=adv_img_supervisions)

        return adv_imgs, adv_txts


class ImageAttacker(VMIFGSM):
    def __init__(self, model, normalization, model_name='', epsilon=2 / 255, alpha=2 / 255, epoch=10, decay=1.,
                 mesh_width=30, mesh_height=30, rho=0.01, num_warping=20, noise_scale=2, beta=2, num_neighbor=20,
                 targeted=False, random_start=False, norm='linfty', loss='crossentropy',
                 device=None, attack='NURBS_Adaptive', **kwargs):
        super().__init__(
            model, model_name, epsilon, alpha, beta, num_neighbor, epoch, decay,
            targeted, random_start, norm, loss, device, attack
        )
        self.normalization = normalization
        self.diversity_prob = 0.7
        self.resize_rate = 1.1
        self.num_scale = 5
        self.kernel_size = 5
        self.alpha = alpha
        self.radius = beta * epsilon
        self.decay = decay
        self.num_neighbor = num_neighbor
        self.num_warping = num_warping
        self.noise_scale = noise_scale
        self.mesh_width = mesh_width
        self.mesh_height = mesh_height
        self.rho = rho
        self.beta = beta



    def dim(self, x, resize_rate, **kwargs):
        """
        Random transform the input images
        """
        # # do not transform the input image
        # if torch.rand(1) > self.diversity_prob:
        #     return x

        img_size = x.shape[-1]
        img_resize = int(img_size * resize_rate)

        # resize the input image to random size
        rnd = torch.randint(low=min(img_size, img_resize), high=max(img_size, img_resize), size=(1,), dtype=torch.int32)
        rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)

        # randomly add padding
        h_rem = img_resize - rnd
        w_rem = img_resize - rnd
        pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
        pad_bottom = h_rem - pad_top
        pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
        pad_right = w_rem - pad_left

        padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

        # resize the image back to img_size
        return F.interpolate(padded, size=[img_size, img_size], mode='bilinear', align_corners=False)



    def ni_transform(self, x, momentum):
        """
        look ahead for NI-FGSM
        """
        return x + self.alpha * self.decay * momentum

    def vwt(self, x, control_points, weights, batch_size):
        """
        使用新的 NURBS 变形函数对输入 x 进行变形。
        :param x: 输入图像张量，形状为 (N, C, H, W)
        :param control_points: 控制点张量，形状为 (N, grid_rows, grid_cols, 2)
        :param weights: 权重张量，形状为 (N, grid_rows, grid_cols)
        :return: 变形后的图像张量，形状为 (N, C, H, W)
        """
        # 设置控制网格
        row_block_num = self.mesh_height
        col_block_num = self.mesh_width

        # 调用新的 NURBS 变形函数
        vwt_x = nurbs_ffd_torch(x, row_block_num, col_block_num, control_points, weights, batch_size)

        return vwt_x



    def dem_trans(self, model, data, delta, label, momentum, **kwargs):
        """
        Calculate the gradient
        """

        # resize_rates = [1.14, 1.27, 1.4, 1.53, 1.66]


        grad = 0
        # ensemble
        for _ in range(self.num_warping):
            # Obtain the output
            x_min = self.dim(data + delta, torch.empty(1).uniform_(1.1, 1.8).item())

            # Calculate the output of the x_min
            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(x_min))
            else:
                adv_imgs_output = model.inference_image(x_min)
            logits = adv_imgs_output['image_feat']

            # Calculate the loss
            loss = calculate_wasserstein_distance(logits, label, 'cos')
            # print('dem_loss:', loss)


            # Calculate the gradients
            grad += F.conv2d(self.get_grad(loss, delta), weight=get_kernel(self.kernel_size).to(data.device), stride=(1, 1),
                             groups=3,
                             padding=(self.kernel_size - 1) // 2)

        return grad


    def vwt_trans(self, model, data, delta, label, momentum, **kwargs):
        """
        Calculate the gradient variance
        """

        grad = 0
        for k in range(self.num_warping):



            # 初始化控制点和权重
            control_points, weights = init_control_points(
                data, self.mesh_height, self.mesh_width, min_offset=-1, max_offset=1
            )

            control_points.requires_grad = True
            weights.requires_grad = True

            control_points = control_points.to(self.device)
            weights = weights.to(self.device)


            vwt_x = self.vwt(data + delta, control_points, weights,
                             data.shape[0])
            # vwt_x = self.vwt(self.ni_transform(datasets + delta, momentum=momentum), control_points, weights,
            #                  datasets.shape[0])

            # Calculate the output of the x_min
            if self.normalization is not None:
                adv_imgs_output = model.inference_image(self.normalization(vwt_x))
            else:
                adv_imgs_output = model.inference_image(vwt_x)
            logits = adv_imgs_output['image_feat']

            # Calculate the loss
            loss = calculate_wasserstein_distance(logits, label, 'cos')
            # print('vwt_loss:', loss)


            grad += F.conv2d(self.get_grad(loss, delta), weight=get_kernel(self.kernel_size).to(data.device), stride=(1, 1),
                             groups=3,
                             padding=(self.kernel_size - 1) // 2)

        return grad

    def no_trans(self, model, data, delta, label, momentum, **kwargs):
        """
        Calculate the gradient
        """



        # Obtain the output
        x_min = data + delta

        # Calculate the output of the x_min
        if self.normalization is not None:
            adv_imgs_output = model.inference_image(self.normalization(x_min))
        else:
            adv_imgs_output = model.inference_image(x_min)
        logits = adv_imgs_output['image_feat']

        # Calculate the loss
        loss = calculate_wasserstein_distance(logits, label, 'cos')

        # Calculate the gradients
        grad = self.get_grad(loss, delta)


        return grad





    def txt_guided_attack(self, model, imgs, txt2img, device, scales=None, txt_embeds=None):
        data = imgs.clone().detach().to(self.device)
        label = txt_embeds.clone().detach().to(self.device)

        # 初始化对抗扰动
        delta = self.init_delta(data)





        momentum, variance_dim, variance_vwt_points, variance_vwt_weights = 0, 0, 0, 0

        for _ in range(self.epoch):

            grad_dim = self.dem_trans(model, data, delta, label, momentum)
            grad_vwt = self.vwt_trans(model, data, delta, label,momentum)
            # grad_ori = self.no_trans(model, datasets, delta, label, momentum)



            noise = (grad_vwt + grad_dim) / 2



            # Calculate the momentum
            momentum = self.get_momentum(noise, momentum)



            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

            # ssim = self.msssim(datasets, datasets + delta)
            # print('ssim:', ssim)


            # generate_attention_map(datasets.detach().cpu().numpy(), delta.detach().cpu().numpy())

        adv_imgs = data + delta

        # noise_end = adv_imgs - imgs
        # noise_end = torch.where((noise_end > 0), torch.full_like(noise_end, 8 / 255), noise_end)
        # noise_end = torch.where((noise_end < 0), torch.full_like(noise_end, -8 / 255),
        #                         noise_end)
        # adv_imgs = imgs + noise_end

        # generate_attention_map(imgs.detach().cpu().numpy(), adv_imgs.detach().cpu().numpy())

        return adv_imgs


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves', '.', '-', 'a the', '/', '?', 'some', '"', ',', 'b', '&', '!',
                '@', '%', '^', '*', '(', ')', "-", '-', '+', '=', '<', '>', '|', ':', ";", '～', '·']
filter_words = set(filter_words)


class TextAttacker():
    def __init__(self, ref_net, tokenizer, cls=True, max_length=30, number_perturbation=1, topk=10,
                 threshold_pred_score=0.3, batch_size=32, device=None):

        self.ref_net = ref_net
        self.tokenizer = tokenizer
        self.max_length = max_length
        # epsilon_txt
        self.num_perturbation = number_perturbation
        self.threshold_pred_score = threshold_pred_score
        self.topk = topk
        self.batch_size = batch_size
        self.cls = cls
        self.substitute = get_default_substitute('glove', device)

    def get_wstar(self, sent, idx, net, img_embeds, adv_img_embeds, max_length=30):
        word = sent[idx]
        device = img_embeds.device

        rep_words = list(map(lambda x: x[0], self.substitute(word)))

        rep_words = list(filter(lambda x: x != word, rep_words))

        if len(rep_words) == 0:
            return (word, 0)

        sents = []
        for rw in rep_words:
            new_sent = sent[:idx] + [rw] + sent[idx + 1:]
            sents.append(' '.join(new_sent))

        sents_text_input = self.tokenizer(sents, padding='max_length', truncation=True,
                                           max_length=max_length, return_tensors='pt').to(device)
        sents_output = net.inference_text(sents_text_input)


        if self.cls:
            sents_embed = sents_output['text_feat'][:, 0, :].detach()
        else:
            sents_embed = sents_output['text_feat'].flatten(1).detach()




        import_scores = 1 - F.cosine_similarity(sents_embed, img_embeds, dim=-1) + (1 - F.cosine_similarity(sents_embed, adv_img_embeds, dim=-1))
        # import_scores = 1 - F.cosine_similarity(sents_embed, img_embeds, dim=-1)


        return rep_words[import_scores.argmax()], import_scores.max()

    def img_guided_attack(self, net, texts, img_embeds=None, adv_img_embeds=None):
        device = self.ref_net.device

        text_inputs = self.tokenizer(texts, padding='max_length', truncation=True, max_length=self.max_length,
                                     return_tensors='pt').to(device)

        # original state
        origin_output = net.inference_text(text_inputs)
        if self.cls:
            origin_embeds = origin_output['text_feat'][:, 0, :].detach()
        else:
            origin_embeds = origin_output['text_feat'].flatten(1).detach()

        final_adverse = []
        for i, text in enumerate(texts):
            # word importance eval
            S = self.get_important_scores(text, net, origin_embeds[i], self.batch_size, self.max_length)
            S = S * 10000

            S_softmax = torch.exp(S - S.max())
            S_softmax = S_softmax / S_softmax.sum()

            words, sub_words, keys = self._tokenize(text)

            w_star = [self.get_wstar(words, j, net, img_embeds[i:i+1], adv_img_embeds[i:i+1], max_length=30) for j in range(len(words))]
            H = [(idx, w_star[idx][0], S_softmax[idx] * w_star[idx][1]) for idx in range(len(words))]

            H = sorted(H, key=lambda x: -x[2])
            ret_sent = words.copy()
            change = 0

            for i in range(len(H)):
                if change >= self.num_perturbation:
                    break
                idx, wd, _ = H[i]
                if ret_sent[idx] in filter_words:
                    continue
                ret_sent[idx] = wd

                curr_sent = ' '.join(ret_sent)
                if ret_sent[idx] != words[idx]:
                    change += 1
            final_adverse.append(curr_sent)

        return final_adverse

    def loss_func(self, txt_embeds, img_embeds, label):
        loss_TaIcpos = -txt_embeds.mul(img_embeds[label].repeat(len(txt_embeds), 1)).sum(-1)
        loss = loss_TaIcpos
        return loss


    def _tokenize(self, text):
        words = text.split(' ')

        sub_words = []
        keys = []
        index = 0
        for word in words:
            sub = self.tokenizer.tokenize(word)
            sub_words += sub
            keys.append([index, index + len(sub)])
            index += len(sub)

        return words, sub_words, keys

    def _get_masked(self, text):
        words = text.split(' ')
        len_text = len(words)
        masked_words = []
        for i in range(len_text):
            masked_words.append(words[0:i] + ['[UNK]'] + words[i + 1:])
        # list of words
        return masked_words

    def get_important_scores(self, text, net, origin_embeds, batch_size, max_length):
        device = origin_embeds.device

        masked_words = self._get_masked(text)
        masked_texts = [' '.join(words) for words in masked_words]  # list of text of masked words

        masked_embeds = []
        for i in range(0, len(masked_texts), batch_size):
            masked_text_input = self.tokenizer(masked_texts[i:i + batch_size], padding='max_length', truncation=True,
                                               max_length=max_length, return_tensors='pt').to(device)
            masked_output = net.inference_text(masked_text_input)
            if self.cls:
                masked_embed = masked_output['text_feat'][:, 0, :].detach()
            else:
                masked_embed = masked_output['text_feat'].flatten(1).detach()
            masked_embeds.append(masked_embed)
        masked_embeds = torch.cat(masked_embeds, dim=0)

        criterion = torch.nn.KLDivLoss(reduction='none')

        import_scores = criterion(masked_embeds.log_softmax(dim=-1),
                                  origin_embeds.softmax(dim=-1).repeat(len(masked_texts), 1))

        return import_scores.sum(dim=-1)


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    # substitues L,k
    # from this matrix to recover a word
    words = []
    sub_len, k = substitutes.size()  # sub-len, k

    if sub_len == 0:
        return words

    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._convert_id_to_token(int(i)))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    #
    # print(words)
    return words


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    # substitutes L, k
    device = mlm_model.device
    substitutes = substitutes[0:12, 0:4]  # maximum BPE candidates

    # find all possible candidates

    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i

    # all substitutes  list of list of token-id (all candidates)
    c_loss = nn.CrossEntropyLoss(reduction='none')
    word_list = []
    # all_substitutes = all_substitutes[:24]
    all_substitutes = torch.tensor(all_substitutes)  # [ N, L ]
    all_substitutes = all_substitutes[:24].to(device)
    # print(substitutes.size(), all_substitutes.size())
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]  # N L vocab-size
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))  # [ N*L ]
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))  # N
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words
