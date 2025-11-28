import torch
import torch.nn as nn
import torch.nn.functional as F

def _len2mask(length, max_len, dtype=torch.float32):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert (grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg


def tv_loss(img, img_lens):
    loss = (recn_l1_loss(img[:, :, 1:, :], img[:, :, :-1, :], img_lens) +
            recn_l1_loss(img[:, :, :, 1:], img[:, :, :, :-1], img_lens - 1)) / 2
    return loss


def recn_l1_loss(img1, img2, img_lens):
    mask = _len2mask(img_lens, img1.size(-1)).to(img1.device)
    diff_img = (img1 - img2) * mask.view(mask.size(0), 1, 1, mask.size(1))
    loss = diff_img.abs().sum() / (diff_img.size(1) * diff_img.size(2) * img_lens.sum())
    return loss


def calc_loss_perceptual(hout, hgt, img_lens):
    loss = 0
    for j in range(3):
        scale = 2 ** (3 - j)
        loss += recn_l1_loss(hout[j], hgt[j], img_lens // scale) / scale
    return loss


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def KLloss(mu, logvar):
    return torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)


##############################################################################
# Contrastive Style Loss (InfoNCE)
##############################################################################
class ContrastiveStyleLoss(nn.Module):
    """
    InfoNCE-style contrastive loss for style learning.
    Pulls same-writer samples together, pushes different writers apart.
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, style_vectors, writer_ids):
        """
        Args:
            style_vectors: [B, D] style vectors (should be normalized)
            writer_ids: [B] writer ID for each sample
        Returns:
            scalar loss
        """
        batch_size = style_vectors.size(0)
        device = style_vectors.device
        
        # Normalize style vectors
        style_vectors = F.normalize(style_vectors, dim=1)
        
        # Compute similarity matrix [B, B]
        sim_matrix = torch.matmul(style_vectors, style_vectors.T) / self.temperature
        
        # Create positive mask (same writer = 1)
        writer_ids = writer_ids.view(-1, 1)
        positive_mask = (writer_ids == writer_ids.T).float()
        
        # Remove self-similarity from positives
        positive_mask.fill_diagonal_(0)
        
        # Mask out self-similarity for denominator
        logits_mask = torch.ones_like(sim_matrix)
        logits_mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(sim_matrix) * logits_mask
        
        # Sum of positive similarities
        pos_sum = (exp_sim * positive_mask).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sum = exp_sim.sum(dim=1)
        
        # InfoNCE loss
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Only compute for samples with positive pairs
        valid_mask = positive_mask.sum(dim=1) > 0
        if valid_mask.sum() > 0:
            loss = loss[valid_mask].mean()
        else:
            loss = torch.tensor(0.0, device=device)
        
        return loss


##############################################################################
# Improved Perceptual Loss (Multi-scale)
##############################################################################
class MultiScalePerceptualLoss(nn.Module):
    """Multi-scale perceptual loss with learned weights."""
    
    def __init__(self, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales))
    
    def forward(self, fake_feats, real_feats, img_lens):
        """
        Args:
            fake_feats: list of feature maps from fake images
            real_feats: list of feature maps from real images
            img_lens: [B] image lengths
        """
        weights = F.softmax(self.scale_weights, dim=0)
        loss = 0
        
        for i, (fake_feat, real_feat) in enumerate(zip(fake_feats[:self.num_scales], 
                                                        real_feats[:self.num_scales])):
            scale = 2 ** (self.num_scales - i)
            scale_loss = recn_l1_loss(fake_feat, real_feat, img_lens // scale)
            loss += weights[i] * scale_loss
        
        return loss


##############################################################################
# Contextual loss
##############################################################################
class CXLoss(nn.Module):
    def __init__(self, sigma=0.5, b=1.0, similarity="consine"):
        super(CXLoss, self).__init__()
        self.similarity = similarity
        self.sigma = sigma
        self.b = b

    def center_by_T(self, featureI, featureT):
        # Calculate mean channel vector for feature map.
        meanT = featureT.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        # Normalize on channel dimension (axis=1)
        norms = features.norm(p=2, dim=1, keepdim=True)
        features = features.div(norms)
        return features

    def patch_decomposition(self, features):
        N, C, H, W = features.shape
        assert N == 1
        P = H * W
        # NCHW --> 1x1xCXHW --> HWxCx1x1
        patches = features.view(1, 1, C, P).permute((3, 2, 0, 1))
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        epsilon = 1e-5
        # [0] means get the value, torch min will return the index as well
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + epsilon)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        W = torch.exp((self.b - dist) / self.sigma)
        W_sum = W.sum(dim=axis, keepdim=True)
        return W.div(W_sum)

    def forward(self, featureT, featureI):
        '''
        :param featureT: target
        :param featureI: inference
        :return:
        '''

        # print("featureT target size:", featureT.shape)
        # print("featureI inference size:", featureI.shape)

        featureI, featureT = self.center_by_T(featureI, featureT)

        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist = []
        N = featureT.size()[0]
        for i in range(N):
            # NCHW
            featureT_i = featureT[i, :, :, :].unsqueeze(0)
            # NCHW
            featureI_i = featureI[i, :, :, :].unsqueeze(0)
            featureT_patch = self.patch_decomposition(featureT_i)
            # Calculate cosine similarity
            dist_i = F.conv2d(featureI_i, featureT_patch)
            dist.append(dist_i)

        # NCHW
        dist = torch.cat(dist, dim=0)

        raw_dist = (1. - dist) / 2.

        relative_dist = self.calc_relative_distances(raw_dist)

        CX = self.calc_CX(relative_dist)
        CX = torch.mean(CX.max(dim=3)[0].max(dim=2)[0], dim=1)
        CX = torch.mean(-torch.log(CX + 1e-5))
        return CX



##############################################################################
# Gram style loss
##############################################################################
class GramStyleLoss(nn.Module):
    def __init__(self):
        super(GramStyleLoss, self).__init__()
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def __call__(self, input_feat, target_feat, feat_len=None):
        input_gram = self.gram(input_feat, feat_len)
        target_gram = self.gram(target_feat, feat_len)
        loss = self.criterion(input_gram, target_gram)
        return loss


class GramMatrix(nn.Module):
    def forward(self, input, feat_len=None):
        a, b, c, d = input.size()

        if feat_len is not None:
            # mask for varying lengths
            mask = _len2mask(feat_len, d).view(a, 1, 1, d)
            input = input * mask

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # compute the gram product

        return G.div(a * b * c * d)
