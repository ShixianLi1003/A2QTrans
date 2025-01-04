# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from re import X
from matplotlib.cbook import flatten

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from torch.autograd import Function
import torch.nn.functional as F

import models.configs as configs

debug_mode = False # For debug
if debug_mode: import random



ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
logger = logging.getLogger(__name__)

ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):
        BCE_loss = torch.nn.CrossEntropyLoss()(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

"""slef-attention"""
class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        self.softmax2 = Softmax(dim=-2)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states) 
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) 
        key_layer = self.transpose_for_scores(mixed_key_layer) 
        value_layer = self.transpose_for_scores(mixed_value_layer) 

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) 
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) 


        attention_probs = self.softmax(attention_scores)
        weights = attention_probs 
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output) 
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        # EXPERIMENTAL. Overlapping patches:
        overlap = False
        if overlap: slide = 12 # 14

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])

            if overlap:
                n_patches = ((img_size[0] - patch_size[0]) // slide + 1) * ((img_size[1] - patch_size[1]) // slide + 1)
            else:
                n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

            self.hybrid = False

        if overlap:
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=(slide, slide) )                 
        else:
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                        out_channels=config.hidden_size,
                                        kernel_size=patch_size,
                                        stride=patch_size )

        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)


        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config)

    def forward(self, x, mask=None):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))

class hash(Function):
    @staticmethod
    def forward(ctx,input):
        return torch.sign(input)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
def hash_layer(input):
    return hash.apply(input)

class BE_Module(nn.Module):
    def __init__(self):
        super(BE_Module, self).__init__()

    def forward(self, mask, hidden_states):

        mask_cls = torch.zeros(mask.size(0), (mask.size(1) + 1)).to(mask.device)
        mask_cls[:, 1:] = mask[:, :]
        mask_cls[:, 0] = 1


        mask_cls = mask_cls.unsqueeze(-1)  # [B, 626, 1]
        mask_hidden_states = hidden_states * mask_cls # [B, 626, 768]
        mask_hidden_states = mask_hidden_states.half()
        return mask_hidden_states


class ATS_Module(nn.Module):
    def __init__(self):
        super(ATS_Module, self).__init__()
        self.threshold = nn.Parameter(torch.tensor(0.0),requires_grad=True)
    def forward(self,x, hidden_states):
        if isinstance(x, list):
            length = len(x)
            attn_map = x[0]
            for i in range(1, length):
                attn_map = torch.matmul(x[i], attn_map)
            attn_map = attn_map[:,:,0,1:]
        else:
            attn_map = x[:,:,0,1:]
        flattened_map = attn_map.reshape(attn_map.shape[0], -1)
        top_values, top_indices = torch.topk(flattened_map, k=int(attn_map.shape[2]))


        col_indices = top_indices % attn_map.shape[2]
        row_indices = top_indices // attn_map.shape[2]
        original_indices = torch.stack((row_indices, col_indices), dim=2)
        original_indices = original_indices[:,:,1]

        select_inx = original_indices + 1
        parts = []
        B, num = select_inx.shape

        for i in range(B):
            parts.append(hidden_states[i, select_inx[i,:]])
        noncls_hidden_states = torch.stack(parts).squeeze(1)

        attn_scores = top_values / top_values.sum(dim=-1, keepdim=True) # [B,625]

        mask = hash_layer(attn_scores - self.threshold)
        mask = (mask + 1) / 2
        mask = torch.round(mask)  
        mask = mask.unsqueeze(-1)

        new_noncls_hidden_states = noncls_hidden_states * mask # [B, 626, 768]
        new_noncls_hidden_states = new_noncls_hidden_states.half()
        non_zero_tokens_mask = (new_noncls_hidden_states != 0).any(dim=-1)
        non_zero_token_counts = non_zero_tokens_mask.sum(dim=1)
        max_non_zero_tokens = non_zero_token_counts.max()
        if max_non_zero_tokens <= 12:
            max_non_zero_tokens = 12

        filtered_hidden_states = []
        for i in range(new_noncls_hidden_states.size(0)):  
            non_zero_tokens = new_noncls_hidden_states[i][non_zero_tokens_mask[i]]
            if non_zero_tokens.size(0) < max_non_zero_tokens:
                padding_size = max_non_zero_tokens - non_zero_tokens.size(0)
                padding = torch.zeros((padding_size, new_noncls_hidden_states.size(-1)), device=new_noncls_hidden_states.device).half()
                non_zero_tokens = torch.cat([non_zero_tokens, padding], dim=0)
            filtered_hidden_states.append(non_zero_tokens)

        filtered_hidden_states = torch.stack(filtered_hidden_states)
        new_hidden_states = torch.cat((hidden_states[:,0].unsqueeze(1), filtered_hidden_states), dim=1)

        threshold_loss = torch.abs(self.threshold - 0.001).mean()

        return new_hidden_states, threshold_loss

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        num_layers = config.transformer["num_layers"] - 1
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.ATS_Module_11 = ATS_Module()
        self.ATS_Module_12 = ATS_Module()
        self.BE_Module = BE_Module()
        self.last_layer1 = Block(config)
        self.last_layer2 = Block(config)
        self.ATS_layer = Block(config)

        for _ in range(num_layers):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, mask=None):
        attn_weights = []
        th_loss = torch.tensor(0, dtype=torch.float16)
        th_loss = th_loss.to(torch.device("cuda"))
        layer_number = 1
        total_threshold_loss = 0

        for layer_block in self.layer:
            if layer_number == 11:
                th_hidden_states, threshold_loss = self.ATS_Module_11(attn_weights, hidden_states)
                total_threshold_loss += threshold_loss
                th_hidden_states, th_attn_weights = self.ATS_layer(th_hidden_states, mask)

                mask_hidden_states,mask_weights = layer_block(hidden_states,mask)
            else:
                hidden_states, weights = layer_block(hidden_states, mask)
                attn_weights.append(weights)
            layer_number += 1

        th_hidden_states,threshold_loss = self.ATS_Module_12(th_attn_weights, th_hidden_states)
        total_threshold_loss += threshold_loss
        th_hidden_states, th_attn_weights = self.last_layer1(th_hidden_states, mask)

        mask_hidden_states = self.BE_Module(mask,mask_hidden_states)
        mask_hidden_states, mask_weights = self.last_layer2(mask_hidden_states,mask)

        th_encoded = self.encoder_norm(th_hidden_states)
        mask_encoded = self.encoder_norm(mask_hidden_states)

        return th_encoded, mask_encoded, th_attn_weights, mask_weights, total_threshold_loss

        
class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids, mask=None):
        embedding_output = self.embeddings(input_ids)
        th_encoded, mask_encoded, th_attn_weights, mask_weights, total_threshold_loss = self.encoder(embedding_output, mask)

        return th_encoded, mask_encoded, th_attn_weights, mask_weights, total_threshold_loss

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=400, num_classes=200, smoothing_value=0, zero_head=False, dataset='CUB', contr_loss=False, focal_loss=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.smoothing_value = smoothing_value
        self.classifier = config.classifier
        self.dataset=dataset

        self.contr_loss = contr_loss
        self.focal_loss = focal_loss

        self.transformer = Transformer(config, img_size)
        self.head = Linear(config.hidden_size, num_classes)

        self.logits_w1 = nn.Parameter(torch.tensor(0.5))  
        self.logits_w2 = nn.Parameter(torch.tensor(0.5))  

    def forward(self, x, labels=None, mask=None):
        x1, x2, th_attn_weights, mask_weights, threshold_loss = self.transformer(x, mask) 


        logits1 = self.head(x1[:, 0])
        logits2 = self.head(x2[:, 0])
        w1_expanded = self.logits_w1.unsqueeze(0)
        w2_expanded = self.logits_w2.unsqueeze(0)
        w1_normalized = torch.nn.functional.softmax(torch.cat([w1_expanded, w2_expanded]), dim=0)[0]
        w2_normalized = torch.nn.functional.softmax(torch.cat([w1_expanded, w2_expanded]), dim=0)[1]
        logits = w1_normalized * logits1 + w2_normalized * logits2
        
        if labels is not None:
            if self.smoothing_value == 0:
                loss_fct = CrossEntropyLoss()
            else:
                loss_fct = LabelSmoothing(self.smoothing_value)

            ce_loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1).long())
            if self.contr_loss:
                contrast_loss1 = con_loss(x1[:, 0], labels.view(-1))
                contrast_loss2 = con_loss(x2[:, 0], labels.view(-1))
                contrast_loss = contrast_loss1 + contrast_loss2
                loss = ce_loss + 0.01*contrast_loss + threshold_loss  
            else:
                loss = ce_loss + threshold_loss  
            return loss, logits
        else:
            return logits, th_attn_weights, mask_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))


            layer_idx = 0
            for bname, block in self.transformer.encoder.named_children():
                if bname.startswith('last') == False:
                    for uname, unit in block.named_children():
                        if layer_idx == 10 and bname.startswith('th') == True:
                            self.transformer.encoder.ATS_layer.load_from(weights, n_block=uname)
                        elif bname.startswith('th') == False:
                            unit.load_from(weights, n_block=uname)
                        layer_idx += 1


            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)


def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss                        
                        

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
}
