"""
Basic ORTrack model.
"""
import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.utils.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy

import numpy as np
from scipy.stats import multivariate_normal
import cv2
from lib.models.dmftrack.deit import deit_tiny_patch16_224, deit_tiny_patch16_224_distill, deit_tiny_distilled_patch16_224
from lib.models.dmftrack.vision_transformer import vit_tiny_patch16_224, vit_tiny_distilled_patch16_224
from lib.models.dmftrack.eva import eva02_tiny_patch14_224, eva02_tiny_patch14_224_distill
from lib.models.dmftrack.dmftrack import DMFTrack
from lib.models.layers.score_decoder import ScoreDecoder

class DMFTrackST(DMFTrack):
    """ This is the base class for DMFTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", cls_head=None, run_cls_head=False):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            box_head: torch module for box prediction
            head_type: Type of prediction head
            cls_head: Classification head module
        """
        super().__init__(transformer, box_head, aux_loss=aux_loss, head_type=head_type)
        
        # Classification head for spatio-temporal modeling
        self.cls_head = cls_head
        self.run_cls_head = run_cls_head

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                is_distill=False,
                ):

        x, aux_dict = self.backbone(z=template, x=search) # [32, 320, 192],['attn']：None

        # Forward head
        feat_last = x  # [32, 320, 192]
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        
        return out

    def forward_box_head(self, cat_feature, gt_score_map=None):
            """
            cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C),also is encoder embeddings
            """
            enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C), cat_feature:[32, 320, 192]
            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()  # [32, 1, 192, 256]
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s) # [32, 192, 16, 16]
            
            # 输出的pred_boxes统一为"CENTER"格式（cx,cy,w,h）
            if self.head_type == "CORNER":
                # run the corner head
                pred_box, score_map = self.box_head(opt_feat, True)
                outputs_coord = box_xyxy_to_cxcywh(pred_box)
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map,
                    }
                return out

            elif self.head_type == "CENTER":
                # run the center head
                score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
                outputs_coord = bbox
                outputs_coord_new = outputs_coord.view(bs, Nq, 4)
                out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
                return out
            else:
                raise NotImplementedError

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C),also is encoder embeddings
        """
        out = {} # add this 
        out_box = self.forward_box_head(cat_feature, gt_score_map)
        out.update(out_box)
        
        # Classification head
        if self.run_cls_head:
            # Process template region features for classification
            tem_opt = (cat_feature[:, :self.feat_len_t].unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = tem_opt.size()
            template_feat = tem_opt.view(-1, C, self.feat_sz_t, self.feat_sz_t) # ->(B,C,H,W)[32, 192, 8, 8] 
            # Process search region features for classification
            enc_opt = (cat_feature[:, -self.feat_len_s:].unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = enc_opt.size()
            search_feat = enc_opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
            
            pred_box = out['pred_boxes']
            gt_bboxes = box_cxcywh_to_xyxy(pred_box.clone().view(-1, 4))
            # (b,c,h,w) --> (b,h,w)
            pred_logits = self.cls_head(search_feat, template_feat, gt_bboxes).view(-1)
            out.update({'pred_logits': pred_logits})

        return out  

def build_dmftrackst(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('DMFTrack_ST' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        # backbone = deit_tiny_distilled_patch16_224(num_classes=0, pretrained=True)
        # patch_start_index = 2 # DeiT-distilled 有 cls_token + dist_token 两个特殊 token，图像 token 从 2 开始（这里从一改到二是因为预训练权重改为使用DeiT-distilled的）
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224':
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224':
        backbone = vit_tiny_distilled_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_distilled_patch14_224':
        backbone = eva02_tiny_patch14_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError


    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224' or cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224' \
           or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224' \
            or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_distilled_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224':
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    # Add classification head for spatio-temporal version
    # hidden_dim = cfg.MODEL.HIDDEN_DIM # 192
    cls_head = ScoreDecoder(pool_size=4, hidden_dim=hidden_dim, num_heads=hidden_dim//64, nlayer_head=cfg.MODEL.NLAYER_HEAD)  # the proposed score prediction module (SPM)

    model = DMFTrackST(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        cls_head=cls_head,
        run_cls_head=getattr(cfg.TRAIN, 'TRAIN_CLS', False)
    )

    if 'DMFTrack_ST' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
