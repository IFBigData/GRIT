import torch.nn as nn
import torch
from model.backbone_roi import resnet101_roi
from model.backbone_roi import transformer_roi
from model.transformer import build_transformer
from model.position_encoding import build_position_encoding
from model.gqm import GQM as gcn_model


class GRIT(nn.Module):
    def __init__(self, args, num_class, max_person):
        super(GRIT, self).__init__()
        self.max_person = max_person
        self.hidden_dim = args.hidden_dim
        self.img_size = args.img_size
        if args.backbone == 'resnet101':
            self.backbone, backbone_dim = resnet101_roi()
        else:
            self.backbone, backbone_dim = transformer_roi(args.img_size, args.backbone)

        if not args.remove_transformer:
            self.transformer = build_transformer(args)
            self.position_encoding = build_position_encoding(args)
            self.img_proj_trans = nn.Conv2d(backbone_dim, args.hidden_dim, kernel_size=1)
            self.position_encoding_decoder = build_position_encoding(args)
        else:
            self.transformer = None

        if not args.remove_gcn:
            self.gcn = gcn_model(state_dim=backbone_dim, num_node=max_person, num_class=num_class, num_layer=2)
        else:
            self.gcn = None
        self.roi_featmap = nn.Linear(backbone_dim * 2, args.hidden_dim)  # reduce dimension

        self.mlp = nn.Sequential(nn.Linear(args.hidden_dim, args.hidden_dim//2),
                                 nn.LeakyReLU(),
                                 nn.Linear(args.hidden_dim//2, num_class)
                                 )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        img, img_bbox = input['img'], input['image_bboxes']
        roi_feature, roi_featmap, img_featmap = self.backbone(img, img_bbox)
        # roi_feature: [batch_size, num_node*(num_node+1), backbone_dim]
        # roi_featmap: [batch_size, num_node*(num_node+1), backbone_dim, featmap_dim, featmap_dim]
        # img_featmap: [batch_size, backbone_dim, featmap_dim, featmap_dim]

        # obtain decoder input
        bbox_emb = roi_feature  # [batch_size, max_person, hidden_dim]

        if self.gcn is not None:
            img_featmap_gcn = img_featmap
            gcn_output = self.gcn(bbox_emb, img_featmap_gcn, input['relation_mask'] + input['relation_mask'].permute(0, 2, 1))
            relation_emb = self.roi_featmap(gcn_output).flatten(start_dim=1, end_dim=2)  # [batch_size, max_person*max_person, hidden_dim]
        else:
            bbox_emb_row = bbox_emb.contiguous().unsqueeze(2).repeat(1, 1, self.max_person, 1)
            bbox_emb_col = bbox_emb.contiguous().unsqueeze(1).repeat(1, self.max_person, 1, 1)
            relation_emb = self.roi_featmap(torch.cat([bbox_emb_row, bbox_emb_col], dim=-1))
            relation_emb = relation_emb.flatten(start_dim=1, end_dim=2)

        if self.transformer is not None:
            img_featmap_trans = self.img_proj_trans(img_featmap)
            pos_emb = self.position_encoding(img_featmap_trans)  # [batch_size, hidden_dim, 14, 14], same shape as img_featmap
            pos_emb_decoder = self.position_encoding_decoder(relation_emb.permute(0,2,1).view(-1, self.hidden_dim, self.max_person, self.max_person))
            relation_mask = input['relation_mask'].flatten(1)  # [batch_size, max_person*max_person]
            trans_out = self.transformer(img_featmap_trans, relation_emb, relation_mask, pos_emb, pos_emb_decoder)
            last_hidden_state = trans_out[0][-1]  # [batch_size, max_person*max_person, hidden_dim]
        else:
            last_hidden_state = relation_emb

        return self.mlp(last_hidden_state)
