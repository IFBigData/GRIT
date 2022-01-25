import torch
import torch.nn as nn
import torch.nn.functional as F

class GQM(nn.Module):
    def __init__(self, state_dim=10, num_node=5, num_class=5, num_layer=5, is_pipa=False):
        super(GQM, self).__init__()
        self.num_layer = num_layer
        self.state_dim = state_dim  # 2048
        self.hidden_dim = state_dim
        self.edge_types = num_class  # num of classes
        self.num_node = num_node
        self.is_pipa = is_pipa

        if is_pipa:
            self.age_gender_emb = nn.Linear(11, 128, bias=False)
            self.first_layer_node_map = nn.Linear(128 + self.hidden_dim, self.hidden_dim)

        # incoming and outgoing edge embedding
        self.node_fc = nn.ModuleList()
        self.node_act = nn.ModuleList()
        self.edge_fc = nn.ModuleList()
        self.edge_act = nn.ModuleList()
        for t in range(self.num_layer):
            self.node_fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.edge_fc.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            self.node_act.append(nn.LeakyReLU())
            self.edge_act.append(nn.LeakyReLU())
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialization()

    def forward(self, roi_feature_single, img_featmap, full_mask):
        # roi_feature: [batch_size, num_node, 2048]
        bs, hidden, height, width = img_featmap.shape
        full_mask = full_mask.view(-1, self.num_node, self.num_node, 1).detach()
        # full_mask_edge = full_mask.repeat(1, 1, 1, self.edge_types).float().detach()  # [batch_size, num_node, num_node, num_classes]
        full_mask_node = full_mask.repeat(1, 1, 1, self.hidden_dim).float().detach()

        node_prop_state = roi_feature_single

        if self.is_pipa:
            age_gender_emb = self.age_gender_emb(age_gender)
            node_prop_state = self.first_layer_node_map(torch.cat([node_prop_state, age_gender_emb], dim=-1))

        # #use edge attention
        # atten_edge_featmap, edge_scale_map = self.EdgeSpatialGate(img_featmap, roi_featmap_union)
        # atten_edge_featmap = self.EdgeSpatialGate(roi_featmap_union.reshape(-1, self.hidden_dim, 14, 14))
        # #no attention V1: use img_featmap as edge feature
        atten_edge_featmap = img_featmap.unsqueeze(1).repeat(1, self.num_node * self.num_node, 1, 1, 1).view(-1,
                                                                                                             self.hidden_dim,
                                                                                                             height, width)
        edge_feat = self.avgpool(atten_edge_featmap).squeeze(-1).squeeze(-1)  # [batch_size*num_node*num_node, 2048]
        edge_prop_state = edge_feat.view(-1, self.num_node, self.num_node, self.hidden_dim)
        # #no attention V2: use uoi featmap as edge feature
        # edge_prop_state = roi_feature_union.reshape(-1, self.num_node, self.num_node, self.hidden_dim)

        for t in range(self.num_layer):
            node_message_states = self.node_fc[t](node_prop_state)  # node feature, [batch, num_node, hidden_dim]
            edge_message_states = self.edge_fc[t](edge_prop_state)

            feature_row_large = node_message_states.contiguous().view(-1, self.num_node, 1, self.hidden_dim).repeat(1,
                                                                                                                    1,
                                                                                                                    self.num_node,
                                                                                                                    1)
            feature_col_large = node_message_states.contiguous().view(-1, 1, self.num_node, self.hidden_dim).repeat(1,
                                                                                                                    self.num_node,
                                                                                                                    1,
                                                                                                                    1)

            # original
            edge_feat = feature_row_large + feature_col_large + edge_message_states
            edge_prop_state = self.edge_act[t](edge_feat)
            edge_prob = edge_prop_state
            node_feat = (node_message_states + torch.sum(edge_prob * feature_col_large * full_mask_node, dim=-2)) / (
                        torch.sum(full_mask, dim=-2) + 1)
            # node_prop_state = node_prop_state + self.act(self.bn1(node_feat.permute(0,2,1)).permute(0,2,1))  # [batch_size, num_node, hidden_dim]
            node_prop_state = node_prop_state + self.node_act[t](node_feat)  # [batch_size, num_node, hidden_dim]

        roi_feature_col = node_prop_state.unsqueeze(2).repeat(1, 1, self.num_node, 1)
        roi_feature_row = node_prop_state.unsqueeze(1).repeat(1, self.num_node, 1, 1)
        roi_feature_all = torch.cat([roi_feature_col, roi_feature_row], dim=-1)

        # relation_score = self.mlp(roi_feature_all)  # [batch_size, node_num, node_num, num_classes]

        return roi_feature_all

    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.01)
                m.bias.data.fill_(0)