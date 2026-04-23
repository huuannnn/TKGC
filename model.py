import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import *
import copy

class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim, out_dim)
        )

    def forward(self, x):
        return self.linear(x)


class CENET(nn.Module):
    def __init__(self, num_e, num_rel, num_t, embedding_dim, dropout, lambdax, alpha, oracle_mode, filtering):
        super(CENET, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.embedding_dim = embedding_dim
        self.lambdax = lambdax
        self.alpha = alpha
        self.oracle_mode = oracle_mode
        self.filtering = filtering

        # entity relation embedding
        self.rel_embeds = nn.Parameter(torch.zeros(2 * num_rel, embedding_dim))
        self.entity_embeds = nn.Parameter(torch.zeros(self.num_e, embedding_dim))

        self.linear_frequency = nn.Linear(self.num_e, embedding_dim)

        self.contrastive_hidden_layer = nn.Linear(3 * embedding_dim, embedding_dim)
        self.oracle_layer = Oracle(3 * embedding_dim, 1)

        self.linear_pred_layer_s1 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * embedding_dim, embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * embedding_dim, embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * embedding_dim, embedding_dim)

        # Initialize weights and embeddings
        self.reset_parameters()

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.crossEntropy = nn.BCELoss()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.rel_embeds, gain=gain)
        nn.init.xavier_uniform_(self.entity_embeds, gain=gain)
        
        # Initialize layers
        self.weights_init(self.linear_frequency)
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)
        self.oracle_layer.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            
    def history_and_non_history_depedency(self, s_frequency, o_frequency, lambdax):
        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = lambdax
        o_history_tag[o_history_tag != 0] = lambdax

        s_non_history_tag[s_history_tag == 1] = -lambdax
        s_non_history_tag[s_history_tag == 0] = lambdax

        o_non_history_tag[o_history_tag == 1] = -lambdax
        o_non_history_tag[o_history_tag == 0] = lambdax

        s_history_tag[s_history_tag == 0] = -lambdax
        o_history_tag[o_history_tag == 0] = -lambdax

        return s_history_tag, o_history_tag, s_non_history_tag, o_non_history_tag

    def forward(self, batch_block, mode_lk, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block

        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            return (None, None) if mode_lk in ['Valid', 'Test'] else None

        s, r, o = quadruples[:, :3].T
        s_history_tag, o_history_tag,\
            s_non_history_tag, o_non_history_tag = self.history_and_non_history_depedency(s_frequency, o_frequency, self.lambdax)

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        if mode_lk == 'Training':
            s_preds_his, s_preds_non_his = self.compute_predictions(s, r, self.rel_embeds[:self.num_rel],
                                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                                    s_history_tag, s_non_history_tag)
            s_preds = s_preds_his + s_preds_non_his
            s_nce_loss, _ = self.calculate_nce_loss(s_preds, o)
            
            o_preds_his, o_preds_non_his = self.compute_predictions(o, r, self.rel_embeds[self.num_rel:],
                                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                                    o_history_tag, o_non_history_tag)
            o_preds = o_preds_his + o_preds_non_his
            o_nce_loss, _ = self.calculate_nce_loss(o_preds, s)
            
            s_spc_proj = self.compute_spc_projections(s, r, self.rel_embeds[:self.num_rel], s_frequency_hidden)
            s_spc_loss = self.calculate_spc_loss(s_spc_proj, s_history_label_true)
            
            o_spc_proj = self.compute_spc_projections(o, r, self.rel_embeds[self.num_rel:], o_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o_spc_proj, o_history_label_true)
            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            # print('nce loss', nce_loss.item(), ' spc loss', spc_loss.item())
            return self.alpha * nce_loss + (1 - self.alpha) * spc_loss

        elif mode_lk in ['Valid', 'Test']:
            # Build history listsa
            s_history_oid, o_history_sid = self.build_history_lists(s_history_event_o, o_history_event_s, quadruples)

            s_preds_his, s_preds_non_his = self.compute_predictions(s, r, self.rel_embeds[:self.num_rel],
                                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2,
                                                                    s_history_tag, s_non_history_tag)
            s_preds = s_preds_his + s_preds_non_his
            s_nce_loss, _ = self.calculate_nce_loss(s_preds, o)
            
            o_preds_his, o_preds_non_his = self.compute_predictions(o, r, self.rel_embeds[self.num_rel:],
                                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2,
                                                                    o_history_tag, o_non_history_tag)
            o_preds = o_preds_his + o_preds_non_his
            o_nce_loss, _ = self.calculate_nce_loss(o_preds, s)

            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)

            # Compute predicted masks
            s_mask = self.compute_predicted_mask(s_pred_history_label, s_history_oid, quadruples)
            o_mask = self.compute_predicted_mask(o_pred_history_label, o_history_sid, quadruples)

            # Compute ground truth masks
            s_mask_gt = self.compute_gt_mask(o, s_history_oid, quadruples)
            o_mask_gt = self.compute_gt_mask(s, o_history_sid, quadruples)


            # Evaluate all three mask variants
            s_total_loss1, sub_rank1, s_total_loss2, sub_rank2, s_total_loss3, sub_rank3 = \
                self.evaluate_predictions(s_nce_loss, s_preds, s_ce_loss, s, o, r, s_mask, s_mask_gt, total_data, 's')
            
            o_total_loss1, obj_rank1, o_total_loss2, obj_rank2, o_total_loss3, obj_rank3 = \
                self.evaluate_predictions(o_nce_loss, o_preds, o_ce_loss, o, s, r, o_mask, o_mask_gt, total_data, 'o')

            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0
            batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0

            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2, \
                   sub_rank3, obj_rank3, batch_loss3, \
                   (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk == 'Oracle':
            # print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01)

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = F.sigmoid(
            self.oracle_layer(torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    def compute_predictions(self, actor1, r, rel_embeds, linear_layer1, linear_layer2, history_tag, non_history_tag):
        sp_transform = self.dropout(torch.cat((self.entity_embeds[actor1], rel_embeds[r]), dim=1))
        
        H_his = self.tanh(linear_layer1(sp_transform)) @ self.entity_embeds.transpose(0, 1) + history_tag
        H_his = F.softmax(H_his, dim=1)

        H_non_his = self.tanh(linear_layer2(sp_transform)) @ self.entity_embeds.transpose(0, 1) + non_history_tag
        H_non_his = F.softmax(H_non_his, dim=1)

        return H_his, H_non_his

    def calculate_nce_loss(self, preds, actor2):
        H = torch.log(preds + 1e-10)  # Add epsilon for numerical stability
        nce = torch.sum(torch.gather(H, 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]

        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]

        return nce, preds

    def apply_oracle_mask(self, preds, trust_musk, oracle):
        if oracle:
            preds = torch.mul(preds, trust_musk)
        return preds

    def filter_predictions(self, preds, actor1, r, all_triples, pred_known):
        if not self.filtering:
            return preds

        preds_filtered = preds.clone()
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            ground = preds[i, preds[i].argmax()].clone().item()
            
            if pred_known == 's':
                s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                idx = all_triples[s_id[idx], 2]
            else:
                s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                idx = all_triples[s_id[idx], 0]
            
            preds_filtered[i, idx] = 0
        
        return preds_filtered

    def compute_ranks(self, preds, actor2):
        ranks = []
        for i in range(preds.shape[0]):
            ground = preds[i, actor2[i]].item()
            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            rank = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
            ranks.append(rank)
        return ranks

    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle):
        preds = self.apply_oracle_mask(preds, trust_musk, oracle)
        preds = self.filter_predictions(preds, actor1, r, all_triples, pred_known)
        ranks = self.compute_ranks(preds, actor2)
        total_loss = nce_loss + ce_loss
        return total_loss, ranks

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    def build_history_lists(self, s_history_event_o, o_history_event_s, quadruples):
        s_history_oid = []
        o_history_sid = []
        for i in range(quadruples.shape[0]):
            s_history_oid.append([])
            o_history_sid.append([])
            for con_events in s_history_event_o[i]:
                s_history_oid[-1] += con_events[:, 1].tolist()
            for con_events in o_history_event_s[i]:
                o_history_sid[-1] += con_events[:, 1].tolist()
        return s_history_oid, o_history_sid

    def compute_predicted_mask(self, pred_history_label, history_ids, quadruples):
        mask = torch.zeros(quadruples.shape[0], self.num_e, device=quadruples.device)
        for i in range(quadruples.shape[0]):
            if pred_history_label[i].item() > 0.5:
                mask[i, history_ids[i]] = 1
            else:
                mask[i, :] = 1
                mask[i, history_ids[i]] = 0
        if self.oracle_mode == 'soft':
            mask = F.softmax(mask, dim=1)
        return mask

    def compute_gt_mask(self, actor_target, history_ids, quadruples):
        mask_gt = torch.zeros(quadruples.shape[0], self.num_e, device=quadruples.device)
        for i in range(quadruples.shape[0]):
            if actor_target[i] in history_ids[i]:
                mask_gt[i, history_ids[i]] = 1
            else:
                mask_gt[i, :] = 1
                mask_gt[i, history_ids[i]] = 0
        return mask_gt

    def evaluate_predictions(self, nce_loss, preds, ce_loss, actor1, actor2, r, mask_pred, mask_gt, total_data, pred_known):
        # Variant 1: Oracle mode with filtering
        total_loss1, rank1 = self.link_predict(nce_loss, preds, ce_loss, actor1, actor2, r,
                                              mask_pred, total_data, pred_known, True)
        
        # Variant 2: Without filtering
        total_loss2, rank2 = self.link_predict(nce_loss, preds, ce_loss, actor1, actor2, r,
                                              mask_pred, total_data, pred_known, False)
        
        # Variant 3: Ground truth mask
        total_loss3, rank3 = self.link_predict(nce_loss, preds, ce_loss, actor1, actor2, r,
                                              mask_gt, total_data, pred_known, True)
        
        return total_loss1, rank1, total_loss2, rank2, total_loss3, rank3

    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)

    def contrastive_layer(self, x):
        x = self.contrastive_hidden_layer(x)
        return x

    def compute_spc_projections(self, actor1, r, rel_embeds, frequency_hidden):
        projections = self.contrastive_layer(
            torch.cat((self.entity_embeds[actor1], rel_embeds[r], frequency_hidden), dim=1))
        return projections

    def calculate_spc_loss(self, projections, targets):
        targets = torch.squeeze(targets)
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        mask_similar_class = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(projections.device)
        mask_anchor_out = (1 - torch.eye(exp_dot_tempered.shape[0])).to(projections.device)
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)
        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return 0
        return supervised_contrastive_loss