import torch
import torch.nn as nn
from deepcluster.modeling.feature_modules.build_feature_module import build_feature_module
import random
import logging
from sklearn.cluster import KMeans
import numpy as np
from deepcluster.modeling.feature_modules.convnet import ExpNorm
import torch.nn.functional as F


class GaussianAttentionClusterHead(nn.Module):
    def __init__(self, classifier_ori, feature_conv=None, att_conv=None, theta_mlp=None, ro_mlp=None, classifier_att=None,
                 num_att_map=1, step=13, eta=None, loss="bce", balance_scores=False, num_cluster=10, batch_size=None,
                 sub_batch_size=None, ignore_label=-1, fea_height=7, fea_width=7, loss_weight=None, lamda=0.05, min_ratio=0.1,
                 mi_att=None, mi_att_oa=None, mi_att_ao=None):

        super(GaussianAttentionClusterHead, self).__init__()
        if loss_weight is None:
            loss_weight = dict(loss_fea=5, loss_rel=1, loss_att=1)
        self.classifier_ori = build_feature_module(classifier_ori)
        self.feature_conv = None
        if feature_conv:
            self.feature_conv = build_feature_module(feature_conv)
        self.num_att_map = num_att_map
        self.theta_mlp = None
        if theta_mlp:
            self.theta_mlp = True
            for i in range(num_att_map):
                theta_mlp_i = build_feature_module(theta_mlp)
                self.__setattr__("theta_mlp_{}".format(i), theta_mlp_i)
        self.ro_mlp = None
        if ro_mlp:
            self.ro_mlp = True
            for i in range(num_att_map):
                ro_mlp_i = build_feature_module(ro_mlp)
                self.__setattr__("ro_mlp_{}".format(i), ro_mlp_i)
        self.att_conv = None
        if att_conv:
            self.att_conv = build_feature_module(att_conv)
        self.classifier_att = None
        if classifier_att:
            self.classifier_att = build_feature_module(classifier_att)

        self.mi_att = None
        if mi_att is not None:
            self.mi_att = build_feature_module(mi_att)

        self.mi_att_oa = None
        if mi_att_oa is not None:
            self.mi_att_oa = build_feature_module(mi_att_oa)

        self.mi_att_ao = None
        if mi_att_ao is not None:
            self.mi_att_ao = build_feature_module(mi_att_ao)

        self.ave_pooling = nn.AvgPool2d((fea_height, fea_width))
        if loss == "bce":
            self.loss_fn = nn.BCELoss()
        else:
            raise TypeError

        self.loss_mse = nn.MSELoss()

        self.step = step
        self.eta = eta
        self.EPS = 1e-5
        self.num_cluster = num_cluster
        prob_pos = float(batch_size - 1) / (num_cluster * batch_size - 1)
        up_pos_num = int(batch_size * (batch_size - 1) * prob_pos + batch_size)
        low_neg_num = int((batch_size ** 2 - up_pos_num) * 0.9)
        self.min_neg_num = low_neg_num
        if loss_weight is None:
            loss_weight = dict(loss_fea=5, loss_relation=1, loss_att=1)
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.balance_scores = balance_scores

        tri_idx_x = []
        tri_idx_y = []
        idx_x = 0
        for i in range(sub_batch_size):
            for ii in range(i + 1):
                tri_idx_x.append(idx_x)
                tri_idx_y.append(ii)
            idx_x += 1

        self.tri_idx_x = tri_idx_x
        self.tri_idx_y = tri_idx_y

        self.CE = nn.CrossEntropyLoss()

        fea_coords = torch.zeros(fea_height * fea_width, 2)
        for i in range(fea_height):
            for j in range(fea_width):
                fea_coords[i * fea_height + j, 0] = i / float(fea_height - 1)
                fea_coords[i * fea_width + j, 1] = j / float(fea_width - 1)
        self.fea_coords = fea_coords
        self.lamda = lamda
        self.fea_height = fea_height
        self.fea_width = fea_width
        self.min_ratio = min_ratio

    def l2_norm(self, x):
        n = torch.pow(torch.sum(x * x, dim=1), 0.5).unsqueeze(dim=1) + self.EPS
        return x / n

    def compute_similarity_mtx(self, x):
        s = torch.mm(x, x.transpose(1, 0))
        return s

    def compute_kmeans(self, fea):
        fea_att_target = self.compute_attention_target(fea)
        fea_att_target_np = fea_att_target.cpu().numpy().astype(np.float64)
        kmeans = KMeans(n_clusters=self.num_cluster, n_init=20).fit(fea_att_target_np)
        self.labels_ = kmeans.labels_
        self.centers_ = torch.from_numpy(kmeans.cluster_centers_).to(fea_att_target.device).to(fea_att_target.dtype)

    def compute_relation_target(self, fea):

        predicted = self.labels_
        num = int(fea.shape[0])
        r = torch.zeros(num, num).to(fea.device)

        num_per_cluster = np.zeros(self.num_cluster)

        for c in range(self.num_cluster):
            num_per_cluster[c] = (predicted==c).sum()

        min_num = num * self.min_ratio / self.num_cluster
        num_cluster_pred = int(np.unique(predicted).shape[0])

        if num_cluster_pred < self.num_cluster or num_per_cluster.min() < min_num:
            dia_idx_x = list(range(num))
            dia_idx_y = list(range(num))
            r[dia_idx_x, dia_idx_y] = 1
        else:
            for c in range(self.num_cluster):
                cc = predicted == c
                assert cc.shape[0] > 0
                idx_c = list(np.argwhere(cc).squeeze())
                id_x = torch.zeros_like(r)
                id_y = torch.zeros_like(r)
                id_x[idx_c, :] = 1
                id_y[:, idx_c] = 1
                id = id_x * id_y
                r[id==1] = 1
        return r

    def compute_attention_target(self, q):
        p = torch.pow(q, 2) / (q.sum(dim=0) + +self.EPS)
        p = p / (p.sum(dim=1).unsqueeze(dim=1)+self.EPS)
        return p

    def sample_selection(self, s, t):
        s_select = s[self.tri_idx_x, self.tri_idx_y]
        t_select = t[self.tri_idx_x, self.tri_idx_y]
        select_idx = t_select != self.ignore_label
        return s_select[select_idx], t_select[select_idx]

    def select_false_fea(self, s, fea):
        num_fea = fea.shape[0]
        idx_select = []
        for i in range(num_fea):
            dis_sim_idx = torch.nonzero(s[i,:] == 0).view(-1)
            if len(dis_sim_idx) > 0:
                i_select = torch.randint(0, len(dis_sim_idx), [1,])
            else:
                if i < num_fea - 1:
                    i_select = i + 1
                else:
                    i_select = 0

            idx_select.append(i_select)

        return fea[idx_select, :]

    def generate_gaussian_att_map(self, theta, ro=None):
        N = theta.shape[0]
        mu = theta[:, ::2].view(N, 1, 2)
        self.fea_coords = self.fea_coords.to(theta.device)
        coords = self.fea_coords.view(1, -1, 2)
        if ro is not None:

            delta = theta[:, 2::].view(N, 1, 2) + self.EPS
            ro = ro.view(N, 1) / 2

            frac = 1 / (2 * self.lamda * (1 - ro ** 2))
            u = (coords[:, :, 0] - mu[:, :, 0]) / delta[:, :, 0]
            v = (coords[:, :, 1] - mu[:, :, 1]) / delta[:, :, 1]
            out = torch.exp(-frac * (u ** 2 - 2 * ro * u * v + v ** 2))

        else:
            delta = theta[:, -1].view(N, 1)
            out = torch.exp(-torch.sum((coords - mu) ** 2, dim=2) / (2 * self.lamda * delta + self.EPS))

        return out.view(N, 1, self.fea_height, self.fea_width)

    def compute_balance_socres(self, q):
        p = q / (q.sum(dim=0) + self.EPS)
        p = p / (p.sum(dim=1).unsqueeze(dim=1)+self.EPS)
        return p

    def reset_weights(self, obj, modules):
        for n in modules:
            m = obj.__getattr__(n)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, fea, return_target=False, return_att_map=False):

        if self.feature_conv is not None:
            fea_conv = self.feature_conv(fea)
        else:
            fea_conv = fea

        feature_ori = self.ave_pooling(fea_conv)
        feature_ori = feature_ori.flatten(start_dim=1)

        cluster_prob_ori = self.classifier_ori(feature_ori)

        if return_target:
            return feature_ori.detach(), cluster_prob_ori.detach()

        if self.theta_mlp is not None:

            if self.att_conv is not None:
                fea_att_conv = self.att_conv(fea)
            else:
                fea_att_conv = fea
            fea_flatten = fea_att_conv.flatten(start_dim=1)
            att_maps = []
            for i in range(self.num_att_map):

                theta_mlp_i = self.__getattr__("theta_mlp_{}".format(i))
                theta_i = theta_mlp_i(fea_flatten)
                ro_i = None
                if self.ro_mlp is not None:
                    ro_mlp_i = self.__getattr__("ro_mlp_{}".format(i))
                    ro_i = ro_mlp_i(fea_flatten)
                att_map_i = self.generate_gaussian_att_map(theta_i, ro_i)
                att_maps.append(att_map_i.unsqueeze(dim=2))

            att_map = torch.cat(att_maps, dim=2)
            att_map = att_map.sum(dim=2)

            feature_att = fea_conv * att_map  # spatial attention
            feature_att = feature_att.sum(dim=-1).sum(dim=-1)  # sum along the spatial dimension.
            if self.classifier_att:
                cluster_prob_att = self.classifier_att(feature_att)
            else:
                cluster_prob_att = self.classifier_ori(feature_att)

        else:
            feature_att = None
            cluster_prob_att = None
            att_map = None

        if return_att_map:
            return [feature_ori, cluster_prob_ori, feature_att, cluster_prob_att, att_map]
        else:
            return [feature_ori, cluster_prob_ori, feature_att, cluster_prob_att]

    def loss(self, x, target=None):
        [fea_ori, prob_ori, fea_att, prob_att] = self.forward(x)

        assert isinstance(target, list) and len(target) == 6

        [target_fea_ori, target_prob_ori, target_rel, target_fea_att, target_prob_att, entropy_loss] = target

        loss = {}
        model_info = {}

        if target_fea_ori is not None:
            # Similarity loss.
            loss_sim_prob = -(prob_ori * target_prob_ori).sum(dim=1).mean()
            loss["loss_similarity"] = loss_sim_prob * self.loss_weight["loss_sim"]

        if entropy_loss:
            prob_ori_mean = prob_ori.mean(dim=0)
            prob_ori_mean[(prob_ori_mean < self.EPS).data] = self.EPS
            loss_ent_ori = (prob_ori_mean * torch.log(prob_ori_mean)).sum()
            loss["loss_entropy_ori"] = loss_ent_ori * self.loss_weight["loss_ent"]

            if target_fea_att is not None:
                assert fea_att is not None
                prob_att_mean = prob_att.mean(dim=0)
                prob_att_mean[(prob_att_mean < self.EPS).data] = self.EPS
                loss_ent_att = (prob_att_mean * torch.log(prob_att_mean)).sum()
                loss["loss_entropy_att"] = loss_ent_att * self.loss_weight["loss_ent"]

        if target_rel is not None:
            s = self.compute_similarity_mtx(self.l2_norm(prob_ori))
            s_predict, s_true = self.sample_selection(s, target_rel)
            s_predict = torch.clamp_max(s_predict, 1)
            s_predict = torch.clamp_min(s_predict, 0)

            loss_relation = self.loss_fn(s_predict, s_true)
            loss["loss_relation"] = loss_relation * self.loss_weight["loss_rel"]

            model_info["num_pos_gt"] = int(s_true.sum())

        if target_fea_att is not None:
            assert target_fea_att is not None
            loss_attention_prob = self.loss_fn(prob_att, target_prob_att)
            loss["loss_attention"] = loss_attention_prob * self.loss_weight["loss_att"]

        if self.mi_att is not None:
            fea_mi_att_true = torch.cat([fea_ori, fea_att], dim=1)
            assert target_rel is not None
            fea_ori_false = self.select_false_fea(target_rel, fea_ori)
            fea_mi_att_false = torch.cat([fea_ori_false, fea_att], dim=1)

            Eo = -F.softplus(-self.mi_att(fea_mi_att_true)).mean()
            Ea = -F.softplus(self.mi_att(fea_mi_att_false)).mean()
            loss["loss_mi_att"] = -(Ea+Eo) * self.loss_weight["loss_mi"]

        if self.mi_att_oa is not None:
            oa_mi_att_true = torch.cat([fea_ori, prob_att], dim=1)
            assert target_rel is not None
            oa_ori_false = self.select_false_fea(target_rel, fea_ori)
            oa_mi_att_false = torch.cat([oa_ori_false, prob_att], dim=1)

            Eoa_t = -F.softplus(-self.mi_att_oa(oa_mi_att_true)).mean()
            Eoa_f = -F.softplus(self.mi_att_oa(oa_mi_att_false)).mean()
            loss["loss_mi_oa"] = -(Eoa_f+Eoa_t) * self.loss_weight["loss_mi"]

        if self.mi_att_ao is not None:
            ao_mi_att_true = torch.cat([fea_att, prob_ori], dim=1)
            assert target_rel is not None
            ao_ori_false = self.select_false_fea(target_rel, fea_att)
            ao_mi_att_false = torch.cat([ao_ori_false, prob_ori], dim=1)

            Eao_t = -F.softplus(-self.mi_att_ao(ao_mi_att_true)).mean()
            Eao_f = -F.softplus(self.mi_att_ao(ao_mi_att_false)).mean()
            loss["loss_mi_ao"] = -(Eao_f+Eao_t) * self.loss_weight["loss_mi"]

        return loss, model_info
