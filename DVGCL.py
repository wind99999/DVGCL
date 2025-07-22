import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
import math
from utility.parser import args

class DVGCL(nn.Module):
    def __init__(self, data_config):
        super(DVGCL, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.plain_adj = data_config['plain_adj']
        self.all_h_list = data_config['all_h_list']
        self.all_t_list = data_config['all_t_list']
        self.A_in_shape = self.plain_adj.tocoo().shape
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda()
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        self.G_indices, self.G_values = self._cal_sparse_adj()

        self.emb_dim = args.embed_size
        self.n_layers = args.n_layers
        self.n_intents = args.n_intents
        self.temp = args.temp
        self.kl_reg = args.kl_reg

        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.int_reg = args.int_reg
        self.ssl_reg = args.ssl_reg

        self.t_size = args.t_size
        self.linear = nn.Linear(self.t_size, self.emb_dim)

        """
        *********************************************************
        Create Model Parameters
        """
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)

        _user_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)

        _item_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        self.eps_weight = torch.nn.Parameter(torch.randn(self.emb_dim, self.emb_dim), requires_grad=True)
        self.eps_bias = torch.nn.Parameter(torch.zeros(self.emb_dim), requires_grad=True)

        """
        *********************************************************
        Initialize Weights
        """
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def _cal_sparse_adj(self):

        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()

        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        D_values = A_tensor.sum(dim=1).pow(-0.5)

        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def inference(self):
        # Graph-based Message Passing
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]
        for i in range(0, self.n_layers):
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            all_embeddings.append(gnn_layer_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return all_embeddings

    def generative_learning(self,emb):
        mean = emb
        std = F.softplus(emb[:, :self.t_size], beta=1)
        std = self.linear(std) + 1e-8
        eps = torch.rand_like(std)  # reparameterization trick
        gen_embeddings = mean + eps * std

        # VGAE
        # mean = emb
        # logstd = torch.matmul(mean, self.eps_weight) + self.eps_bias
        # std = torch.exp(logstd)
        # eps = torch.rand_like(std)
        # gen_embeddings = mean + eps * std

        return gen_embeddings,mean,std

    def kl_loss(self,mean,std):
        kl_loss = -0.5 * (1 + 2 * std - torch.square(mean) - torch.square(torch.exp(std)))
        kl_loss[torch.isinf(kl_loss) | torch.isnan(kl_loss)] = 0
        return kl_loss.sum(1).mean()

    def intent_model(self,all_embeddings):
        u_embeddings, i_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
        u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
        i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
        int_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)
        return int_embeddings

    def cal_ssl_loss(self, users, items, gen_embs,int_embs):
        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), dim=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        u_gen_embs, i_gen_embs = torch.split(gen_embs, [self.n_users, self.n_items], 0)
        u_gen_embs = F.normalize(u_gen_embs[users], dim=1)
        i_gen_embs = F.normalize(i_gen_embs[items], dim=1)

        u_int_embs, i_int_embs = torch.split(int_embs, [self.n_users, self.n_items], 0)
        u_int_embs = F.normalize(u_int_embs[users], dim=1)
        i_int_embs = F.normalize(i_int_embs[items], dim=1)

        cl_loss += cal_loss(u_gen_embs, u_int_embs)
        cl_loss += cal_loss(i_gen_embs, i_int_embs)

        return cl_loss

    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()
        all_embedings = self.inference()
        gen_embedings,mean,std = self.generative_learning(all_embedings)
        int_embedings = self.intent_model(all_embedings)

        # gen_loss
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        kl_loss = self.kl_loss(mean, std)
        kl_loss = self.kl_reg * kl_loss
        gen_loss = bpr_loss + kl_loss

        # emb_loss
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss

        # int_loss
        int_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        int_loss = self.int_reg * int_loss

        # ssl_loss
        cl_loss = self.ssl_reg * self.cal_ssl_loss(users, pos_items, gen_embedings,int_embedings)

        return gen_loss, cl_loss, emb_loss, int_loss



    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings
