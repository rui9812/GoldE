#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import average_precision_score

from torch.utils.data import DataLoader

from dataloader import TestDataset

import math

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, house_dim, house_num, housd_num, thred, p, q, e_norm, r_norm, s_weight, h_weight, ell_min, ell_max, p_bias, normalize, map,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = int(hidden_dim / house_dim)
        self.house_dim = house_dim
        self.house_num = house_num
        self.epsilon = 2.0
        self.housd_num = housd_num

        self.p = p
        # self.q = q
        self.q = self.house_dim-self.p

        self.thred = thred
        if model_name == 'HousE' or model_name == 'HousE_plus':
            self.house_num = house_num + (2*self.housd_num)
        else:
            self.house_num = house_num

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]), 
            requires_grad=False
        )
        
        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / (self.hidden_dim * (self.house_dim ** 0.5))]),
            requires_grad=False
        )
        
        self.entity_dim = self.hidden_dim
        self.relation_dim = self.hidden_dim
        
        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim, self.house_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

        self.head_bias = nn.Parameter(torch.zeros(nentity))
        nn.init.constant_(
        tensor=self.head_bias,
        val=0.0
        )

        self.tail_bias = nn.Parameter(torch.zeros(nentity))
        nn.init.constant_(
        tensor=self.tail_bias,
        val=0.0
        )

        self.relation_p = nn.Parameter(torch.zeros(nrelation*2, self.relation_dim, self.p*self.p))
        nn.init.uniform_(
            tensor=self.relation_p,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.q != 0:
            self.relation_q = nn.Parameter(torch.zeros(nrelation*2, self.relation_dim, self.q*self.q))
            nn.init.uniform_(
                tensor=self.relation_q,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        self.relation_v = nn.Parameter(torch.zeros(nrelation*2, self.relation_dim, self.p))
        nn.init.constant_(
        tensor=self.relation_v,
        val=0.0
        )
        
        self.relation_bias = nn.Parameter(torch.zeros((nrelation*2, self.relation_dim, self.p*3)))
        nn.init.uniform_(
            tensor=self.relation_bias,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_theta = nn.Parameter(torch.zeros(self.nrelation*2, self.relation_dim, self.q))
        nn.init.uniform_(
            tensor=self.relation_theta,
            a=0.5,
            b=1.0
        )

        self.ell_min = ell_min
        self.ell_max = ell_max
        self.c_clamp = e_norm
        self.r_norm = r_norm
        self.p_bias = p_bias
        self.normalize = normalize
        self.map = map

        self.c = nn.Parameter(torch.zeros(nrelation*2, self.relation_dim, 1))
        nn.init.constant_(
        tensor=self.c,
        val=-1.0
        )

        self.s_weight = nn.Parameter(torch.Tensor([[s_weight]]), requires_grad=False)

        self.h_weight = nn.Parameter(torch.Tensor([[h_weight]]), requires_grad=False)

        self.loss = torch.nn.BCEWithLogitsLoss()

        #Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['GoldE']:
            raise ValueError('model %s not supported' % model_name)
    
    def process_relation(self):
        
        elliptic_theta = torch.clamp(torch.abs(self.relation_theta), min=self.ell_min, max=self.ell_max)

        theta = torch.ones((self.nrelation*2, self.relation_dim, self.p)).cuda()
        minus_one = - torch.ones((self.nrelation*2, self.relation_dim, 1)).cuda()
        theta = torch.cat([theta, minus_one], dim=-1)

        r_p = torch.chunk(self.relation_p, self.p, 2)
        if self.q != 0:
            r_q = torch.chunk(self.relation_q, self.q, 2)

        normed_r_p = []
        normed_r_q = []
        factor_b_list = []
        for i in range(self.p):
            r_i = torch.nn.functional.normalize(r_p[i], dim=2, p=2)
            normed_r_p.append(r_i)
        
        for i in range(self.q):
            b = r_q[i] # (nrelation, relation_dim, house_dim)
            normed_r_q.append(b)
            factor_b = (b*elliptic_theta)
            b_norm =  (b*elliptic_theta*b).sum(dim=2, keepdim=True)
            factor_b = factor_b / b_norm
            factor_b_list.append(factor_b)

        r_p = torch.cat(normed_r_p, dim=2)
        if self.q != 0:
            r_factor = torch.cat(factor_b_list, dim=2)
            r_q = torch.cat(normed_r_q, dim=2)
        else:
            r_factor = None
            r_q = None

        r_v = self.relation_v.reshape(-1, self.p)
        r_v = torch.renorm(r_v, p=2, dim=-2, maxnorm=self.r_norm)
        r_v = r_v.reshape(self.nrelation*2, self.relation_dim, -1)

        return theta, r_p, r_q, r_v, r_factor, elliptic_theta


    def split_pq(self, entity_p, c):
        if self.map == 'exp':
            c = -c
            x_norm = entity_p.norm(dim=-1, keepdim=True)
            x0 = torch.cosh(c.sqrt() * x_norm) * c.sqrt()
            xr = torch.sinh(c.sqrt() * x_norm) * entity_p * c.sqrt() / x_norm
            entity_p = torch.cat([xr, x0], dim=-1) 
        else:
            x0 = torch.sqrt(
            -c + (entity_p * entity_p).sum(dim=-1, keepdim=True)
            )  
            entity_p = torch.cat((entity_p, x0), dim=-1)     
        return entity_p
        

    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        entity_embedding = self.entity_embedding
        beta = torch.clamp(self.c, max=-1 + self.c_clamp, min=-1.5)
        r_theta, r_p, r_q, r_v, r_f, f_theta = self.process_relation()

        if mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)

            hb = torch.index_select(
                self.head_bias,
                dim=0,
                index=head_part.view(-1)
            ).view(batch_size, negative_sample_size)

            relation_v = torch.index_select(
                r_v,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            relation_p = torch.index_select(
                r_p,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            if self.q != 0:
                relation_q = torch.index_select(
                    r_q,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                relation_f = torch.index_select(
                    r_f,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

            else:
                relation_q = None
                relation_f = None

            relation_c = torch.index_select(
                beta,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)
            
            r_bias = torch.index_select(
                self.relation_bias,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            theta = torch.index_select(
                r_theta,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            e_theta = torch.index_select(
                f_theta,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            tb = torch.index_select(
                self.tail_bias,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            hb = torch.index_select(
                self.head_bias,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation_v = torch.index_select(
                r_v,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            relation_p = torch.index_select(
                r_p,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if self.q != 0:
                relation_q = torch.index_select(
                    r_q,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

                relation_f = torch.index_select(
                    r_f,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)
            else:
                relation_q = None
                relation_f = None

            relation_c = torch.index_select(
                beta,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)
            
            r_bias = torch.index_select(
                self.relation_bias,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            theta = torch.index_select(
                r_theta,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            e_theta = torch.index_select(
                f_theta,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                entity_embedding,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size, self.entity_dim, -1)

            tb = torch.index_select(
                self.tail_bias,
                dim=0,
                index=tail_part.view(-1)
            ).view(batch_size, negative_sample_size)

        else:
            raise ValueError('mode %s not supported' % mode)

        if self.model_name == 'GoldE':
            score = self.GoldE(head, hb, relation_v, relation_p, relation_q, relation_f, relation_c, r_bias, theta, e_theta, tail, tb, beta, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score
    
        
    def GoldE(self, head, hb, relation_v, relation_p, relation_q, relation_f, relation_c, r_bias, theta, e_theta, tail, tb, beta, mode):
        r_p = torch.chunk(relation_p, self.p, 3)
        if self.q != 0:
            r_q = torch.chunk(relation_q, self.q, 3)
            r_f = torch.chunk(relation_f, self.q, 3)

        head_p = head[:, :, :, :self.p]
        head_q = head[:, :, :, self.p:]

        tail_p = tail[:, :, :, :self.p]
        tail_q = tail[:, :, :, self.p:]
        
        h_bias, r_bias, t_bias = torch.chunk(r_bias, 3, -1)
        
        head_x = self.split_pq(head_p, relation_c)
        tail_x = self.split_pq(tail_p, relation_c)


        for i in range(self.q):
            head_q = head_q - 2 * (r_f[i] * head_q).sum(dim=-1, keepdim=True) * r_q[i]


        head_p = head_x[:, :, :, :self.p]
        t1 = head_x[:, :, :, self.p:]
        rv2 = (relation_v*relation_v).sum(dim=-1, keepdim=True)
        g = 1 / (1-rv2).sqrt()
        factored_head_q = g * t1 - g * (relation_v * head_p).sum(dim=-1, keepdim=True)
        assert not torch.any(factored_head_q < 0)
        factored_head_p = - g * t1 * relation_v + (head_p + ((g**2)/(1+g))*(relation_v * head_p).sum(dim=-1, keepdim=True)*relation_v)
        
        l_head_p = factored_head_p
        for i in range(self.p):
            l_head_p = l_head_p - 2 * (r_p[i] * l_head_p).sum(dim=-1, keepdim=True) * r_p[i]
        
        if self.p_bias:
            p_norm = torch.norm(l_head_p, p=2, dim=-1, keepdim=True)
            l_head_p = l_head_p + r_bias
            l_head_p = torch.nn.functional.normalize(l_head_p, dim=-1, p=2) * p_norm
        
        l_head = torch.cat([l_head_p, factored_head_q], dim=-1)
        l_tail = tail_x

        if self.q != 0:
            if self.normalize:
                normalized_head_q = F.normalize(head_q, p=2, dim=-1)
                normalized_tail_q = F.normalize(tail_q, p=2, dim=-1)
                spe_distance = torch.acos(torch.clamp((normalized_head_q*normalized_tail_q).sum(dim=-1, keepdim=False), max=1-1e-5, min=-1+1e-5)).mean(dim=-1)
            else:
                spe_distance = ((head_q - tail_q)*(e_theta.sqrt())).norm(dim=3, p=2).sum(dim=-1)
        hyp_distance = torch.acosh(torch.clamp((l_head*theta*l_tail).sum(dim=-1, keepdim=False)/relation_c.squeeze(-1), min=1+1e-5))

        if self.q != 0:
            cos_score = hyp_distance.sum(dim=-1) + self.s_weight.abs() * spe_distance
            assert not torch.any(torch.isnan(spe_distance))
        else:
            cos_score = hyp_distance.sum(dim=-1)
    
        score = self.gamma.item() - (cos_score)
        assert not torch.any(torch.isnan(hyp_distance))
        assert not torch.any(torch.isnan(score))
        return score
    
    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode='tail-batch')
        # print(negative_score)

        if args.negative_adversarial_sampling:
            #In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim = 1).detach() 
                              * F.logsigmoid(-negative_score)).sum(dim = 1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim = 1)

        # if mode == 'head-batch':
        #     pos_part = positive_sample[:, 0].unsqueeze(dim=1)
        # else:
        #     pos_part = positive_sample[:, 2].unsqueeze(dim=1)
        pos_part = positive_sample[:, 2].unsqueeze(dim=1)
        positive_score = model((positive_sample, pos_part), mode='tail-batch')

        positive_score = F.logsigmoid(positive_score).squeeze(dim = 1)

        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum()/subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum()/subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss)/2        
        assert not torch.any(torch.isnan(loss))
        
        if args.regularization != 0.0:
            if args.normalize:
                regularization = args.regularization * (
                    (model.entity_embedding[:, :, :model.p] * model.entity_embedding[:, :, :model.p]).sum(dim=-1).sum(dim=-1).sqrt().mean()
                )
            else:
                regularization = args.regularization * (
                    (model.entity_embedding * model.entity_embedding).sum(dim=-1).sum(dim=-1).sqrt().mean()
                )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}
            
        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log
    
    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''
        
        model.eval()
        
        if args.countries:
            #Countries S* datasets are evaluated on AUC-PR
            #Process test data for AUC-PR evaluation
            sample = list()
            y_true  = list()
            for head, relation, tail in test_triples:
                for candidate_region in args.regions:
                    y_true.append(1 if candidate_region == tail else 0)
                    sample.append((head, relation, candidate_region))

            sample = torch.LongTensor(sample)
            if args.cuda:
                sample = sample.cuda()

            with torch.no_grad():
                y_score = model(sample).squeeze(1).cpu().numpy()

            y_true = np.array(y_true)

            #average_precision_score is the same as auc_pr
            auc_pr = average_precision_score(y_true, y_score)

            metrics = {'auc_pr': auc_pr}
            
        else:
            #Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
            #Prepare dataloader for evaluation
            test_dataloader_head = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    args.self_filter,
                    'head-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )

            test_dataloader_tail = DataLoader(
                TestDataset(
                    test_triples, 
                    all_true_triples, 
                    args.nentity, 
                    args.nrelation, 
                    args.self_filter,
                    'tail-batch'
                ), 
                batch_size=args.test_batch_size,
                num_workers=max(1, args.cpu_num//2), 
                collate_fn=TestDataset.collate_fn
            )
            
            test_dataset_list = [test_dataloader_head, test_dataloader_tail]
            
            logs = []

            step = 0
            total_steps = sum([len(dataset) for dataset in test_dataset_list])

            with torch.no_grad():
                for test_dataset in test_dataset_list:
                    for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                        if args.cuda:
                            positive_sample = positive_sample.cuda()
                            negative_sample = negative_sample.cuda()
                            filter_bias = filter_bias.cuda()

                        batch_size = positive_sample.size(0)

                        score = model((positive_sample, negative_sample), 'tail-batch')
                        score += filter_bias

                        #Explicitly sort all the entities to ensure that there is no test exposure bias
                        argsort = torch.argsort(score, dim = 1, descending=True)

                        # if mode == 'head-batch':
                        #     positive_arg = positive_sample[:, 0]
                        # elif mode == 'tail-batch':
                        #     positive_arg = positive_sample[:, 2]
                        # else:
                        #     raise ValueError('mode %s not supported' % mode)
                        positive_arg = positive_sample[:, 2]

                        for i in range(batch_size):
                            #Notice that argsort is not ranking
                            ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                            assert ranking.size(0) == 1

                            #ranking + 1 is the true ranking used in evaluation metrics
                            ranking = 1 + ranking.item()
                            logs.append({
                                'MRR': 1.0/ranking,
                                'MR': float(ranking),
                                'HITS@1': 1.0 if ranking <= 1 else 0.0,
                                'HITS@3': 1.0 if ranking <= 3 else 0.0,
                                'HITS@10': 1.0 if ranking <= 10 else 0.0,
                            })

                        if step % args.test_log_steps == 0:
                            logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                        step += 1

            metrics = {}
            for metric in logs[0].keys():
                metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        return metrics
