import math
import torch
import logging
import qpsolvers
import numpy as np

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedmdfgServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedmdfgServer, self).__init__(**kwargs)
        self.opt_kwargs = dict(lr=self.args.lr)

    def _solve_qp(self, vec):
        P = np.dot(vec, vec.T)
        n = P.shape[0]
        q = np.zeros(n)
        G = - np.eye(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.ones(1)
        sol = qpsolvers.solve_qp(P, q, G, h, A, b, solver='osqp')
        return sol

    def _solve_mdfg(self, grads, value):
        fair_guidance_vec = torch.ones(len(value)).to(grads.device)
        fair_guidance_vec = fair_guidance_vec / torch.norm(fair_guidance_vec)

        fair_grad = None
        norm_values = value / torch.norm(value)
        
        cos = max(-1, min(1, norm_values @ fair_guidance_vec))
        bias = math.acos(cos) / math.pi * 180

        norm_vec = torch.norm(grads, dim=1)
        indices = list(range(len(norm_vec)))
        vec = norm_vec[indices].reshape(-1, 1) * grads / (norm_vec + 1e-6).reshape(-1, 1)
        if bias <= self.args.fair_const:
            vec = grads
        else:
            h_vec = (fair_guidance_vec @ norm_values * norm_values - fair_guidance_vec).reshape(1, -1)
            h_vec /= torch.norm(h_vec)
            fair_grad = h_vec @ vec
            vec = torch.cat((vec, fair_grad))
        
        sol = self._solve_qp(vec.cpu().detach().numpy())
        sol = torch.from_numpy(sol).float().to(grads.device)
        d = sol @ vec

        weights = torch.ones(len(value)).float().reciprocal().to(d.device)
        g_norm = torch.norm(weights @ grads)
        d = d / torch.norm(d)
        d = d * g_norm
        return d

    def _aggregate(self, server_optimizer, selected_ids):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # accumulate weights
        flattened_grads = []
        for identifier in selected_ids:
            local_grads = []
            local_layers_iterator = self.clients[identifier].upload()
            for idx, (name, local_param) in enumerate(local_layers_iterator):
                if 'num_batches_tracked' in name:
                    continue
                if ((idx == 1) and local_param.dtype == torch.long):
                    continue
                local_grads.append(local_param.grad.data.view(-1))
            flattened_grads.append(torch.cat(local_grads))
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        
        # check elimination condition
        grads = torch.stack(flattened_grads)
        live_idx = torch.where(torch.norm(grads, dim=1) > 1e-6)[0]

        if len(live_idx) == 0:
            return server_optimizer, None, None
        grads = grads[live_idx, :]
        indices = [selected_ids[i] for i in live_idx]
 
        # scale the outliers of all gradient norms
        miu = torch.mean(torch.norm(grads, dim=1))
        grads = grads.div(torch.norm(grads, dim=1, keepdim=True)).mul(miu)
        return server_optimizer, grads, indices

    def update(self):
        """Update the global model through federated learning.
        """
        #################
        # Client Update #
        #################
        # randomly select clients
        selected_ids = self._sample_clients(train=True)
        
        # request losses on client's training set (to check current global model's generalization performance on local TRAINING set)
        # NOTE: it is for evaluating the quality of actions decided at the server (i.e., utility of Leader's objective - deciding mixing coefficients)
        losses = self._request(selected_ids, eval=True, participated=True, need_feedback=True, save_raw=False) 

        # request update to selected clients (to update the global model parallely with each client's data)
        _ = self._request(selected_ids, eval=False, participated=True, need_feedback=True, save_raw=False) 

        # request evaluation on client's test set (to check current global model's generalization performance on local TEST set)
        # it is for evaluating the quality of the global model (i.e., utility of Follower's obejctive - minimizing local losses)
        _ = self._request(selected_ids, eval=True, participated=True, need_feedback=False, save_raw=False)

        #################
        # Server Update #
        #################
        # aggregate into a new global model & return next decision
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer, grads, survived_ids = self._aggregate(server_optimizer, selected_ids) # aggregate local updates
        if survived_ids is None:
            return selected_ids
        
        # solve MDFG direction
        losses_vec = torch.tensor([losses[idx]['loss'] for idx in survived_ids])
        direction = self._solve_mdfg(grads, losses_vec)
        server_optimizer.step(direction) # update global model with the aggregated update

        # adjust learning rate (step decaying)
        if self.round % self.args.lr_decay_step == 0:
            self.opt_kwargs['lr'] *= self.args.lr_decay ## for server
            # NOTE: See section E and Corollary 1 & 2 of https://arxiv.org/abs/2003.00295
        return selected_ids
      