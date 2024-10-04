import torch
import logging
import qpsolvers

import numpy as np

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class FedmgdaServer(FedavgServer):
    def __init__(self, **kwargs):
        super(FedmgdaServer, self).__init__(**kwargs)
        self.opt_kwargs = dict(lr=self.args.server_lr)

    def _aggregate(self, server_optimizer, selected_ids):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # accumulate weights
        flattened_deltas = []
        for identifier in selected_ids:
            local_layers_iterator = self.clients[identifier].upload()
            normalized_and_flattened_delta = server_optimizer.accumulate(local_layers_iterator)
            flattened_deltas.append(normalized_and_flattened_delta)
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return server_optimizer, flattened_deltas

    def update(self):
        """Update the global model through federated learning.
        """
        @torch.no_grad()    
        def _solve_lambda(raw_deltas, eps):
            """Solve quadratic programming to find an optimal lambda.
            """
            def _to_numpy(tensor):
                return tensor.cpu().numpy().astype('double')
            
            deltas = torch.stack(raw_deltas).double()
            N = len(deltas)

            P = torch.matmul(deltas, deltas.T)
            q = torch.zeros(N).double() 
            I = torch.eye(N).double()

            G = torch.cat([-I, -I, I], dim=1).T.double()
            h = torch.cat([
                    torch.zeros(N).double(),
                    torch.ones(N).mul(eps - (1 / N)).double(),
                    torch.ones(N).mul(eps + (1 / N)).double()
                ])
            
            A = torch.ones(1, N).double()
            b = torch.ones(1).double()
            new_lambda = qpsolvers.solve_qp(
                _to_numpy(P), _to_numpy(q), 
                _to_numpy(G), _to_numpy(h), 
                _to_numpy(A), _to_numpy(b),
                solver='ecos'
            )

            # cornercase: if infeasible, use the uniform distribution
            if (new_lambda is None) or np.isnan(new_lambda).any():
                new_lambda = torch.ones(len(deltas)).div(len(deltas))
            else:
                new_lambda = torch.tensor(new_lambda)
            return new_lambda.float()

        #################
        # Client Update #
        #################
        # randomly select clients
        selected_ids = self._sample_clients()

        # request losses on client's training set (to check current global model's generalization performance on local TRAINING set)
        # NOTE: it is for evaluating the quality of actions decided at the server (i.e., utility of Leader's objective - deciding mixing coefficients)
        losses = self._request(selected_ids, eval=True, participated=True, need_feedback=True, save_raw=False) 

        # request update to selected clients (to update the global model parallely with each client's data)
        local_sizes = self._request(selected_ids, eval=False, participated=True, need_feedback=False, save_raw=False) 

        # request evaluation on client's test set (to check current global model's generalization performance on local TEST set)
        # it is for evaluating the quality of the global model
        _ = self._request(selected_ids, eval=True, participated=True, need_feedback=False, save_raw=False)

        #################
        # Server Update #
        #################
        # output base mixing coefficients
        w_obs = {idx: local_sizes[idx] / sum(local_sizes.values()) for idx in selected_ids}
        w_t = torch.zeros(self.args.K)
        w_t[selected_ids] = torch.tensor([w_obs[idx] for idx in selected_ids])

        ## update regret
        if self.args.C == 1:
            r_t = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
            l_t = r_t.dot(self.mix_coefs).mul(-1)
            self.cum_server_obj = self.cum_server_obj.add(l_t)
            self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
            self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')

        # aggregate into a new global model & return next decision
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer, normalized_deltas = self._aggregate(server_optimizer, selected_ids) # aggregate local updates
        p_new = _solve_lambda(normalized_deltas, self.args.fair_const)
        server_optimizer.step(p_new) # update global model with the aggregated update

        if self.args.C == 1:
            self.mix_coefs = p_new.clone()
        print('fedmgda', p_new)

        # adjust learning rate (step decaying)
        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay ## for client
            self.opt_kwargs['lr'] /= self.args.lr_decay ## for server
            # NOTE: See section E and Corollary 1 & 2 of https://arxiv.org/abs/2003.00295

        # logging
        ## calculate KL-divergence from w_t (base coefficients; i.e., static ERM coefficients)
        kl_div = self.kl_div(
            p_new, 
            w_t[selected_ids].div(w_t[selected_ids].sum())
        ) # http://joschu.net/blog/kl-approx.html

        ## KL divergence logging
        self.writer.add_scalars('KL Divergence from Base Coefficients', {'KL Divergence': kl_div.mean().item()}, self.round)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\n\t- KL Divergence from Base Coefficients: {kl_div.mean().item():.2f}')
        return selected_ids