import gc
import math
import torch
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class QfedavgServer(FedavgServer):
    def __init__(self, **kwargs):
        super(QfedavgServer, self).__init__(**kwargs)
        self.opt_kwargs = dict(
            lipschitz=self.args.server_lr / self.args.lr,
            lr=self.args.server_lr,
            q=self.args.fair_const
        )

    def _aggregate(self, server_optimizer, selected_ids, global_model_losses):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # accumulate weights
        for identifier in selected_ids:
            local_layers_iterator = self.clients[identifier].upload()
            server_optimizer.accumulate(local_layers_iterator, global_model_losses[identifier]['loss'])
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return server_optimizer

    def update(self):
        """Update the global model through federated learning.
        """
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
        # it is for evaluating the quality of the global model (i.e., utility of Follower's obejctive - minimizing local losses)
        _ = self._request(selected_ids, eval=True, participated=True, need_feedback=False, save_raw=False)

        #################
        # Server Update #
        #################
        # output base mixing coefficients
        w_obs = {idx: local_sizes[idx] / sum(local_sizes.values()) for idx in selected_ids}
        w_t = torch.zeros(self.args.K)
        w_t[selected_ids] = torch.tensor([w_obs[idx] for idx in selected_ids])

        # aggregate into a new global model & return next decision
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, selected_ids, losses) # aggregate local updates
        p_new = server_optimizer.step() # update global model with the aggregated update
        p_new.div_(p_new.sum()) # normalize to have sum equal to 1

        # update regret
        if self.args.C == 1:
            r_t = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
            l_t = r_t.dot(self.mix_coefs).mul(-1)
            self.cum_server_obj = self.cum_server_obj.add(l_t)
            self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
            self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')

            ## assign next action
            self.mix_coefs = p_new.clone()
        print('qfedavg', torch.tensor([losses[idx]['loss'] for idx in selected_ids]), '\n', w_t[selected_ids].div(w_t[selected_ids].sum()), '\n', p_new)

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