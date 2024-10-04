import gc
import math
import torch
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class TermServer(FedavgServer):
    def __init__(self, **kwargs):
        super(TermServer, self).__init__(**kwargs)
        self.evidence = torch.zeros(1)

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

        # calculate tilted loss
        local_losses = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
        new_evidence = local_losses.sub(local_losses.max()).mul(self.args.fair_const).exp().mean()
        self.evidence = self.evidence.mul(0.5).add(new_evidence.mul(0.5))

        tilted_losses = {idx: torch.tensor(losses[idx]['loss']).sub(local_losses.max()).mul(self.args.fair_const) for idx in selected_ids}
        p_new = {idx: loss.exp().div(self.evidence.mul(len(selected_ids))) for (idx, loss) in tilted_losses.items()}

        # update regret
        if self.args.C == 1:
            r_t = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
            l_t = r_t.dot(self.mix_coefs).mul(-1)
            self.cum_server_obj = self.cum_server_obj.add(l_t)
            self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
            self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')

            ## assign next action
            self.mix_coefs = torch.tensor([p_new[idx] for idx in selected_ids])

        ## renormalize mixing coefficients only for selected clients to be used for model aggregation 
        coef_map = {idx: loss.div(sum(p_new.values())) for (idx, loss) in p_new.items()}
        print('term', torch.tensor([losses[idx]['loss'] for idx in selected_ids]), '\n', torch.tensor([tilted_losses[idx] for idx in selected_ids]), '\n', torch.tensor([coef_map[idx] for idx in selected_ids]))

        # aggregate into a new global model & return next decision
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, coef_map) # aggregate local updates
        server_optimizer.step() # update global model with the aggregated update

        # adjust learning rate (step decaying)
        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay ## for client
            self.opt_kwargs['lr'] /= self.args.lr_decay ## for server
            # NOTE: See section E and Corollary 1 & 2 of https://arxiv.org/abs/2003.00295
            
        # logging
        ## calculate KL-divergence from w_t (base coefficients; i.e., static ERM coefficients)
        kl_div = self.kl_div(
            torch.tensor([coef_map[idx] for idx in selected_ids]), 
            w_t[selected_ids].div(w_t[selected_ids].sum())
        ) # http://joschu.net/blog/kl-approx.html
        
        ## KL divergence logging
        self.writer.add_scalars('KL Divergence from Base Coefficients', {'KL Divergence': kl_div.mean().item()}, self.round)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\n\t- KL Divergence from Base Coefficients: {kl_div.mean().item():.2f}')
        return selected_ids