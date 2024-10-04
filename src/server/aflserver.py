import gc
import torch
import logging

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)



class AflServer(FedavgServer):
    def __init__(self, **kwargs):
        super(AflServer, self).__init__(**kwargs)

    def update(self):
        """Update the global model through federated learning.
        """
        def _project(v, s=1):
            assert s > 0, f'Radius {s} must be strictly positive!'
            if (v.sum() == s) and (v >= 0).all():
                return v
            N = len(v)
            u = torch.flip(torch.sort(v)[0], dims=(0, ))
            
            cssv = torch.cumsum(u, dim=0)
            nonzero = torch.nonzero(u.mul(torch.arange(1, N + 1)) > (cssv - s), as_tuple=False)
            if len(nonzero) == 0:
                rho = 0.
            else:
                rho = nonzero[-1].squeeze()
            # compute the Lagrange multiplier associated to the simplex constraint
            theta = cssv[rho].sub(s).div(rho.add(1.))

            # compute the projection by thresholding v using theta
            w = (v.sub(theta)).clamp(min=0)
            return w 

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
        
        # update regret
        if self.args.C == 1:
            r_t = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
            l_t = r_t.dot(self.mix_coefs).mul(-1)
            self.cum_server_obj = self.cum_server_obj.add(l_t)
            self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
            self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')

        # construct new mixing coefficients
        local_losses = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
        p_new = torch.zeros(self.args.K)
        p_new[selected_ids] = self.mix_coefs[selected_ids].add(local_losses.mul(self.args.fair_const))
        p_new[selected_ids] = _project(p_new[selected_ids])

        mask = torch.zeros(self.args.K)
        mask[selected_ids] = 1
        self.mix_coefs = p_new.mul(mask).add(self.mix_coefs.mul(1 - mask))
        coef_map = {idx: self.mix_coefs[idx] for idx in selected_ids}
        print('afl', torch.tensor([losses[idx]['loss'] for idx in selected_ids]), '\n', torch.tensor([coef_map[idx] for idx in selected_ids]))

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
            self.mix_coefs[selected_ids].div(self.mix_coefs[selected_ids].sum()), 
            w_t[selected_ids].div(w_t[selected_ids].sum())
        ) # http://joschu.net/blog/kl-approx.html

        ## KL divergence logging
        self.writer.add_scalars('KL Divergence from Base Coefficients', {'KL Divergence': kl_div.mean().item()}, self.round)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\n\t- KL Divergence from Base Coefficients: {kl_div.mean().item():.2f}')
        return selected_ids