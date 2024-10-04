import math
import torch
import logging
import cvxpy as cp

from .fedadagradserver import FedadagradServer

logger = logging.getLogger(__name__)



class AaggffadagradServer(FedadagradServer):
    def __init__(self, **kwargs):
        super(AaggffadagradServer, self).__init__(**kwargs)
        # grdaient container for the central learner
        if self.args.C == 1: # cross-silo setting - ONS
            lipschitz = 1 / self.args.K
            self.alpha = 4 * self.args.K * lipschitz
            self.beta = 1 / (4 * lipschitz)

            self.grad_hist = torch.zeros(self.args.K)
            self.hess_hist = torch.eye(self.args.K).mul(self.alpha)
        else: # cross-device setting - Linearized FTRL
            self.grad_hist = torch.zeros(self.args.K)

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

        # output local responses 
        ## (i.e., local losses on a global model BEFORE local update)
        r_t = torch.zeros(self.args.K)
        r_t[selected_ids] = torch.tensor([losses[idx]['loss'] for idx in selected_ids])

        ## update regret
        l_t = r_t.dot(self.mix_coefs)

        ## log regret
        self.cum_server_obj = self.cum_server_obj.add(l_t)
        self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
        self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')
        
        ## transform range of return vectors to be bounded using a CDF: [0, 1]
        if self.args.fair_const == 0: # Weibull CDF (shape=2, scale=1; shape > 1 where outliers increased through training)
            r_t[selected_ids] = (r_t[selected_ids].div(r_t[selected_ids].mean())).pow(2).mul(-1).exp().mul(-1).add(1)
        elif self.args.fair_const == 1: # Exponential CDF (i.e., Weibull CDF with shape=1, scale=1; where outliers are randomly occurred)
            r_t[selected_ids] = (r_t[selected_ids].div(r_t[selected_ids].mean())).mul(-1).exp().mul(-1).add(1)
        elif self.args.fair_const == 2: # Frechet CDF (shape=1, scale=1, m=0; for modeling Pareto RV's maximum)
            r_t[selected_ids] = (r_t[selected_ids].div(r_t[selected_ids].mean())).reciprocal().mul(-1).exp()
        elif self.args.fair_const == 3: # Normal CDF (mean=1, std=1)
            r_t[selected_ids] = torch.erfc((r_t[selected_ids].div(r_t[selected_ids].mean())).sub(1).div(2**0.5))
        elif self.args.fair_const == 4: # Gumbel CDF (location=1, scale=1)
            r_t[selected_ids] = (r_t[selected_ids].div(r_t[selected_ids].mean())).sub(1).mul(-1).exp().mul(-1).exp()
        elif self.args.fair_const == 5: # Logistic CDF (location=1, scale=1)
            r_t[selected_ids] = (r_t[selected_ids].div(r_t[selected_ids].mean())).sub(1).mul(-1).exp().add(1).reciprocal()
        
        ## change bound to achieve O(1) Lipschitz continuity
        if self.args.C < 1:
            r_t[selected_ids] = r_t[selected_ids].mul(self.args.C)
        else:
            r_t[selected_ids] = r_t[selected_ids].div(self.args.K)

        ## decide next decision
        if self.args.C == 1: ### cross-silo setting
            #### return gradient
            grad = -r_t.div(self.mix_coefs.dot(r_t).add(1))
            hess = grad.outer(grad)

            #### accumulate Hessian estimates
            self.grad_hist += grad
            self.hess_hist += hess

            #### return next decision
            p_var = cp.Variable(self.args.K)
            obj = self.grad_hist @ p_var +  0.5 * self.alpha * cp.sum_squares(p_var) + 0.5 * self.beta * cp.quad_form(p_var,  self.hess_hist)
            _ = cp.Problem(cp.Minimize(obj), [cp.sum(p_var) == 1, p_var >= 0]).solve()
            p_new = torch.tensor(p_var.value).float()
        else: ### cross-device setting
            r_dr_t = torch.ones(self.args.K).mul(r_t[selected_ids].mean())
            r_dr_t[selected_ids] = (r_t[selected_ids].sub(r_dr_t[selected_ids])).div(self.args.C) # doubly-roubst estimator

            #### return linearized gradient
            denominator = self.mix_coefs.sum().mul(r_t[selected_ids].mean()).add(1)
            first_term = -r_dr_t.div(denominator)
            second_term = torch.ones(self.args.K).view(-1, 1).matmul(self.mix_coefs.view(1, -1)).mul(r_t[selected_ids].mean()).matmul(
                (r_dr_t.sub(torch.ones(self.args.K).mul(r_t[selected_ids].mean()))).view(-1, 1)
                ).div(denominator.pow(2)).squeeze()
            grad = first_term.add(second_term)

            #### accumulate gradient estimates
            self.grad_hist += grad

            #### return next decision
            neg_sum_grads = -self.grad_hist
            grads_mul_beta = neg_sum_grads.mul(math.sqrt(math.log(self.args.K))).div(math.sqrt(self.round + 1)).div(2+ self.args.C)
            p_new = grads_mul_beta.sub(grads_mul_beta.max()).exp()
            p_new = p_new.div(p_new.sum())
        
        ## assign next action
        self.mix_coefs = p_new.clone()
        print(
            'adaptive', self.args.fair_const, 
            torch.tensor([losses[idx]['loss'] for idx in selected_ids]), 
            r_t[selected_ids], '\n', 
            grad[selected_ids], '\n', 
            self.mix_coefs[selected_ids].div(self.mix_coefs[selected_ids].sum())
        )

        ## renormalize mixing coefficients only for selected clients to be used for model aggregation 
        coef_map = {idx: self.mix_coefs[idx].div(self.mix_coefs[selected_ids].sum()) for idx in selected_ids}

        # aggregate into a new global model
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, coef_map) # aggregate local updates
        server_optimizer.step() # update global model with the aggregated update

        # adjust learning rate (exponential decay)
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
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- KL Divergence from Base Coefficients: {kl_div.mean().item():.2f}')
        return selected_ids