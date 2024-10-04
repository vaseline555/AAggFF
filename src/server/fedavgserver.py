import os
import json
import torch
import random
import logging
import concurrent.futures

import numpy as np

from importlib import import_module
from collections import ChainMap, defaultdict

from src import init_weights, TqdmToLogger, MetricManager
from .baseserver import BaseServer

logger = logging.getLogger(__name__)



class FedavgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer
        
        # default FL configs
        self.round = 0 # round indicator
        if self.args.eval_type != 'local': # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # global model
        self.opt_kwargs = dict(lr=self.args.server_lr, momentum=self.args.beta1) # federation algorithm arguments
        self.curr_lr = self.args.lr # learning rate
        self.clients = self._create_clients(client_datasets) # clients container
        self.results = defaultdict(dict) # logging results container

        # online learning configs
        ## reward containers
        self.cum_server_obj = torch.zeros(1)
        self.cum_global_obj = torch.zeros(1)
        
        ## online decision making of the server
        self.mix_coefs = torch.ones(self.args.K).mul(self.args.C) # mixing coefficients for each client

        ## measure kl divergence
        self.kl_div = lambda p, q: torch.nan_to_num((p.log().sub(q.log()).exp().sub(1)).sub(p.log().sub(q.log())), posinf=0, neginf=0)

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        optimizer = ALGORITHM_CLASS(params=model.parameters(), **kwargs)
        if self.args.algorithm != 'fedsgd': 
            optimizer.add_param_group(dict(params=list(self.global_model.buffers()))) # add buffered tensors (i.e., gamma and beta of batchnorm layers)
        return optimizer

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() + 4)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)
                ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    def _sample_clients(self, exclude_ids=[], train=True):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')
        if train: # Update - randomly select max(floor(C * K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample([i for i in range(self.args.K)], num_sampled_clients))
        else: # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            if exclude_ids == []:
                num_sampled_clients = self.args.K
                sampled_client_ids = sorted([i for i in range(self.args.K)])
            else:
                num_unparticipated_clients = self.args.K - len(exclude_ids)
                if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                    num_sampled_clients = self.args.K
                    sampled_client_ids = sorted([i for i in range(self.args.K)])
                else:
                    num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                    sampled_client_ids = sorted(random.sample([identifier for identifier in [i for i in range(self.args.K)] if identifier not in exclude_ids], num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.info(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()
        num = max(1, int(len(losses_array) * 0.1))
        top10_indices = np.argpartition(losses_array, -num)[-num:]
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean = top10.dot(np.atleast_1d(num_samples[top10_indices]) / num_samples[top10_indices].sum()) 
        top10_std = np.sqrt(np.power(top10 - top10_mean, 2).dot(np.atleast_1d(num_samples[top10_indices]) / num_samples[top10_indices].sum()))

        bot10_indices = np.argpartition(losses_array, num)[:num]
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean = bot10.dot(np.atleast_1d(num_samples[bot10_indices]) / num_samples[bot10_indices].sum())
        bot10_std = np.sqrt(np.power(bot10 - bot10_mean, 2).dot(np.atleast_1d(num_samples[bot10_indices]) / num_samples[bot10_indices].sum()))

        losses_sorted = np.sort(losses_array) 
        indices = np.arange(1, len(losses_sorted) + 1)
        gini = ((np.sum((2 * indices - len(losses_sorted)  - 1) * losses_sorted)) / (len(losses_sorted) * np.sum(losses_sorted)))
        gini = np.nan_to_num(gini, nan=1.0)
        snr = np.nan_to_num(weighted / std) if std > 0 else np.inf

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f}) | SNR ({snr:.4f}) | Gini ({gini:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float), 'snr': snr, 'gini': gini,
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss' + eval * f' ({"In" if participated else "Out"})',
            {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
            self.round
        )

        self.writer.add_scalars(
            f'Client-level Fairness for Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
            {'SNR': np.nan_to_num(weighted / std), 'Gini': gini},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()
            num = max(1, int(len(val_array) * 0.1))
            top10_indices = np.argpartition(val_array, -num)[-num:]
            top10 = np.atleast_1d(val_array[top10_indices])
            top10_mean = top10.dot(np.atleast_1d(num_samples[top10_indices]) / num_samples[top10_indices].sum())
            top10_std = np.sqrt(np.power(top10 - top10_mean, 2).dot(np.atleast_1d(num_samples[top10_indices]) / num_samples[top10_indices].sum()))

            bot10_indices = np.argpartition(val_array, num)[:num]
            bot10 = np.atleast_1d(val_array[bot10_indices])
            bot10_mean = bot10.dot(np.atleast_1d(num_samples[bot10_indices]) / num_samples[bot10_indices].sum())
            bot10_std = np.sqrt(np.power(bot10 - bot10_mean, 2).dot(np.atleast_1d(num_samples[bot10_indices]) / num_samples[bot10_indices].sum()))

            val_sorted = np.sort(val_array) 
            indices = np.arange(1, len(val_sorted) + 1)
            gini = ((np.sum((2 * indices - len(val_sorted)  - 1) * val_sorted)) / (len(val_sorted) * np.sum(val_sorted)))
            gini = np.nan_to_num(gini, nan=1.0)
            snr = np.nan_to_num(weighted / std) if std > 0 else np.inf

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f}) | SNR ({snr:.4f}) | Gini ({gini:.4f})'
            result_dict[name] = {
                'avg': weighted.astype(float), 'std': std.astype(float), 'snr': snr, 'gini': gini,
                'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
                'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
            }
                
            if save_raw:
                result_dict[name]['raw'] = val

            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
                self.round
            )
            
            self.writer.add_scalars(
                f'Client-level Fairness for Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'SNR': np.nan_to_num(weighted / std), 'Gini': gini},
                self.round
            )

        # log total message
        self.writer.flush()
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval, participated, need_feedback, save_raw):
        def __update_clients(client):
            client.download(self.global_model)
            client.args.lr = self.curr_lr
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client, participated, need_feedback):
            if client.model is None:
                client.download(self.global_model)
            eval_result = client.evaluate(need_feedback=need_feedback) 
            if not participated:
                client.model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation" if not need_feedback else "losses"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args.train_only: return
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() + 4) if self.args.max_workers == -1 else self.args.max_workers) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx], participated, need_feedback))
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            _eval_sizes, eval_results = list(map(list, zip(*results)))
            _eval_sizes, eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*eval_results))
            if not need_feedback:
                self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                    _eval_sizes, 
                    eval_results, 
                    eval=True, 
                    participated=participated,
                    save_raw=save_raw
                )
                logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
            return eval_results
        else:
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), os.cpu_count() + 4) if self.args.max_workers == -1 else self.args.max_workers) as workhorse:
                for idx in TqdmToLogger(
                    ids, 
                    logger=logger, 
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__update_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            update_sizes, _update_results = list(map(list, zip(*results)))
            update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                _update_results, 
                eval=False, 
                participated=True,
                save_raw=False
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
            
    def _aggregate(self, server_optimizer, coef_map):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')

        # accumulate weights
        for identifier in coef_map.keys():
            local_layers_iterator = self.clients[identifier].upload()
            server_optimizer.accumulate(coef_map[identifier], local_layers_iterator)
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return server_optimizer

    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.to(self.args.device)

        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)

            outputs = self.global_model(inputs)
            loss = torch.nn.__dict__[self.args.criterion]()(outputs, targets)

            mm.track(loss.item(), outputs.detach().cpu(), targets.detach().cpu())
        else:
            self.global_model.to('cpu')
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result

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

        # update regret
        if self.args.C == 1:
            r_t = torch.tensor([losses[idx]['loss'] for idx in selected_ids])
            l_t = r_t.dot(w_t[selected_ids]).mul(-1)
            self.cum_server_obj = self.cum_server_obj.add(l_t)
            self.results[self.round]['global_obj'] = {'global_obj': l_t.item()}
            self.writer.add_scalars('Cumulative Global Objective Value', {'Cumulative Server Objective Value': self.cum_server_obj.item()}, self.round)
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [SERVER]\t- Cumulative Server Objective Value: {self.cum_server_obj.item():.4f}')

            ## assign next action
            self.mix_coefs = w_t.clone()

        # renormalize mixing coefficients only for selected clients to be used for model aggregation 
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

    def evaluate(self, exclude_ids):
        """Evaluate the global model located at the server.
        """
        ##############
        # Evaluation #
        ##############
        if self.args.eval_type != 'global': # `local` or `both`: evaluate on selected clients' holdout set
            selected_ids = self._sample_clients(exclude_ids=exclude_ids, train=False)
            _ = self._request(selected_ids, eval=True, participated=False, need_feedback=False, save_raw=self.round == self.args.R)
        if self.args.eval_type != 'local': # `global` or `both`: evaluate on the server's global holdout set 
            self._central_evaluate()

        # calculate generalization gap
        if (not self.args.train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if 'avg' in name:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: # save results
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        torch.save(self.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')) # save model checkpoint
        self.writer.close()
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
        if self.args.use_tb:
            input('[FINISH] ...press <Enter> to exit after tidying up your TensorBoard logging!')
