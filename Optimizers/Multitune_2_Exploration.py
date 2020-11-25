import os
import copy
import numpy as np
from datetime import datetime as dt
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB
from hpbandster.core.result import Result
from Utility.CONFIG import *
import pandas as pd
import time
from math import ceil


class Multitune_2_Exploration(Master):
    def __init__(self, configspace=None, eta=2, result_logger=None,
                 time_deadline=None, max_trainset_iterations=None, min_bohb_count=5,
                 target_acc=None, acc_saturation_check=True, acc_saturation_delta=0.01,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 7, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):

        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        self.res_dir = 'Results'.join(kwargs['working_directory'].split('Work'))

        cg = None
        bohb_params = {
            "configspace": configspace,
            "min_points_in_model": min_points_in_model,
            "top_n_percent": top_n_percent,
            "num_samples": num_samples,
            "random_fraction": random_fraction,
            "bandwidth_factor": bandwidth_factor,
            "min_bandwidth": min_bandwidth
        }
        self.configspace = configspace

        self.epoch_bohb_cg = CG_BOHB(**bohb_params)
        self.trainset_bohb_cg = CG_BOHB(**bohb_params)
        self.time_bohb_cg = CG_BOHB(**bohb_params)

        self.curr_bohb_cg = None
        self.result_logger = result_logger

        super().__init__(config_generator=cg, **kwargs)

        self.algo_type = None

        # BOHB stuff
        self.eta = eta
        self.min_budget = None
        self.max_budget = None
        self.max_SH_iter = None
        self.budgets = None

        # Worker Interface Parameters
        # epoch-with-increasing-trainset params
        self.trainset_budget = None
        self.epoch_multiplier = 1
        # trainset-with-increasing-epc params
        self.epc = None
        # time-based params
        self.time_multiplier = None
        # Reset trainset after every switch of algo
        self.trainset_reset = True

        # Trackers
        # Format for entry of epc_result_tracker: temp = {'epc': 2, 'best_val_acc': 45}
        self.budget_result_tracker = []
        self.run_details = {}
        self.evaluated_algo = {}
        self.active_test_algo = None
        # self.epoch_trial_sh = None
        # self.trainset_trial_sh = None

        # Termination parameters
        self.time_deadline = time_deadline
        self.experiment_start_time = dt.now()
        self.max_trainset_iterations = max_trainset_iterations
        self.current_trainset_iterations = 0
        self.target_acc = target_acc
        self.best_acc_received = 0
        self.acc_saturation_check = acc_saturation_check
        self.acc_updated = True
        self.acc_saturation_delta = acc_saturation_delta
        self.min_bohb_count = min_bohb_count

        self.config.update({
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

        self.e_min = self.e_max = self.t_min = self.t_max = None

    def register_miniature_results_for_experiment(self, algo_type):
        self.iterations = []
        self.iterations.append(self.get_iteration_miniature(algo_type, 0,
                                                            iteration_kwargs={'result_logger': self.result_logger}))
        # Registering the readings from the miniature evaluation
        hp_pointer = 0
        stage = 0
        for i, j in enumerate(self.evaluated_algo[algo_type]):
            job = j['job_object']
            if i < self.iterations[0].num_configs[0]:
                self.iterations[0].add_configuration(job.kwargs['config'], {'model_based_pick': False})
            elif hp_pointer == self.iterations[0].num_configs[stage]:
                stage += 1
                hp_pointer = 0
                self.iterations[0].process_results()
            hp_pointer += 1
            job.id = (0, job.id[1], job.id[2])
            self.iterations[0].register_result(job, skip_sanity_checks=True)
            self.register_result_for_optim(job.id, job.result)

        self.iterations[0].process_results()

    def set_algo_params(self, algo_type):
        """
        Sets all the parameters required to perform BOHB experiment.
        This is decided based on the first SH iteration of both of the algo types.
        :param algo_type: {'epoch-with-increasing-trainset', 'trainset-with-increasing-epc'}
        :return: None
        """
        # Registering miniature results for iteration 0
        self.register_miniature_results_for_experiment(algo_type)
        NUM_TERMINATIONS = 4

        # Goal: max_budget / min_budget = (eta ** 5)
        # NOTE: None of the aux budgets will be multiplied by 2, since the actual SH starts with iteration 1
        num_configs = len(self.configspace.get_hyperparameters())
        e_min = 2
        e_max = self.e_min * (
                    self.eta ** (NUM_TERMINATIONS - 1))  # 5 is the num_terminations in the blueprint bohb
        e_max = max(self.e_max, (self.e_min * num_configs))
        t_max = 1
        t_min = self.t_max / (self.eta ** (NUM_TERMINATIONS - 1))
        t_min = min(self.t_min, self.t_max / num_configs)


        if algo_type == 'epoch-with-increasing-trainset':
            self.min_budget = e_min
            # 4 since we need a total of 5 iterations [2, 4, 8, 16, 32]
            self.max_budget = e_max
            self.epoch_multiplier = 1
            self.trainset_budget = t_min
            self.epc = None

        elif algo_type == 'trainset-with-increasing-epc':
            self.max_budget = t_max
            self.min_budget = t_min
            self.epc = 2
            self.trainset_budget = None
            self.epoch_multiplier = None

        else:
            self.min_budget = 0.25
            self.max_budget = self.min_budget * (self.eta ** (NUM_TERMINATIONS - 1))
            self.time_multiplier = 0.1
            self.epoch_multiplier = None
            self.trainset_budget = t_min
            self.epc = None

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(self.min_budget / self.max_budget) / np.log(self.eta)) + 1
        self.budgets = self.max_budget * np.power(self.eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

        self.config.update({
            'eta': self.eta,
            'min_budget': self.min_budget,
            'max_budget': self.max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'time_multiplier': self.time_multiplier,
            'epc': self.epc,
            'trainset_budget': self.trainset_budget,
            'epoch_multiplier': self.epoch_multiplier
        })

    def get_next_auxilary_budget(self, current_aux_budget):
        if len(self.budget_result_tracker) < 2:
            return current_aux_budget * 2
        p1 = self.budget_result_tracker[-2]
        p2 = self.budget_result_tracker[-1]
        slope = (p2['best_val_acc'] - p1['best_val_acc']) / (p2['aux_1'] - p1['aux_1'])
        next_epc = p2['aux_1'] * (1 + slope)
        if next_epc == p2['aux_1']:
            return next_epc * 2
        return next_epc

    def get_next_auxilary_budgets(self, current_aux_1, current_aux_2):
        if len(self.budget_result_tracker) < 1:
            return current_aux_1 * 2, current_aux_2 * 2
        elif len(self.budget_result_tracker) < 2:
            p1 = {
                'best_val_acc': 0,
                'aux_1': 0,
                'aux_2': 0
            }
        else:
            p1 = self.budget_result_tracker[-2]
        p2 = self.budget_result_tracker[-1]
        slope_1, slope_2 = 0, 0

        # Need to check for equality, since trainset is always capped at 1.0
        # Even if we double if slope diff is < 0.2, p1 and p2 will remain same if trainset budget reaches 1.0
        if p2['aux_1'] != p1['aux_1']:
            slope_1 = (p2['best_val_acc'] - p1['best_val_acc']) / (p2['aux_1'] - p1['aux_1'])
        if p2['aux_2'] != p1['aux_2']:
            slope_2 = (p2['best_val_acc'] - p1['best_val_acc']) / (p2['aux_2'] - p1['aux_2'])
        next_a1 = p2['aux_1'] * (1 + slope_1)
        next_a2 = p2['aux_2'] * (1 + slope_2)
        if slope_1 <= 0.2:
            next_a1 = p2['aux_1'] * 2
        if slope_2 <= 0.2:
            next_a2 = p2['aux_2'] * 2
        return next_a1, next_a2

    def check_terminal_condition(self):
        """
        Returns true if no more iterations is to be run
        """

        if self.time_deadline is not None:
            if (dt.now() - self.experiment_start_time).total_seconds() >= (self.time_deadline * 60):
                return True

        if self.max_trainset_iterations is not None:
            if self.current_trainset_iterations >= self.max_trainset_iterations:
                return True

        if self.target_acc is not None:
            if self.best_acc_received >= self.target_acc:
                return True

        if self.acc_saturation_check:
            # If acc not updated, stop running
            # return not self.acc_updated
            if not self.acc_updated:
                if self.trainset_budget is None:
                    return True
                elif self.trainset_budget >= 1.0:
                    return True
        return False

    def get_iteration_miniature(self, algo, iteration, iteration_kwargs):
        NUM_TERMINATIONS = 4
        NUM_TERMINATIONS_IN_SH = 2
        num_configs = len(self.configspace.get_hyperparameters())
        self.e_min = 2
        self.e_max = self.e_min * (self.eta ** (NUM_TERMINATIONS - 1))  # 5 is the num_terminations in the blueprint bohb
        self.e_max = max(self.e_max, (self.e_min * num_configs))
        self.t_max = 1
        self.t_min = self.t_max / (self.eta ** (NUM_TERMINATIONS - 1))
        self.t_min = min(self.t_min, self.t_max / num_configs)

        if algo == 'epoch-with-increasing-trainset':
            # self.iterations = []
            min_budget = self.e_min
            max_budget = self.e_max
            self.trainset_budget = self.t_min
            self.epoch_multiplier = 1
            self.epc = None
            cg = self.epoch_bohb_cg

        elif algo == 'trainset-with-increasing-epc':
            min_budget = self.t_min
            max_budget = self.t_max
            self.trainset_budget = None
            self.epoch_multiplier = None
            self.epc = 2
            cg = self.trainset_bohb_cg

        else:
            min_budget = 0.25
            max_budget = min_budget * (self.eta ** (NUM_TERMINATIONS - 1))
            self.trainset_budget = self.t_min
            self.epoch_multiplier = None
            self.time_multiplier = 0.1
            cg = self.time_bohb_cg

        # iteration = 0
        max_sh = -int(np.log(min_budget / max_budget) / np.log(self.eta)) + 1
        # Updating to the global max_SH_iter, since it will be used while update_run_to_optimizer
        # to register old miniature results
        self.max_SH_iter = max_sh
        budgets = max_budget * np.power(self.eta, -np.linspace(max_sh - 1, 0, max_sh))
        s = max_sh - 1 - (0 % max_sh)
        n0 = int(np.floor(max_sh / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(NUM_TERMINATIONS_IN_SH)]
        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns,
                                  budgets=budgets[(-s - 1):(-s - 1 + NUM_TERMINATIONS_IN_SH)],
                                  config_sampler=cg.get_config, **iteration_kwargs))

    def store_miniature_results(self, store_csv=True, store_pkl=False):
        store_res = []
        # methods = ['epoch-with-increasing-trainset',
        #            'trainset-with-increasing-epc', 'time-based']
        methods = ['epoch-with-increasing-trainset',
                   'trainset-with-increasing-epc']
        temp = {'Method': None, 'Budget': None, 'ConfigID': None, 'ValidationAccuracy': None,
                'TestAccuracy': None, 'ValidationConfidence': None, 'TestConfidence': None,
                'EpochMultiplier': None, 'TrainsetBudget': None, 'EPC': None, 'TimeMultiplier': None,
                'TrainsetConsumed': None, 'epochs_for_time_budget': None}
        for m in methods:
            eval_res = self.evaluated_algo[m]
            for row in eval_res:
                temp = {
                    'Method': m,
                    'Budget': row['budget'],
                    'ConfigID': row['job_object'].id,
                    'TFinish': time.strftime('%H:%M:%S', time.localtime(row['tfinish'])),
                    'ValidationAccuracy': row['job_object'].result['info']['validation_accuracy'],
                    'TestAccuracy': row['job_object'].result['info']['test_accuracy'],
                    'ValidationConfidence': row['job_object'].result['info']['validation_confidence'],
                    'TestConfidence': row['job_object'].result['info']['test_confidence'],
                    'EpochMultiplier': row['job_object'].result['info']['epoch_multiplier'],
                    'TrainsetBudget': row['job_object'].result['info']['trainset_budget'],
                    'EPC': row['job_object'].result['info']['epc'],
                    'TimeMultiplier': row['job_object'].result['info']['time_multiplier'],
                    'TrainsetConsumed': row['job_object'].result['info']['trainset_consumed'],
                    'epochs_for_time_budget': row['job_object'].result['info']['epochs_for_time_budget']
                }
                store_res.append(temp)
        df = pd.DataFrame(store_res)

        if store_pkl or store_csv:
            if not os.path.isdir(self.res_dir):
                os.makedirs(self.res_dir)
            if store_csv:
                df.to_csv(self.res_dir + 'miniature.csv')
            if store_pkl:
                df.to_pickle(self.res_dir + 'miniature.pkl')

    def update_curr_bayesian(self, best_method):
        """
        This function updates the Bayesian model to be used for the experiment, post miniature eval.
        Utilizes the results collected throughout the miniature tests
        The general format of translation => new_budget = (a / b) * c
        |-------------------------------------------------------------------------------------------|
        |  Type  1  |  Type  2  |     a           | b                 |     c                       |
        |-------------------------------------------------------------------------------------------|
        |  Epoch    |  Trainset | budget          | e_min             | t_min (trainset_budget)     |
        |-------------------------------------------------------------------------------------------|
        |  Epoch    |  Time     | budget          | epochs_per_minute | 1                           |
        |-------------------------------------------------------------------------------------------|
        |  Trainset |  Epoch    | budget          | t_min             | e_min (epc)                 |
        |-------------------------------------------------------------------------------------------|
        |  Trainset |  Time     | budget          | t_min             | (e_min * epochs_per_minute) |
        |-------------------------------------------------------------------------------------------|
        |  Time     |  Epoch    | epochs_obtained | 1                 | 1                           |
        |-------------------------------------------------------------------------------------------|
        |  Time     |  Trainset | epochs_obtained | e_min             | t_min                       |
        |-------------------------------------------------------------------------------------------|
        """
        """
        These parameters are only present for time based
        # The last item of time-based will have run for 0.25 minutes. Multiply the epochs_for_time_budget by 4
        last_time_job = self.evaluated_algo['time-based'][-1]['job_object']
        epochs_per_minute = last_time_job.result['info']['epochs_for_time_budget']
        epochs_per_minute *= 4
        epochs_per_minute = round(epochs_per_minute)
        """
        epochs_per_minute = 1

        a_b_c_dict = {
            'epoch-with-increasing-trainset': {
                'trainset-with-increasing-epc': {
                    'a': 'budget',
                    'b': self.e_min,
                    'c': self.t_min
                },
                'time-based': {
                    'a': 'budget',
                    'b': epochs_per_minute,
                    'c': 1
                }
            },
            'trainset-with-increasing-epc': {
                'epoch-with-increasing-trainset': {
                    'a': 'budget',
                    'b': self.t_min,
                    'c': self.e_min
                },
                'time-based': {
                    'a': 'budget',
                    'b': self.t_min,
                    'c': self.e_min * epochs_per_minute
                }
            },
            'time-based': {
                'epoch-with-increasing-trainset': {
                    'a': epochs_per_minute,
                    'b': 1,
                    'c': 1
                },
                'trainset-with-increasing-epc': {
                    'a': epochs_per_minute,
                    'b': self.e_min,
                    'c': self.t_min
                }
            }
        }

        for method in self.evaluated_algo:
            if method == best_method:
                continue
            temp = a_b_c_dict[method][best_method]
            for j in self.evaluated_algo[method]:
                job = j['job_object']
                a = job.kwargs[temp['a']] if isinstance(temp['a'], str) else temp['a']
                b = temp['b']
                c = temp['c']
                new_budget = (a / b) * c
                job.kwargs['budget'] = new_budget
                self.curr_bohb_cg.new_result(job)

    def choose_experiment_type_and_setup(self):
        self.active_test_algo = None
        info = []

        # acc_epoch_with_increasing_trainset = self.evaluated_algo['epoch-with-increasing-trainset'][-1]['acc']
        # methods = ['epoch-with-increasing-trainset', 'trainset-with-increasing-epc', 'time-based']
        methods = ['epoch-with-increasing-trainset', 'trainset-with-increasing-epc']
        for method in methods:
            m = max(self.evaluated_algo[method], key=lambda x: x['acc'])
            start_time = self.evaluated_algo[method][0]['tstart']
            end_time = m['tfinish']
            temp = {
                'method': method,
                'acc': m['acc'],
                'dur': end_time - start_time
            }
            info.append(temp)

        info = sorted(info, key=lambda x: x['acc'], reverse=True)
        """
        Don't pop here, since it is 2d only. 
        """
        # info.pop()  # Remove the worst performing method
        acc_delta = info[0]['acc'] - info[1]['acc']
        time_delta = info[0]['dur'] - info[1]['dur']
        best_method = info[0]['method']

        ACC_DIFF_THRESH = 0.03
        TIME_DIFF_THRESH = 30  # seconds

        if acc_delta > ACC_DIFF_THRESH:
            self.algo_type = info[0]['method']
        else:
            if abs(time_delta) > TIME_DIFF_THRESH:
                info = sorted(info, key=lambda x: x['dur'], reverse=True)
                self.algo_type = info[0]['method']  # 0 index here is not necessarily same as the else part
                # 0 in the above line denotes the fastest experiment
            else:
                self.algo_type = info[0]['method']  # 0 here denotes the miniature with best acc

        if best_method == 'epoch-with-increasing-trainset':
            self.curr_bohb_cg = self.epoch_bohb_cg
        elif best_method == 'trainset-with-increasing-epc':
            self.curr_bohb_cg = self.trainset_bohb_cg
        else:
            self.curr_bohb_cg = self.time_bohb_cg

        self.update_curr_bayesian(best_method)
        self.set_algo_params(self.algo_type)
        self.trainset_reset = True
        self.store_miniature_results()

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        if self.algo_type is None:
            # if 'time-based' not in self.evaluated_algo:
            #     self.trainset_reset = True
            #     self.active_test_algo = 'time-based'
            #     time_sh = self.get_iteration_miniature('time-based', iteration, iteration_kwargs)
            #     return time_sh

            if 'epoch-with-increasing-trainset' not in self.evaluated_algo:
                self.trainset_reset = True
                self.active_test_algo = 'epoch-with-increasing-trainset'
                epoch_trial_sh = self.get_iteration_miniature('epoch-with-increasing-trainset',
                                                              iteration, iteration_kwargs)
                return epoch_trial_sh

            elif 'trainset-with-increasing-epc' not in self.evaluated_algo:
                self.trainset_reset = True
                self.active_test_algo = 'trainset-with-increasing-epc'
                trainset_trial_sh = self.get_iteration_miniature('trainset-with-increasing-epc',
                                                                 iteration, iteration_kwargs)
                return trainset_trial_sh


            else:
                # If sufficient data has been collected to define the experiment type, setup params
                # Trainset will be reset inside the choose_experiment_type_and_setup function
                self.choose_experiment_type_and_setup()
                iteration = 1  # 0th iteration is being used from the miniature

        if len(self.run_details) > 1 and self.check_terminal_condition():
            return None

        if iteration % self.max_SH_iter == 0:
            self.update_aux_budget_tracker(iteration)
            if self.algo_type in EPOCH_BASED_METHODS:
                trainset_budget, epoch_multiplier = self.get_next_auxilary_budgets(self.trainset_budget,
                                                                                   self.epoch_multiplier)
                self.trainset_budget = min(1, trainset_budget)
                self.epoch_multiplier = epoch_multiplier

            elif self.algo_type == 'trainset-with-increasing-epc':
                self.epc = ceil(self.get_next_auxilary_budget(self.epc))

            else:
                trainset_budget, time_multiplier = self.get_next_auxilary_budgets(self.trainset_budget,
                                                                                   self.time_multiplier)
                self.time_multiplier = time_multiplier
                self.trainset_budget = min(1, trainset_budget)

        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        cg = self.curr_bohb_cg

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s - 1):],
                                  config_sampler=cg.get_config, **iteration_kwargs))

    def run(self, n_iterations=np.inf, min_n_workers=1, iteration_kwargs={}, ):
        self.wait_for_workers(min_n_workers)

        iteration_kwargs.update({'result_logger': self.result_logger})

        if self.time_ref is None:
            self.time_ref = time.time()
            self.config['time_ref'] = self.time_ref

            self.logger.info('HBMASTER: starting run at %s' % (str(self.time_ref)))

        self.thread_cond.acquire()
        while True:

            self._queue_wait()

            next_run = None
            # find a new run to schedule
            for i in self.active_iterations():
                next_run = self.iterations[i].get_next_run()
                if not next_run is None: break

            if not next_run is None:
                self.logger.debug('HBMASTER: schedule new run for iteration %i' % i)
                self._submit_job(*next_run)
                continue
            else:
                if n_iterations > 0:  # we might be able to start the next iteration
                    next_iteration = self.get_next_iteration(len(self.iterations), iteration_kwargs)
                    if next_iteration is not None:
                        self.iterations.append(next_iteration)
                        n_iterations -= 1
                        continue
                    else:
                        pass

            # at this point there is no imediate run that can be scheduled,
            # so wait for some job to finish if there are active iterations
            if self.active_iterations():
                self.thread_cond.wait()
            else:
                break

        self.thread_cond.release()

        for i in self.warmstart_iteration:
            i.fix_timestamps(self.time_ref)

        ws_data = [i.data for i in self.warmstart_iteration]

        return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)

    def register_result_for_optim(self, config_id, result):
        if config_id[0] not in self.run_details:
            self.run_details[config_id[0]] = {}
        self.run_details[config_id[0]][config_id] = {
            'val_acc': 1 - result['loss'],
            'info': result['info']
        }

    def update_aux_budget_tracker(self, iteration):
        if len(self.run_details) == 0:
            return False
        upper = iteration    # exclusive
        lower = iteration - self.max_SH_iter    # inclusive
        best_acc_in_current_bohb = self._find_best_acc_in_bohb_iter(range(lower, upper))
        if best_acc_in_current_bohb > self.best_acc_received + self.acc_saturation_delta:
            self.best_acc_received = best_acc_in_current_bohb
            self.acc_updated = True
        else:
            self.acc_updated = False

        if self.active_test_algo in EPOCH_BASED_METHODS or self.algo_type in EPOCH_BASED_METHODS:
            trainset_budget = self.trainset_budget
            epoch_multiplier = self.epoch_multiplier
            aux_budget = trainset_budget if trainset_budget < 1.0 else epoch_multiplier
            aux_1 = trainset_budget
            aux_2 = epoch_multiplier

        elif self.active_test_algo == 'trainset-with-increasing-epc' or \
                self.algo_type == 'trainset-with-increasing-epc':
            aux_budget = self.epc
            aux_1 = aux_budget
            aux_2 = None

        else:
            trainset_budget = self.trainset_budget
            time_multiplier = self.time_multiplier
            aux_budget = trainset_budget if trainset_budget < 1.0 else time_multiplier
            aux_1 = trainset_budget
            aux_2 = time_multiplier

        temp = {'aux_1': aux_1, 'aux_2': aux_2, 'best_val_acc': best_acc_in_current_bohb}
        self.budget_result_tracker.append(temp)

        return True

    def _find_best_acc_in_bohb_iter(self, sh_iteration_ids):
        best = 0
        for i in sh_iteration_ids:
            for configs in self.run_details[i]:
                if self.run_details[i][configs]['val_acc'] > best:
                    best = self.run_details[i][configs]['val_acc']
        return best

    def log_pre_tune_results(self, job):
        if self.active_test_algo not in self.evaluated_algo:
            self.evaluated_algo[self.active_test_algo] = []
        temp = {
            'budget': job.kwargs['budget'],
            'acc': 1 - job.result['loss'],
            'tstart': job.timestamps['started'],
            'tfinish': job.timestamps['finished'],
            'job_object': job
        }
        self.evaluated_algo[self.active_test_algo].append(temp)

    def job_callback(self, job):
        if self.algo_type is None:
            self.log_pre_tune_results(job)
            # return

        elif job.result is not None:
            self.register_result_for_optim(job.id, job.result)

        # super().job_callback(job)
        self.logger.debug('job_callback for %s started' % str(job.id))
        with self.thread_cond:
            self.logger.debug('job_callback for %s got condition' % str(job.id))
            self.num_running_jobs -= 1

            if self.result_logger is not None and self.algo_type is not None:
                self.result_logger(job)

            # if self.algo_type is not None:
            #     self.config_generator.new_result(job)
            if self.algo_type is None:
                if self.active_test_algo in EPOCH_BASED_METHODS:
                    self.epoch_bohb_cg.new_result(job)
                elif self.active_test_algo == 'trainset-with-increasing-epc':
                    self.trainset_bohb_cg.new_result(job)
                else:
                    self.time_bohb_cg.new_result(job)
            elif self.curr_bohb_cg is not None:
                self.curr_bohb_cg.new_result(job)
            self.iterations[job.id[0]].register_result(job)

            if self.num_running_jobs <= self.job_queue_sizes[0]:
                self.logger.debug("HBMASTER: Trying to run another job!")
                self.thread_cond.notify()

        self.logger.debug('job_callback for %s finished' % str(job.id))
