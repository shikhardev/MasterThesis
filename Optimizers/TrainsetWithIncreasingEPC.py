import time
import copy
import numpy as np
from datetime import datetime as dt
from hpbandster.core.master import Master
from hpbandster.optimizers.iterations import SuccessiveHalving
from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB
from hpbandster.core.result import Result
from math import ceil


class TrainsetWithIncreasingEPC(Master):
    def __init__(self, configspace, eta=2,
                 time_deadline=None, max_trainset_iterations=None, min_epochs_per_config=2, min_bohb_count=5,
                 target_acc=None, acc_saturation_check=True, acc_saturation_delta=0.01,
                 min_points_in_model=None, top_n_percent=15,
                 num_samples=64, random_fraction=1 / 7, bandwidth_factor=3,
                 min_bandwidth=1e-3,
                 **kwargs):
        """
                BOHB performs robust and efficient hyperparameter optimization
                at scale by combining the speed of Hyperband searches with the
                guidance and guarantees of convergence of Bayesian
                Optimization. Instead of sampling new configurations at random,
                BOHB uses kernel density estimators to select promising candidates.

                .. highlight:: none

                For reference: ::

            @InProceedings{falkner-icml-18,
              title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
              author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
              booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
              pages =        {1436--1445},
              year =         {2018},
            }

        Parameters
        ----------
        configspace: ConfigSpace object
            valid representation of the search space
        eta : float
            In each iteration, a complete run of sequential halving is executed. In it,
            after evaluating each configuration on the same subset size, only a fraction of
            1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        min_budget : float
            The smallest budget to consider. Needs to be positive!
        max_budget : float
            The largest budget to consider. Needs to be larger than min_budget!
            The budgets will be geometrically distributed
                        :math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
        min_points_in_model: int
            number of observations to start building a KDE. Default 'None' means
            dim+1, the bare minimum.
        top_n_percent: int
            percentage ( between 1 and 99, default 15) of the observations that are considered good.
        num_samples: int
            number of samples to optimize EI (default 64)
        random_fraction: float
            fraction of purely random configurations that are sampled from the
            prior without the model.
        bandwidth_factor: float
            to encourage diversity, the points proposed to optimize EI, are sampled
            from a 'widened' KDE where the bandwidth is multiplied by this factor (default: 3)
        min_bandwidth: float
            to keep diversity, even when all (good) samples have the same value for one of the parameters,
            a minimum bandwidth (Default: 1e-3) is used instead of zero.
        iteration_kwargs: dict
            kwargs to be added to the instantiation of each iteration
        """

        if configspace is None:
            raise ValueError("You have to provide a valid CofigSpace object")

        # Worker Interface Parameters
        # epoch-with-increasing-trainset params
        self.trainset_budget = None
        self.epoch_multiplier = 1
        # trainset-with-increasing-epc params
        self.epc = None
        # Algo Type
        self.algo_type = 'trainset-with-increasing-epc'
        self.active_test_algo = None

        cg = CG_BOHB(configspace=configspace,
                     min_points_in_model=min_points_in_model,
                     top_n_percent=top_n_percent,
                     num_samples=num_samples,
                     random_fraction=random_fraction,
                     bandwidth_factor=bandwidth_factor,
                     min_bandwidth=min_bandwidth
                     )

        super().__init__(config_generator=cg, **kwargs)

        max_budget = 1
        NUM_ITERATIONS = 4
        min_budget = max_budget / (eta ** (NUM_ITERATIONS - 1))

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_SH_iter - 1, 0, self.max_SH_iter))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_SH_iter': self.max_SH_iter,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

        self.epc = min_epochs_per_config / 2    # Since it will be multiplied by 2 before the first iteration of BOHB
        self.epc_result_tracker = []
        # Format for entry of epc_result_tracker: temp = {'epc': 2, 'best_val_acc': 45}

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
        self.run_details = {}

    def check_terminal_condition(self):
        """
        Returns true if no more iterations is to be run
        """

        # if self.bohb_instance < self.min_bohb_count:
        #     return False

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
            return not self.acc_updated
        return False

    def get_next_epc(self):
        if len(self.epc_result_tracker) < 2:
            return ceil(self.epc * 2)
        p1 = self.epc_result_tracker[-2]
        p2 = self.epc_result_tracker[-1]
        slope = (p2['best_val_acc'] - p1['best_val_acc']) / (p2['epc'] - p1['epc'])
        next_epc = p2['epc'] * (1 + slope)
        if next_epc == p2['epc']:
            return ceil(next_epc * 2)
        return ceil(next_epc)

    def get_next_iteration(self, iteration, iteration_kwargs={}):
        """
        BO-HB uses (just like Hyperband) SuccessiveHalving for each iteration.
        See Li et al. (2016) for reference.

        Parameters
        ----------
            iteration: int
                the index of the iteration to be instantiated

        Returns
        -------
            SuccessiveHalving: the SuccessiveHalving iteration with the
                corresponding number of configurations
        """

        if len(self.run_details) > 1 and self.check_terminal_condition():
            return None

        if iteration % self.max_SH_iter == 0:
            self.update_aux_budget_tracker(iteration)
            self.epc = self.get_next_epc()

        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # number of configurations in that bracket
        n0 = int(np.floor(self.max_SH_iter / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return (SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s - 1):],
                                  config_sampler=self.config_generator.get_config, **iteration_kwargs))

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
        upper = iteration  # exclusive
        lower = iteration - self.max_SH_iter  # inclusive
        best_acc_in_current_bohb = self._find_best_acc_in_bohb_iter(range(lower, upper))
        if best_acc_in_current_bohb > self.best_acc_received + self.acc_saturation_delta:
            self.best_acc_received = best_acc_in_current_bohb
            self.acc_updated = True
        else:
            self.acc_updated = False

        temp = {
            'epc': self.epc,
            'best_val_acc': best_acc_in_current_bohb
        }
        self.epc_result_tracker.append(temp)
        return True

    def _find_best_acc_in_bohb_iter(self, sh_iteration_ids):
        best = 0
        for i in sh_iteration_ids:
            for configs in self.run_details[i]:
                if self.run_details[i][configs]['val_acc'] > best:
                    best = self.run_details[i][configs]['val_acc']
        return best

    def job_callback(self, job):
        if job.result is not None:
            self.register_result_for_optim(job.id, job.result)
            # self.update_run_to_optimizer(job.id, job.result)
        super().job_callback(job)
