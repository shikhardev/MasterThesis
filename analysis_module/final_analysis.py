from Utility.Result_Utils import *
from Utility.CONFIG import *


class ExperimentResult:
    """
    Object contains details of all the runs.
    Object: self.results
        {
            dataset_name: {
                method_name: {
                    run_id: {
                        'raw_results': 'result_object',
                        'run_overview': 'overview_dict_for_one_run',
                        'verbose_df': 'verbose_dataframe_of_run'
                    }
                }
            }
        }
    """
    def __init__(self):
        self.results = {}

    def add_new_result(self, dataset_name, method_name, result_object, store_intermediaries=True, res_dir=None):
        if dataset_name not in self.results:
            self.results[dataset_name] = {}

        if method_name not in self.results[dataset_name]:
            self.results[dataset_name][method_name] = {}
            run_id = 0
        else:
            run_id = max(self.results[dataset_name][method_name].keys())
            run_id += 1

        self.results[dataset_name][method_name][run_id] = {}
        # We'll create a temp dict and store it here at the end
        temp = {}
        temp['raw_result'] = result_object
        temp['run_overview'] = result_object_to_overview(result_object)
        temp['verbose_df'] = result_to_verbose_dataframe(result_object)
        self.results[dataset_name][method_name][run_id] = temp

        d_m_r = dataset_name + '_' + method_name + '_' + str(run_id)
        overview_dir = res_dir + '/Results/Experiments_Overview/' + d_m_r + '/'
        verbose_dir = res_dir + '/Results/Experiments_Verbose/' + d_m_r + '/'
        if store_intermediaries:
            store_run_details(temp['run_overview'], overview_dir, d_m_r)
            store_run_details(temp['verbose_df'], verbose_dir, d_m_r)



