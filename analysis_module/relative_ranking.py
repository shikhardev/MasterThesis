import glob
import pandas as pd
import IPython
from scipy.stats import spearmanr
from scipy.stats import kendalltau
import numpy as np

def get_percentile(total, current):
    return (1 + total - current) / total


def compute_averages(correlation_result):
    res = {}
    for b in correlation_result:
        ignore_count = 0
        s = 0
        for i in correlation_result[b]:
            # if np.isnan(i['correlation']):
            if i['correlation'] == 'nan':
                ignore_count += 1
                continue
            s += i['correlation']
        count = (len(correlation_result[b]) - ignore_count)
        if count == 0:
            res[b] = 0
        else:
            a = s / count
            res[b] = float("{:.3f}".format(a))

    s = 0
    for b in res:
        s += res[b]
    a = s / len(res)
    res['overall'] = float("{:.3f}".format(a))
    return res


def pretty_results(correlation_result):
    res = {}
    for b in correlation_result:
        res[b] = {}
        for i in correlation_result[b]:
            res[b][i['budget']] = i['correlation']

    return res


def get_rank_correlation(d: pd.DataFrame):
    res = {}
    d['bracket'] = d.cid.apply(lambda x: x[0])
    d['hp_id'] = d.cid.apply(lambda x: x[2])
    # We are only concerned with the first SH bracket.
    sh_data = d.groupby(['bracket']).get_group(0)
    sh_data = sh_data.groupby(['budget'])
    budgets = list(sh_data.groups.keys())[::-1]  # Traverse all budgets in reverse order
    total_budgets = len(budgets)
    for i, b in enumerate(budgets):
        cd = sh_data.get_group(b)  # Get all the data for current budget
        cd = cd.sort_values(by='hp_id')  # Sort by validation loss in ascending order
        hp = cd['hp_id'].tolist()  # All the hyperparameters in the current budget
        r1 = cd['validation_loss'].rank(method='max').tolist()
        # for all budgets < current budget, find the correlation
        b = str(b)
        res[b] = []

        for _ in range(i + 1, total_budgets):
            td = sh_data.get_group(budgets[_])
            td['ranks'] = td['validation_loss'].rank(method='max')
            p = 0
            k = 0
            if len(hp) == 1:
                r2 = td[td.hp_id == hp[0]]['ranks'].values[0]  # Rank of hp in target budget
                p = get_percentile(td.shape[0], r2)
            else:
                td = td.loc[data['hp_id'].isin(hp)]
                td = td.sort_values(by='hp_id')  # Sort by validation loss in ascending order
                # Ranking: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rank.html
                r2 = td['validation_loss'].rank(method='max').tolist()
                p = spearmanr(r1, r2).correlation

            temp = {
                # 'index': _,
                'budget': str(budgets[_]),
                'configs': str(hp),
                'r1': str(r1),
                'r2': str(r2),
                'correlation': float("{:.3f}".format(p)),
            }
            if np.isnan(p):
                temp['correlation'] = str(p)
            res[b].append(temp)
    return res


if __name__ == '__main__':
    # d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/MNIST/Epochs/bohb/verbose/'
    # d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/MNIST/Trainset/bohb/verbose/'

    d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Cifar10/Epochs/bohb/verbose/'
    d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_1/Cifar10/Iter/bohb/verbose/'
    d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_1/Cifar10/Trainset/bohb/verbose/'
    # d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Cifar10/Trainset/bohb/verbose/'
    #
    # d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Fashion/Epochs/bohb/verbose/'
    # d = '/Users/shikhar/Desktop/thesis/Analysis Results/Batch_2/Fashion/Trainset/bohb/verbose/'

    for fn in glob.glob(d + '/*.pkl'):
        data = pd.read_pickle(fn)
        r = get_rank_correlation(data)
        # a = compute_averages(r)
        print(fn)
        # print(r)
        print(pretty_results(r))
        print('-'*20)
        # print(a)