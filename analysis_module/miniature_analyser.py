import pandas as pd
from datetime import datetime as dt

home_loc = "/Users/shikhar/Desktop/execute_snats/Downloads/new_downloads/"
ITER = 3

datasets = [
    'CIFAR10',
    'FASHION',
    'MNIST_CNN',
    'SST',
    'TWITTER'
]

miniature_loc = '/Multitune/Run_0/miniature.csv'
miniature_nb_loc = '/Multitune_NoBayesian/Run_0/miniature.csv'
evals = {
    'miniature_overview.csv': '/Multitune/Run_0/miniature.csv',
    'miniature_nb_overview.csv': '/Multitune_NoBayesian/Run_0/miniature.csv'
}

for e in evals:
    mother_of_all = []
    for i in range(ITER):
        for d in datasets:
            file_name = home_loc + str(i) + '/' + d + evals[e]
            df = pd.read_csv(file_name)
            m = df.groupby('Method')
            methods = [m.get_group(x) for x in m.groups]
            for m in methods:
                temp = m.groupby('Budget')
                b = [temp.get_group(x) for x in temp.groups]
                # Only need to look in b[1], since that is the higher budget
                row = b[1]['ValidationAccuracy'].argmax()
                best_time = b[1].iloc[row]['TFinish']
                best_time = dt.strptime(best_time, '%H:%M:%S')
                start_time = b[0].iloc[0]['TFinish']
                start_time = dt.strptime(start_time, '%H:%M:%S')

                curr = {
                    'run': i,
                    'dataset': d,
                    'method': b[1].iloc[0]['Method'],
                    'best_acc': b[1].iloc[row]['ValidationAccuracy'],
                    'duration': (best_time - start_time).seconds
                }
                mother_of_all.append(curr)

    df = pd.DataFrame(mother_of_all).sort_values(['dataset', 'run'])
    df.to_csv(e)