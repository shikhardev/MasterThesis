import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_processed_titanic_dataset():
    """
    https://www.kaggle.com/goldens/titanic-on-the-top-with-a-simple-model
    """
    import pandas as pd
    train = pd.read_csv("~/Data/TITANIC/train.csv")
    test = pd.read_csv("~/Data/TITANIC/test.csv")
    """
    Following fields are nulls
    Survived     418
    Age          263
    Fare           1
    Cabin       1014
    Embarked       2
    """
    train.Fare = train.Fare.fillna(train.Fare.mean())
    test.Fare = test.Fare.fillna(train.Fare.mean())
    train.Cabin = train.Cabin.fillna("unknow")
    test.Cabin = test.Cabin.fillna("unknow")
    train.Embarked = train.Embarked.fillna(train.Embarked.mode()[0])
    test.Embarked = test.Embarked.fillna(train.Embarked.mode()[0])
    train['title'] = train.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
    test['title'] = test.Name.apply(lambda x: x.split('.')[0].split(',')[1].strip())
    newtitles = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir": "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess": "Royalty",
        "Dona": "Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr": "Mr",
        "Mrs": "Mrs",
        "Miss": "Miss",
        "Master": "Master",
        "Lady": "Royalty"}
    train['title'] = train.title.map(newtitles)
    test['title'] = test.title.map(newtitles)

    def newage(cols):
        title = cols[0]
        Sex = cols[1]
        Age = cols[2]
        if pd.isnull(Age):
            if title == 'Master' and Sex == "male":
                return 4.57
            elif title == 'Miss' and Sex == 'female':
                return 21.8
            elif title == 'Mr' and Sex == 'male':
                return 32.37
            elif title == 'Mrs' and Sex == 'female':
                return 35.72
            elif title == 'Officer' and Sex == 'female':
                return 49
            elif title == 'Officer' and Sex == 'male':
                return 46.56
            elif title == 'Royalty' and Sex == 'female':
                return 40.50
            else:
                return 42.33
        else:
            return Age

    train.Age = train[['title', 'Sex', 'Age']].apply(newage, axis=1)
    test.Age = test[['title', 'Sex', 'Age']].apply(newage, axis=1)

    train['Relatives'] = train.SibSp + train.Parch
    test['Relatives'] = test.SibSp + test.Parch

    train['Ticket2'] = train.Ticket.apply(lambda x: len(x))
    test['Ticket2'] = test.Ticket.apply(lambda x: len(x))

    train['Cabin2'] = train.Cabin.apply(lambda x: len(x))
    test['Cabin2'] = test.Cabin.apply(lambda x: len(x))

    train['Name2'] = train.Name.apply(lambda x: x.split(',')[0].strip())
    test['Name2'] = test.Name.apply(lambda x: x.split(',')[0].strip())

    # droping features not used in model
    train.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    test.drop(['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    train.Survived = train.Survived.astype('int')
    xtrain=train.drop("Survived",axis=1)
    ytrain=train['Survived']
    xtest=test.drop("Survived", axis=1)

def get_dataset(dataset_name):
    """
    Returns the train and test dataset
    """
    DATA_LOCATION = '~/Data/' + dataset_name
    if dataset_name == 'MNIST':
        train_dataset = torchvision.datasets.MNIST(root=DATA_LOCATION, train=True, transform=transforms.ToTensor(),
                                                   download=True)
        test_dataset = torchvision.datasets.MNIST(root=DATA_LOCATION, train=False, transform=transforms.ToTensor(),
                                                  download=True)

    elif dataset_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = torchvision.datasets.CIFAR10(root=DATA_LOCATION, train=True, transform=transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=DATA_LOCATION, train=False, transform=transform, download=True)

    elif dataset_name == 'FASHION':
        # normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                                  std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        train_dataset = torchvision.datasets.FashionMNIST(root=DATA_LOCATION, train=True, transform=transform,
                                                          download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=DATA_LOCATION, train=False, transform=transform,
                                                         download=True)

    elif dataset_name == 'TITANIC':
        pass



        
    return train_dataset, test_dataset


def dataset_to_dataloaders(dataset, sizes=None, batch_sizes=(1024,), shuffles=(False,), num_workers=4):
    """
    Returns one or more dataloaders from the dataset

    :param dataset: The dataset for which loaders need to be created.
    If the dataset is to be split into multiple loaders, this is the full dataset

    :param sizes: The split of sizes for each of the dataset.
    If sizes is None, the whole of the dataset is to be used to create one dataloaders.
    Otherwise, len(sizes) should be the same as len(batch_sizes) and len(shuffles).
    Assumes sum(sizes) = len(dataset)

    :param batch_sizes: List of batch sizes in the loaders.
    len(batch_sizes) should be one if sizes is None.

    :param shuffles: Is the dataset in the split to be shuffled?
    len(shuffles) should be one if sizes is None.

    :param num_workers: Might want to experiment with different values of num_workers. Might result in different speeds.

    :return: List of dataloaders. Length of the returned list is the same as
    """
    if torch.cuda.is_available():
        pin_memory = True
    else:
        pin_memory = False
    if sizes is None:
        return DataLoader(dataset=dataset, batch_size=batch_sizes[0], shuffle=shuffles[0],
                          num_workers=num_workers, pin_memory=pin_memory)

    loaders = []
    res_datasets = torch.utils.data.random_split(dataset, sizes)

    for ind, val in enumerate(batch_sizes):
        temp = DataLoader(dataset=res_datasets[ind], batch_size=batch_sizes[ind], shuffle=shuffles[ind],
                          num_workers=num_workers, pin_memory=pin_memory)
        loaders.append(temp)
    return loaders


def calculate_trainset_data_budget(old_budget, required_budget):
    """
    x % of remaining data = required data. Required data is not 100%.
    doesn't So, 50% budget doesn't really mean 50% of current data, since
    chunks of data have been discarded in the previous budget.
    This function returns the percentage of data to be taken from the remaining dataset
    to match the original required budget
    :param old_budget: The percentage of the whole dataset that has already been used
    :param required_budget: Budget requirement for the rung, in respect to the whole dataset
    :return: Budget requirement fot the rung in respect to the remaining dataset
    """
    delta = required_budget - old_budget
    remaining_budget = 1.0 - old_budget
    # x % of remaining_budget = delta
    x = delta / remaining_budget
    return x


def get_fraction_of_data(fraction, dataset):
    """
    Returns two datasets split from input dataset in the input fraction
    :return: d1, d2:    d1 is the dataset to be used for this evaluation,
                        d2 is the remaining dataset
    """
    total_size = len(dataset)
    s1 = int(fraction * total_size)
    s2 = total_size - s1
    if s1 == 0:
        return 0, dataset
    elif s2 == 0:
        return dataset, 0

    d1, d2 = torch.utils.data.random_split(dataset, [s1, s2])
    return d1, d2
