import logging

from Utility.Worker_Utils import *

logging.basicConfig(level=logging.DEBUG)
import numpy as np
import torch
from hpbandster.core.worker import Worker
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F
from Utility.helper_sst import *
from Utility.CONFIG import *
from Models.SST import WordAveragingLinear


class SSTWorker(Worker):
    def __init__(self, **kwargs):
        self.fixed_execution_type = None  # This will only be set if the execution type is fixed horizon
        self.fixed_epc = None
        """
        possibilities for fixed_execution_type: 'trainset', 'epoch', 'time'
        """

        self.DATA_LOC = 'SST_Data/'
        super().__init__(**kwargs)
        self.hpo_optimizer = None

        vocab_size = 10000
        max_example_len = 30

        self.X_train, self.Y_train = load_data('SST_Data/senti.binary.train')
        self.X_dev, self.Y_dev = load_data('SST_Data/senti.binary.dev')
        self.X_test, self.Y_test = load_data('SST_Data/senti.binary.test')

        # /Users/shikhar/Data/SST
        # SST_Data/
        # self.X_train, self.Y_train = load_data('/Users/shikhar/Data/SST/senti.binary.train')
        # self.X_dev, self.Y_dev = load_data('/Users/shikhar/Data/SST/senti.binary.dev')
        # self.X_test, self.Y_test = load_data('/Users/shikhar/Data/SST/senti.binary.test')

        vocab = get_vocab(self.X_train)
        self.X_train = text_to_rank(self.X_train, vocab, vocab_size)
        self.X_dev = text_to_rank(self.X_dev, vocab, vocab_size)
        self.X_test = text_to_rank(self.X_test, vocab, vocab_size)

        self.X_train = np.array(pad_sequences(self.X_train, maxlen=max_example_len), dtype=np.long)
        self.X_dev = np.array(pad_sequences(self.X_dev, maxlen=max_example_len), dtype=np.long)
        self.X_test = np.array(pad_sequences(self.X_test, maxlen=max_example_len), dtype=np.long)

        self.Y_train = np.array(self.Y_train, dtype=np.long)
        self.Y_dev = np.array(self.Y_dev, dtype=np.long)
        self.Y_test = np.array(self.Y_test, dtype=np.long)
        print('Data loaded')

        self.num_classes = 2

    def update_fixed_exp_type(self, exp_type, epc=None):
        self.fixed_execution_type = exp_type
        self.fixed_epc = epc


    def prepare_data(self, corruption_matrix, gold_fraction=0.5, sample_percentage=1.0):
        examples = np.copy(self.X_train)
        labels = np.copy(self.Y_train)
        ret_len = int(sample_percentage * len(examples))
        examples = examples[:ret_len]
        labels = labels[:ret_len]

        indices = np.arange(len(labels))
        np.random.shuffle(indices)

        examples = examples[indices]
        labels = labels[indices]

        num_gold = int(len(labels) * gold_fraction)
        num_silver = len(labels) - num_gold

        for i in range(num_silver):
            labels[i] = np.random.choice(self.num_classes, p=corruption_matrix[labels[i]])

        dataset = {'x': examples, 'y': labels}
        gold = {'x': dataset['x'][num_silver:], 'y': dataset['y'][num_silver:]}

        return dataset, gold, num_gold, num_silver

    def uniform_mix_C(self, mixing_ratio=0):
        '''
        returns a linear interpolation of a uniform matrix and an identity matrix
        '''
        nc = self.num_classes
        return mixing_ratio * np.full((nc, nc), 1 / nc) + \
               (1 - mixing_ratio) * np.eye(nc)

    def flip_labels_C(self, corruption_prob=0):
        '''
        returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
        concentrated in only one other entry for each row
        '''

        C = np.eye(self.num_classes) * (1 - corruption_prob)
        row_indices = np.arange(self.num_classes)
        for i in range(self.num_classes):
            C[i][np.random.choice(row_indices[row_indices != i])] = corruption_prob
        return C

    def train_and_test_epochs(self, num_epochs, lr, decay_param, batch_size, vocab_size, embedding_dims,
                              sample_percentage=1.0,
                              method='ours',
                              corruption_level=0,
                              gold_fraction=0.5,
                              get_C=uniform_mix_C):

        net = WordAveragingLinear(vocab_size, embedding_dims, self.num_classes).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay_param)

        C = get_C(self, corruption_level)

        dataset, gold, num_gold, num_silver = self.prepare_data(C, gold_fraction, sample_percentage=sample_percentage)

        # //////////////////////// train for estimation ////////////////////////

        if method == 'ours' or method == 'confusion' or method == 'forward_gold' or method == 'ideal':
            num_examples = num_silver
        else:
            # elif method == 'forward':
            num_examples = dataset['y'].shape[0]

        num_batches = num_examples // batch_size

        indices = np.arange(num_examples)
        num_epochs = int(num_epochs)
        for epoch in range(num_epochs):
            # shuffle data every epoch
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size

                x_batch = dataset['x'][indices[offset:offset + batch_size]]
                y_batch = dataset['y'][indices[offset:offset + batch_size]]
                data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

                # forward
                output = net(data)

                # backward
                l2_loss = (net.out.weight ** 2).sum() / 2

                # TODO: Is it really decay_param?
                loss = F.cross_entropy(output, target) + (decay_param * l2_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_train).cuda(), torch.from_numpy(self.Y_train).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        # baseline_acc = correct.item() / len(self.Y_test)
        baseline_acc = torch.true_divide(correct, len(self.Y_test))

        # //////////////////////// estimate C ////////////////////////
        if method == 'ours':
            with torch.no_grad():
                t1 = torch.from_numpy(gold['x']).cuda()
                probs = F.softmax(net(t1), 1).data.cpu().numpy()
                C_hat = np.zeros((self.num_classes, self.num_classes))
                for label in range(self.num_classes):
                    indices = np.arange(len(gold['y']))[gold['y'] == label]
                    C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)

        elif method == 'forward' or method == 'forward_gold':
            with torch.no_grad():
                t1 = torch.from_numpy(dataset['x']).cuda()
                probs = F.softmax(net(t1), 1).data.cpu().numpy()
                C_hat = np.zeros((self.num_classes, self.num_classes))
                for label in range(self.num_classes):
                    class_probs = probs[:, label]
                    thresh = np.percentile(class_probs, 97, interpolation='higher')
                    class_probs[class_probs >= thresh] = 0

                    C_hat[label] = probs[np.argsort(class_probs)][-1]

        elif method == 'ideal':
            C_hat = C

        elif method == 'confusion':
            # directly estimate confusion matrix on gold
            probs = F.softmax(net(V(torch.from_numpy(gold['x']).cuda(), volatile=True))).data.cpu().numpy()
            preds = np.argmax(probs, axis=1)

            C_hat = np.zeros([self.num_classes, self.num_classes])

            for i in range(len(gold['y'])):
                C_hat[gold['y'][i], preds[i]] += 1

            C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)

            C_hat = C_hat * 0.99 + np.full_like(C_hat, 1 / self.num_classes) * 0.01

        print('True C:', np.round(C, decimals=3))
        print('C_hat:', np.round(C_hat, decimals=3))

        C_hat = V(torch.from_numpy(C_hat.astype(np.float32))).cuda()

        # //////////////////////// retrain with correction ////////////////////////
        net.train()
        net.init_weights()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)

        if method == 'ours' or method == 'ideal' or method == 'confusion' or method == 'forward_gold':
            num_examples = dataset['y'].shape[0]
            num_batches = num_examples // batch_size

            indices = np.arange(num_examples)
            for epoch in range(num_epochs):
                np.random.shuffle(indices)

                for i in range(num_batches):
                    offset = i * batch_size
                    current_indices = indices[offset:offset + batch_size]

                    data = dataset['x'][current_indices]
                    target = dataset['y'][current_indices]

                    gold_indices = current_indices >= num_silver
                    silver_indices = current_indices < num_silver

                    gold_len = np.sum(gold_indices)
                    if gold_len > 0:
                        data_g, target_g = data[gold_indices], target[gold_indices]
                        data_g, target_g = V(torch.LongTensor(data_g).cuda()), \
                                           V(torch.from_numpy(target_g).long().cuda())

                    silver_len = np.sum(silver_indices)
                    if silver_len > 0:
                        data_s, target_s = data[silver_indices], target[silver_indices]
                        data_s, target_s = V(torch.LongTensor(data_s).cuda()), \
                                           V(torch.from_numpy(target_s).long().cuda())

                    # forward
                    loss_s = 0
                    if silver_len > 0:
                        output_s = net(data_s)
                        output_s -= torch.max(output_s, 1, keepdim=True)[0]
                        output_s = torch.log(torch.mm(F.softmax(output_s, 1), C_hat))
                        loss_s = F.cross_entropy(output_s, target_s, size_average=False)
                        # pre1 = C_hat.t()[torch.cuda.LongTensor(target_s.data)]
                        # pre2 = torch.mul(F.softmax(output_s), pre1)
                        # loss_s = -(torch.log(pre2.sum(1))).sum(0)
                    loss_g = 0
                    if gold_len > 0:
                        output_g = net(data_g)
                        loss_g = F.cross_entropy(output_g, target_g, size_average=False)

                    # backward
                    l2_loss = (net.out.weight ** 2).sum() / 2
                    loss = (loss_g + loss_s) / batch_size + (decay_param * l2_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        elif method == 'forward':
            num_examples = dataset['y'].shape[0]
            num_batches = num_examples // batch_size

            indices = np.arange(num_examples)
            for epoch in range(num_epochs):
                np.random.shuffle(indices)

                for i in range(num_batches):
                    offset = i * batch_size

                    x_batch = dataset['x'][indices[offset:offset + batch_size]]
                    y_batch = dataset['y'][indices[offset:offset + batch_size]]
                    data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

                    # forward
                    output = net(data)
                    pre1 = C_hat.t()[torch.cuda.LongTensor(target.data)]
                    pre2 = torch.mul(F.softmax(output), pre1)
                    loss = -(torch.log(pre2.sum(1))).mean(0)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # //////////////////////// evaluate method ////////////////////////
        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_test).cuda(), \
                           torch.from_numpy(self.Y_test.astype(np.long)).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        test_acc = correct.item() / len(self.Y_test)

        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_dev).cuda(), \
                           torch.from_numpy(self.Y_dev.astype(np.long)).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        val_acc = correct.item() / len(self.Y_dev)

        # nudge garbage collector
        del dataset;
        del gold

        return baseline_acc, val_acc, test_acc

    def train_and_test_duration(self, duration, lr, decay_param, batch_size, vocab_size, embedding_dims,
                                sample_percentage=1.0,
                                method='ours',
                                corruption_level=0,
                                gold_fraction=0.5,
                                get_C=uniform_mix_C):

        duration *= 60
        duration /= 2
        # Dividing by 2, since the models are trained twice

        e1 = 0
        e2 = 0

        net = WordAveragingLinear(vocab_size, embedding_dims, self.num_classes).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay_param)

        C = get_C(self, corruption_level)

        dataset, gold, num_gold, num_silver = self.prepare_data(C, gold_fraction, sample_percentage=sample_percentage)

        # //////////////////////// train for estimation ////////////////////////

        if method == 'ours' or method == 'confusion' or method == 'forward_gold' or method == 'ideal':
            num_examples = num_silver
        else:
            # elif method == 'forward':
            num_examples = dataset['y'].shape[0]

        num_batches = num_examples // batch_size

        indices = np.arange(num_examples)

        # for epoch in range(num_epochs):
        start_time = dt.now()
        while True:
            if (dt.now() - start_time).seconds >= duration:
                break
            e1 += 1

            # shuffle data every epoch
            np.random.shuffle(indices)

            for i in range(num_batches):
                offset = i * batch_size

                x_batch = dataset['x'][indices[offset:offset + batch_size]]
                y_batch = dataset['y'][indices[offset:offset + batch_size]]
                data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

                # forward
                output = net(data)

                # backward
                l2_loss = (net.out.weight ** 2).sum() / 2

                # TODO: Is it really decay_param?
                loss = F.cross_entropy(output, target) + (decay_param * l2_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_train).cuda(), torch.from_numpy(self.Y_train).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        baseline_acc = correct.item() / len(self.Y_test)

        # //////////////////////// estimate C ////////////////////////
        if method == 'ours':
            with torch.no_grad():
                t1 = torch.from_numpy(gold['x']).cuda()
                probs = F.softmax(net(t1), 1).data.cpu().numpy()
                C_hat = np.zeros((self.num_classes, self.num_classes))
                for label in range(self.num_classes):
                    indices = np.arange(len(gold['y']))[gold['y'] == label]
                    C_hat[label] = np.mean(probs[indices], axis=0, keepdims=True)

        elif method == 'forward' or method == 'forward_gold':
            with torch.no_grad():
                t1 = torch.from_numpy(dataset['x']).cuda()
                probs = F.softmax(net(t1), 1).data.cpu().numpy()
                C_hat = np.zeros((self.num_classes, self.num_classes))
                for label in range(self.num_classes):
                    class_probs = probs[:, label]
                    thresh = np.percentile(class_probs, 97, interpolation='higher')
                    class_probs[class_probs >= thresh] = 0

                    C_hat[label] = probs[np.argsort(class_probs)][-1]

        elif method == 'ideal':
            C_hat = C

        elif method == 'confusion':
            # directly estimate confusion matrix on gold
            probs = F.softmax(net(V(torch.from_numpy(gold['x']).cuda(), volatile=True))).data.cpu().numpy()
            preds = np.argmax(probs, axis=1)

            C_hat = np.zeros([self.num_classes, self.num_classes])

            for i in range(len(gold['y'])):
                C_hat[gold['y'][i], preds[i]] += 1

            C_hat /= (np.sum(C_hat, axis=1, keepdims=True) + 1e-7)

            C_hat = C_hat * 0.99 + np.full_like(C_hat, 1 / self.num_classes) * 0.01

        print('True C:', np.round(C, decimals=3))
        print('C_hat:', np.round(C_hat, decimals=3))

        C_hat = V(torch.from_numpy(C_hat.astype(np.float32))).cuda()

        # //////////////////////// retrain with correction ////////////////////////
        net.train()
        net.init_weights()
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0)

        if method == 'ours' or method == 'ideal' or method == 'confusion' or method == 'forward_gold':
            num_examples = dataset['y'].shape[0]
            num_batches = num_examples // batch_size

            indices = np.arange(num_examples)
            e2 = 0
            # for epoch in range(num_epochs):
            start_time = dt.now()
            while True:
                if (dt.now() - start_time).seconds >= duration:
                    break
                e2 += 1
                np.random.shuffle(indices)

                for i in range(num_batches):
                    offset = i * batch_size
                    current_indices = indices[offset:offset + batch_size]

                    data = dataset['x'][current_indices]
                    target = dataset['y'][current_indices]

                    gold_indices = current_indices >= num_silver
                    silver_indices = current_indices < num_silver

                    gold_len = np.sum(gold_indices)
                    if gold_len > 0:
                        data_g, target_g = data[gold_indices], target[gold_indices]
                        data_g, target_g = V(torch.LongTensor(data_g).cuda()), \
                                           V(torch.from_numpy(target_g).long().cuda())

                    silver_len = np.sum(silver_indices)
                    if silver_len > 0:
                        data_s, target_s = data[silver_indices], target[silver_indices]
                        data_s, target_s = V(torch.LongTensor(data_s).cuda()), \
                                           V(torch.from_numpy(target_s).long().cuda())

                    # forward
                    loss_s = 0
                    if silver_len > 0:
                        output_s = net(data_s)
                        output_s -= torch.max(output_s, 1, keepdim=True)[0]
                        output_s = torch.log(torch.mm(F.softmax(output_s, 1), C_hat))
                        loss_s = F.cross_entropy(output_s, target_s, size_average=False)
                        # pre1 = C_hat.t()[torch.cuda.LongTensor(target_s.data)]
                        # pre2 = torch.mul(F.softmax(output_s), pre1)
                        # loss_s = -(torch.log(pre2.sum(1))).sum(0)
                    loss_g = 0
                    if gold_len > 0:
                        output_g = net(data_g)
                        loss_g = F.cross_entropy(output_g, target_g, size_average=False)

                    # backward
                    l2_loss = (net.out.weight ** 2).sum() / 2
                    loss = (loss_g + loss_s) / batch_size + (decay_param * l2_loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        elif method == 'forward':
            num_examples = dataset['y'].shape[0]
            num_batches = num_examples // batch_size

            indices = np.arange(num_examples)
            e2 = 0
            # for epoch in range(num_epochs):
            start_time = dt.now()
            while True:
                if (dt.now() - start_time).seconds >= duration:
                    break
                e2 += 1
                np.random.shuffle(indices)

                for i in range(num_batches):
                    offset = i * batch_size

                    x_batch = dataset['x'][indices[offset:offset + batch_size]]
                    y_batch = dataset['y'][indices[offset:offset + batch_size]]
                    data, target = V(torch.from_numpy(x_batch).cuda()), V(torch.from_numpy(y_batch).cuda())

                    # forward
                    output = net(data)
                    pre1 = C_hat.t()[torch.cuda.LongTensor(target.data)]
                    pre2 = torch.mul(F.softmax(output), pre1)
                    loss = -(torch.log(pre2.sum(1))).mean(0)

                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        # //////////////////////// evaluate method ////////////////////////
        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_test).cuda(), \
                           torch.from_numpy(self.Y_test.astype(np.long)).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        test_acc = correct.item() / len(self.Y_test)

        net.eval()
        with torch.no_grad():
            data, target = torch.from_numpy(self.X_dev).cuda(), \
                           torch.from_numpy(self.Y_dev.astype(np.long)).cuda()
            output = net(data)
            pred = output.data.max(1)[1]
            correct = pred.eq(target.data).sum()

        val_acc = correct.item() / len(self.Y_dev)
        avg_epoch_consumed = (e1 + e2) // 2

        # nudge garbage collector
        del dataset;
        del gold

        return baseline_acc, val_acc, test_acc, avg_epoch_consumed

    def return_temp_results(self, optimizer):
        return {
            'loss': 0.9,  # remember: HpBandSter always minimizes!
            'info': {
                'epc': self.hpo_optimizer.epc,
                'trainset_budget': self.hpo_optimizer.trainset_budget,
                'epoch_multiplier': self.hpo_optimizer.epoch_multiplier,
                'test_accuracy': 0,
                'validation_accuracy': 0,
                'test_confidence': 0,
                'validation_confidence': 0,
                'trainset_consumed': 0,
                'epochs_for_time_budget': 0,
                'time_multiplier': 0
            }
        }

    def compute(self, config_id, config, budget, working_directory):
        # if self.hpo_optimizer.algo_type == 'time-based' or self.hpo_optimizer.active_test_algo == 'time-based':
        #     pass
        # else:
        #     return self.return_temp_results(None)

        batch_size = config['batch_size']
        # embedding_dims = config['embedding_dims']
        embedding_dims = 100
        lr = config['lr']
        decay_param = config['decay_param']
        gold_fraction = config['gold_fraction']
        vocab_size = 10000

        epoch_multiplier = 1
        time_multiplier = 1
        trainset_budget = 1
        epc = None
        epochs_for_time_budget = None
        execution_algo_type = 'epoch_based'
        if hasattr(self.hpo_optimizer, 'epoch_multiplier'):
            epoch_multiplier = self.hpo_optimizer.epoch_multiplier
        if hasattr(self.hpo_optimizer, 'trainset_budget'):
            trainset_budget = self.hpo_optimizer.trainset_budget
        if hasattr(self.hpo_optimizer, 'epc'):
            epc = self.hpo_optimizer.epc
        if hasattr(self.hpo_optimizer, 'time_multiplier'):
            time_multiplier = self.hpo_optimizer.time_multiplier
        if hasattr(self.hpo_optimizer, 'algo_type'):
            if self.hpo_optimizer.algo_type in EPOCH_BASED_METHODS \
                    or self.hpo_optimizer.active_test_algo in EPOCH_BASED_METHODS:
                execution_algo_type = 'epoch_based'
            elif self.hpo_optimizer.algo_type == 'trainset-with-increasing-epc' or \
                    self.hpo_optimizer.active_test_algo == 'trainset-with-increasing-epc':
                execution_algo_type = 'trainset_based'
            else:
                execution_algo_type = 'time_based'
        else:
            if self.fixed_execution_type == 'epoch':
                execution_algo_type = 'epoch_based'
            elif self.fixed_execution_type == 'trainset':
                execution_algo_type = 'trainset_based'
                epc = self.fixed_epc
            elif self.fixed_execution_type == 'time':
                execution_algo_type = 'time_based'

        if execution_algo_type == 'epoch_based':
            e = budget * epoch_multiplier
            trainset_consumed = trainset_budget * e
            trainset_percentage = trainset_budget
        elif execution_algo_type == 'trainset_based':
            e = epc
            trainset_consumed = budget * e
            trainset_percentage = budget
        else:
            trainset_consumed = trainset_budget  # Will be multiplied by epochs after eval
            trainset_percentage = trainset_budget

        if DEBUG:
            return return_fake_results(config_id, self.hpo_optimizer, trainset_consumed)

        elif execution_algo_type == 'time_based':
            train_acc, val_acc, test_acc, epochs_for_time_budget = self.train_and_test_duration(
                duration=budget * time_multiplier,
                lr=lr,
                decay_param=decay_param,
                batch_size=batch_size,
                vocab_size=vocab_size,
                embedding_dims=embedding_dims,
                sample_percentage=trainset_percentage,
                gold_fraction=gold_fraction
            )
            trainset_consumed *= epochs_for_time_budget
        else:
            train_acc, val_acc, test_acc = self.train_and_test_epochs(
                num_epochs=e,
                lr=lr,
                decay_param=decay_param,
                batch_size=batch_size,
                vocab_size=vocab_size,
                embedding_dims=embedding_dims,
                sample_percentage=trainset_percentage,
                gold_fraction=gold_fraction
            )



        return {
            'loss': 1 - val_acc,  # remember: HpBandSter always minimizes!
            'info': {
                'epc': epc,
                'trainset_budget': trainset_budget,
                'epoch_multiplier': epoch_multiplier,
                'test_accuracy': test_acc,
                'validation_accuracy': val_acc,
                'test_confidence': None,
                'validation_confidence': None,
                'trainset_consumed': trainset_consumed,
                'epochs_for_time_budget': epochs_for_time_budget,
                'time_multiplier': time_multiplier
            }
        }

    @staticmethod
    def get_configspace():
        cs = CS.ConfigurationSpace()
        gold_fraction = CSH.UniformFloatHyperparameter('gold_fraction', lower=0.1, upper=1)
        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-5, upper=1e-1, default_value='1e-2', log=True)
        batch_size = CSH.CategoricalHyperparameter('batch_size', [32, 64, 128, 256, 512])
        decay_param = CSH.UniformFloatHyperparameter('decay_param', lower=1e-7, upper=1e-3)
        cs.add_hyperparameters([gold_fraction, lr, batch_size, decay_param])
        return cs


if __name__ == '__main__':
    worker = SSTWorker(run_id='0')
    cs = worker.get_configspace()
    c = cs.sample_configuration().get_dictionary()
    print(c)
    res = worker.compute(config=c, config_id=(0, 0, 0), budget=2, working_directory='.')
    print(res)
