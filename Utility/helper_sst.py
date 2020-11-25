import re
import numpy as np
import collections


def load_data(filename='SST_Data/senti.train.onlyroot'):
    '''
    :param filename: the system location of the data to load
    :return: the text (x) and its label (y)
             the text is a list of words and is not processed
    '''

    # stop words taken from nltk
    stop_words = ['i','me','my','myself','we','our','ours','ourselves','you','your','yours',
                  'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
                  'it','its','itself','they','them','their','theirs','themselves','what','which',
                  'who','whom','this','that','these','those','am','is','are','was','were','be',
                  'been','being','have','has','had','having','do','does','did','doing','a','an',
                  'the','and','but','if','or','because','as','until','while','of','at','by','for',
                  'with','about','against','between','into','through','during','before','after',
                  'above','below','to','from','up','down','in','out','on','off','over','under',
                  'again','further','then','once','here','there','when','where','why','how','all',
                  'any','both','each','few','more','most','other','some','such','no','nor','not',
                  'only','own','same','so','than','too','very','s','t','can','will','just','don',
                  'should','now','d','ll','m','o','re','ve','y','ain','aren','couldn','didn',
                  'doesn','hadn','hasn','haven','isn','ma','mightn','mustn','needn','shan',
                  'shouldn','wasn','weren','won','wouldn']

    x, y = [], []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = re.sub(r'\W+', ' ', line).strip().lower()  # perhaps don't make words lowercase?
            x.append(line[:-1])
            x[-1] = ' '.join(word for word in x[-1].split() if word not in stop_words)
            y.append(line[-1])
    return x, np.array(y, dtype=int)


def get_vocab(dataset):
    '''
    :param dataset: the text from load_data
    :return: a _ordered_ dictionary from words to counts
    '''
    vocab = {}

    # create a counter for each word
    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] = 0

    for example in dataset:
        example_as_list = example.split()
        for word in example_as_list:
            vocab[word] += 1

    # sort from greatest to least by count
    return collections.OrderedDict(sorted(vocab.items(), key=lambda x: x[1], reverse=True))


def text_to_rank(dataset, _vocab, desired_vocab_size=5000):
    '''
    :param dataset: the text from load_data
    :vocab: a _ordered_ dictionary of vocab words and counts from get_vocab
    :param desired_vocab_size: the desired vocabulary size
    words no longer in vocab become UUUNNNKKK
    :return: the text corpus with words mapped to their vocab rank,
    with all sufficiently infrequent words mapped to UUUNNNKKK; UUUNNNKKK has rank desired_vocab_size
    (the infrequent word cutoff is determined by desired_vocab size)
    '''
    _dataset = dataset[:]     # aliasing safeguard
    vocab_ordered = list(_vocab)
    count_cutoff = _vocab[vocab_ordered[desired_vocab_size-1]] # get word by its rank and map to its count

    word_to_rank = {}
    for i in range(len(vocab_ordered)):
        # we add one to make room for any future padding symbol with value 0
        word_to_rank[vocab_ordered[i]] = i + 1

    # we need to ensure that other words below the word on the edge of our desired_vocab size
    # are not also on the count cutoff, so we subtract a bit
    # this is likely quicker than adding another preventative if case
    for i in range(len(vocab_ordered[desired_vocab_size:])):
        _vocab[vocab_ordered[desired_vocab_size+i]] -= 0.1

    for i in range(len(_dataset)):
        example = _dataset[i]
        example_as_list = example.split()
        for j in range(len(example_as_list)):
            try:
                if _vocab[example_as_list[j]] >= count_cutoff:
                    example_as_list[j] = word_to_rank[example_as_list[j]]
                else:
                    example_as_list[j] = desired_vocab_size  # UUUNNNKKK
            except:
                example_as_list[j] = desired_vocab_size  # UUUNNNKKK
        _dataset[i] = example_as_list

    return _dataset


# taken from keras
def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    '''Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x
