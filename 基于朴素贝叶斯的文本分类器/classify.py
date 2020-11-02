import os
import math


# 列出在filepath这个路径下的所有出现过cutoff次或者以上次数的次
# >>> create_vocabulary('./EasyFiles/', 1)
# => [',', '.', '19', '2020', 'a', 'cat', 'chases', 'dog', 'february', 'hello', 'is', 'it', 'world']
# >>> create_vocabulary('./EasyFiles/', 2)
# => ['.', 'a']
def create_vocabulary(filepath, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    vocab,temp,files= [],{},[]
    if os.path.isdir(filepath):
        files=read_dir(filepath,[])
    else:
        parent = filepath.split('/')
        year = str(parent[len(parent)-2])
        files=[[filepath,year]]
    for file in files:
        f = open(file[0])
        for line in f:
            line = line.strip('\n')
            if line not in temp:
                temp[line]=1
            else:
                temp[line]+=1
        for word in temp:
            if temp[word]>=cutoff and word not in vocab:
                vocab.append(word)
    return sorted(vocab)

# 由create_vocabulary得到某个路径下的词汇表
# >>> vocab = create_vocabulary('./EasyFiles/', 1)
# 计算filepath路径下这个文件里的词汇在词汇表中出现的次数
# >>> create_bow(vocab, './EasyFiles/2016/1.txt')
# => {'a': 2, 'dog': 1, 'chases': 1, 'cat': 1, '.': 1}

# 如果在文档中出现但没有在词汇表中出现用none代替
# >>> vocab = create_vocabulary('./EasyFiles/', 2)
# >>> create_bow(vocab, './EasyFiles/2016/1.txt')
# => {'a': 2, None: 3, '.': 1}
def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {}
    # prepare word direction for this file
    f=open(filepath)
    temp={}
    for line in f:
        line = line.strip('\n')
        if line not in temp:
            temp[line] = 1
        else:
            temp[line] += 1
    # compare with the vocab
    for word in temp:
        if word not in vocab:
            if 'None' not in bow:
                bow['None']=1
            else:
                bow['None'] += 1
        else:
            if word not in bow:
                bow[word]=temp[word]
    return bow

# 分别计算directory下所有文档的词袋
# >>> vocab = create_vocabulary('./EasyFiles/', 1)
# >>> load_training_data(vocab,'./EasyFiles/')
# => [{'label': '2020', 'bow': {'it': 1, 'is': 1, 'february': 1, '19': 1, ',': 1, '2020': 1, '.': 1}},
#     {'label': '2016', 'bow': {'hello': 1, 'world': 1}},
#     {'label': '2016', 'bow': {'a': 2, 'dog': 1, 'chases': 1, 'cat': 1, '.': 1}}]

def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    dataset = []
    files = read_dir(directory, [])
    for file in files:
        temp={}
        temp['label']=file[1]
        temp['bow']=create_bow(vocab,file[0])
        dataset.append(temp)

    return dataset


def read_dir(path,files):
    # if the path is point to a dir, get its files
    if os.path.isdir(path):
        dir=os.listdir(path)
        for names in dir:
            cur_path=path+'/'+names
            if os.path.isdir(cur_path):
                read_dir(cur_path,files)
            else:
                if cur_path not in files:
                    parent=path.split('/')
                    year=str(parent[-1])
                    files.append([cur_path,year])
        return files

# logP(label)
# p=(某标签下的文件数量+smooth)/（训练集中文件总数+|v|）
# Laplace smoothing 是为了防止0出现影响取对数
# >>> vocab = create_vocabulary('./corpus/training/', 2)
# >>> training_data = load_training_data(vocab,'./corpus/training/')
# >>> prior(training_data, ['2020', '2016'])
# => {'2020': -0.32171182103809226, '2016': -1.2906462863976689}
def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """
    smooth = 1  # smoothing factor
    logprob= {}
    for label in label_list:
        num=0
        for dir in training_data:
            if dir['label']==label:
                num+=1
        p=(num+smooth)/(len(training_data)+2)
        logprob[label]=math.log(p)
    return logprob

# logP(word|label)
# 在某标签下某词汇出现的频率
# >>> vocab = create_vocabulary('./EasyFiles/', 2)
# >>> training_data = load_training_data(vocab, './EasyFiles/')
# >>> p_word_given_label(vocab, training_data, '2020')
# => {'.': -1.6094379124341005, 'a': -2.302585092994046, None: -0.35667494393873267}
# >>> p_word_given_label(vocab, training_data, '2016')
# => {'.': -1.7047480922384253, 'a': -1.2992829841302609, None: -0.6061358035703157}
def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    smooth = 1  # smoothing factor
    word_prob = {}
    # the number of words in the label
    num_label = 0
    for dir in training_data:
        if dir['label'] == label:
                num_label += sum(dir['bow'][i] for i in dir['bow'])

    # None should be included
    voc_copy = vocab[:]
    voc_copy.append('None')
    for word in voc_copy:
        # the number of the word in the label
        num_word_label=0
        for dir in training_data:
            if dir['label']==label and word in dir['bow']:
                num_word_label+=dir['bow'][word]

        p=(num_word_label+smooth)/(num_label+len(voc_copy))
        word_prob[word]=math.log(p)
    return word_prob


##################################################################################
# 整合之前的方程，对训练路径下的文档进行计算概率
# >>> train('./EasyFiles/', 2)
# => {'vocabulary': ['.', 'a'],
#     'log prior': {'2020': -0.916290731874155, '2016': -0.5108256237659905},
#     'log p(w|y=2020)': {'.': -1.6094379124341005, 'a': -2.302585092994046, None: -0.35667494393873267},
#     'log p(w|y=2016)': {'.': -1.7047480922384253, 'a': -1.2992829841302609, None: -0.6061358035703157}}
def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    vocab = create_vocabulary(training_directory, cutoff)
    training_data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior']=prior(training_data, ['2020','2016'])
    retval['log p(w|y=2016)']=p_word_given_label(vocab, training_data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, training_data, '2020')
    return retval

# P=P(label)P(word_1|label)P(word_2|label)....
# 找出对于两个label可能性较大的
# >>> model = train('./corpus/training/', 2)
# >>> classify(model, './corpus/test/2016/0.txt')
# => {'log p(y=2020|x)': -3906.351945884105, 'log p(y=2016|x)': -3916.458747858926, 'predicted y': '2020'}
def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    vocab=model['vocabulary']
    voc_file=create_bow(vocab, filepath)
    p_2016,p_2020=0,0
    for word in voc_file:
        if word in model['log p(w|y=2016)']:
            p_2016+=(model['log p(w|y=2016)'][word]*voc_file[word])
        if word in model['log p(w|y=2020)']:
            p_2020+=(model['log p(w|y=2020)'][word]*voc_file[word])
    p_2016+=model['log prior']['2016']
    p_2020+= model['log prior']['2020']
    if p_2016>=p_2020:
        retval['predicted y']='2016'
    else:
        retval['predicted y'] = '2020'
    retval['log p(y=2016|x)']=p_2016
    retval['log p(y=2020|x)'] = p_2020
    return retval

# import time
# time_start=time.time()
# model = train('./corpus/training/', 2)
# print(classify(model, './corpus/test/2016/0.txt'))
# time_end=time.time()
# print('time cost',time_end-time_start,'s')
# print(create_vocabulary('./EasyFiles/', 1))