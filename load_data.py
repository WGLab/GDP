import numpy as np
import gdp_model as dm
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import dtypes
import collections

Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


class DataSet(object):

    """Format the traing data
    """

    def __init__(self,
                 features,
                 dates,
                 censors,
                 #at_risks,
                 feature_groups,
                 seed=None):
        #TODO: check if seed work as expected
        seed1,seed2=random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)

        assert features.shape[0] == dates.shape[0],(
         'features.shape: %s dates.shape: %s' % (features.shape, dates.shape))
        self._patients_num=features.shape[0]
        self._features=features
        self._feature_groups=feature_groups
        self._dates=dates
        self._censors=censors
        #self._at_risks=at_risks
        self._epochs_completed=0
        self._index_in_epoch=0

    @property
    def features(self):
        return self._features
    @property
    def feature_size(self):
        return self._features.shape[1]

    @property
    def feature_groups(self):
        return self._feature_groups

    @property
    def dates(self):
        return self._dates

    @property
    def censors(self):
        return self._censors
#    @property
#    def at_risks(self):
#        return self._at_risks
    @property
    def patients_num(self):
        return self._patients_num

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size,shuffle=True):
        start=self._index_in_epoch
        # shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            index_perm=np.arange(self._patients_num)
            np.random.shuffle(index_perm)
            self._features=self._features[index_perm]
            self._dates=self._dates[index_perm]
            self._censors=self._censors[index_perm]
            #self._at_risks=self._at_risks[index_perm]
            #self._at_risks=[self._at_risks[k] for k in index_perm]
            #self._features,self._dates,self._censors,self._at_risks=dm.calc_at_risk(self._features[index_perm],self._dates[index_perm],self._censors[index_perm])

        # Go to the next epoch
        if start + batch_size > self._patients_num:
            # epoch completed
            self._epochs_completed+=1
            # get the remained ones
            rest_patients_num=self._patients_num - start
            features_rest_part=self._features[start:self._patients_num]
            dates_rest_part=self._dates[start:self._patients_num]
            censors_rest_part=self._censors[start:self._patients_num]
            #at_risks_rest_part=self._at_risks[start:self._patients_num]
            #shuffle the data
            if shuffle:
                index_perm1=np.arange(self._patients_num)
                np.random.shuffle(index_perm1)
                self._features=self._features[index_perm1]
                self._dates=self._dates[index_perm1]
                self._censors=self._censors[index_perm1]
                #self._at_risks=self._at_risks[index_perm1] # _at_risks is a python list
                #self._at_risks=[self._at_risks[k] for k in index_perm1]
            # start next epoch
            start = 0
            self._index_in_epoch=batch_size-rest_patients_num
            end=self._index_in_epoch
            features_new_part=self._features[start:end]
            dates_new_part=self._dates[start:end]
            censors_new_part=self._censors[start:end]
            #at_risks_new_part=self._at_risks[start:end]
            features_=np.concatenate((features_rest_part,features_new_part),axis=0)
            dates_=np.concatenate((dates_rest_part,dates_new_part),axis=0)
            censors_=np.concatenate((censors_rest_part,censors_new_part),axis=0)
            return dm.calc_at_risk(features_,dates_,censors_)
            #return np.concatenate((features_rest_part,features_new_part),axis=0),np.concatenate((dates_rest_part,dates_new_part),axis=0),np.concatenate((censors_rest_part,censors_new_part),axis=0),np.concatenate((at_risks_rest_part,at_risks_new_part),axis=0)
        else:
            self._index_in_epoch+=batch_size
            end=self._index_in_epoch
            #return self._features[start:end],self._dates[start:end],self._censors[start:end],self._at_risks[start:end]
            return dm.calc_at_risk(self._features[start:end],self._dates[start:end],self._censors[start:end])



def read_data_sets(train_dir="./example",
                   train_file="test_100.csv",
                   train_frac=0.6,
                   valid_frac=0.2,
                   test_frac=0.2,
                   seed=None,
                   shuffle_cval=True,
                   shuffle_all=False):
    """Read the input data into DataSet tuples

    Args:
        train_dir: the directory to the training data (where the validation and testing datasets chosen from)
        train_file: file name of the training data
        train_frac: fraction of the data used for training, default 60%
        valid_frac: fraction of the data used for validation, default 20%
        test_frac: fraction of the data used for testing, default 20%
        seed: seed used for numpy random number control

    Results:
        tuples of DataSet, containing three tuple, train, validation, and test

    """
    fileName=train_dir+"/"+train_file
    data=np.genfromtxt(fileName,dtype=float,missing_values="None",delimiter=",",skip_header=2)
    f=open(fileName)
    group=f.readline().rstrip().split(",")
    group=[int(x) for x in group]

    #TODO: Add types of censoring: eg. left truncation, right censoring, left censoring , unclear

    sample_size=data.shape[0]
    #whether to shuffle the input data
    if shuffle_all:
        seed1,seed2=random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        #sample_num=data.shape[0]
        index_perm=np.arange(sample_size)
        np.random.shuffle(index_perm)
        T=data[index_perm,-2] # dates
        O=data[index_perm,-1] #TODO: check censors and death relationship;  dealth = 1 - censors ?
        X=data[index_perm,:-2] # features
    else:
        T=data[:,-2] # dates
        O=data[:,-1] #TODO: check censors and death relationship;  dealth = 1 - censors ?
        X=data[:,:-2] # features
    #A: at_risk

    datasets={"train":{},"eval":{},"test":{}}
    index_s=0
    index_e=0
    if shuffle_cval:
        #shuffle the cross validation data sets
        train_sample_size=int(sample_size*train_frac)
        val_sample_size=int(sample_size*valid_frac)
        cval_num=train_sample_size+val_sample_size
        cval_num=int(cval_num)
        seed1,seed2=random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        index_perm=np.arange(cval_num)
        np.random.shuffle(index_perm)
        data1=data[index_perm,:] # get shuffled cross validation data sets
        train_index_s=0
        train_index_e=train_index_s+train_sample_size
        datasets["train"]['X']=data1[train_index_s:train_index_e,:-2]
        datasets["train"]['O']=data1[train_index_s:train_index_e,-1]
        datasets["train"]['T']=data1[train_index_s:train_index_e,-2]
        val_index_s=train_index_e
        val_index_e=cval_num
        datasets["eval"]['X']=data1[val_index_s:val_index_e,:-2]
        datasets["eval"]['O']=data1[val_index_s:val_index_e,-1]
        datasets["eval"]['T']=data1[val_index_s:val_index_e,-2]

        #keep test set the same
        test_index_s=cval_num
        test_index_e=sample_size
        datasets["test"]['X']=data[test_index_s:test_index_e,:-2]
        datasets["test"]['O']=data[test_index_s:test_index_e,-1]
        datasets["test"]['T']=data[test_index_s:test_index_e,-2]

    else:
        for dtype,frac in {"train":train_frac,"eval":valid_frac,"test":test_frac}.items():
            index_e=index_s+int(sample_size*frac)
            if(dtype=='test'):
                index_e=sample_size
            datasets[dtype]['X'],datasets[dtype]['T'],datasets[dtype]['O']=X[index_s:index_e],T[index_s:index_e],O[index_s:index_e]
            index_s=index_e

    train=DataSet(datasets["train"]['X'],datasets["train"]['T'],datasets["train"]['O'],group,seed)
    validation=DataSet(datasets["eval"]["X"],datasets["eval"]['T'],datasets["eval"]['O'],group,seed)
    test=DataSet(datasets["test"]["X"],datasets["test"]['T'],datasets["test"]["O"],group,seed)
    return Datasets(train=train,validation=validation,test=test)

def read_prediction_data(file_dir="./example",
                   file_name="test_100.csv",
                   seed=None):
    """Read the data for prediction into DataSet tuples

    Args:
        file_dir: the directory to where the prediction file is located
	file_name: the file name of the prediction file
        seed: seed used for numpy random number control

    Results:
        DataSet

    """
    fileName=file_dir+"/"+file_name
    data=np.genfromtxt(fileName,dtype=float,missing_values="None",delimiter=",",skip_header=2)
    f=open(fileName)
    group=f.readline().rstrip().split(",")
    group=[int(x) for x in group]

    #TODO: Add types of censoring: eg. left truncation, right censoring, left censoring , unclear

    sample_size=data.shape[0]
    T=data[:,-2] # dates
    O=data[:,-1] #TODO: check censors and death relationship;  dealth = 1 - censors ?
    X=data[:,:-2] # features
    pre_data=DataSet(X,T,O,group,seed)
    return pre_data
