# models.py

import torch
import torch.nn as nn
from torch import optim as opt
import numpy as np
import random
from torch.utils import data
from sentiment_data import *


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str]) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]


class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str]) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1


class NeuralSentimentClassifier(SentimentClassifier):
    """
    Implement your NeuralSentimentClassifier here. This should wrap an instance of the network with learned weights
    along with everything needed to run it on new data (word embeddings, etc.)
    """

   # def __init__(self,model):



class deep_averaging_network(nn.Module):
    def __init__(self,inp ,hid ,out,word_embeddings):
        """
                :param inp: size of input (integer)
                :param hid: size of hidden layer(integer)
                :param out: size of output (integer), which should be the number of classes
                :param word_embeddings: set of loaded word embeddings from train_deep_averaging_network, that is generated from read_word_embeddings
                """
        super(deep_averaging_network, self).__init__()
        self.lin1 = nn.Linear(inp, hid)
        # self.g = nn.Tanh()
        self.rel1 = nn.ReLU()
        self.lin3 = nn.Linear(hid, hid)
        self.rel3 = nn.ReLU()
        self.lin4 = nn.Linear(hid, hid)
        self.w_e = word_embeddings
        self.rel2 = nn.ReLU()
        self.e = nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings.vectors), padding_idx=0)
        self.lin2 = nn.Linear(hid, hid)
        self.lin5 = nn.Linear(hid, out)
        # self.log_softmax = nn.LogSoftmax(dim=0)
        # Initialize weights according to a formula due to Xavier Glorot.
        # nn.init.xavier_uniform_(self.V.weight)
        # nn.init.xavier_uniform_(self.W.weight)
        # Initialize with zeros instead
        # nn.init.zeros_(self.V.weight)
        # nn.init.zeros_(self.W.weight)

    def feedForward(self, input):
        #call lin1
        #then relu function
        #then lin2
        #then relu again
        #then the third lin
        #then relu again
        #then the fourth lin
        #I'm getting the highest accuracy at 4 lins
        # Will try 5 if time permitted
        return self.lin5(self.rel1(self.lin4(self.rel1(self.lin3(self.rel1(self.lin2(self.rel1(self.lin1(input.float())))))))))


    def predict(self, words):
        #do the data preprocessing
        #feed it forward and call linear twice and relu once on it
        #then get the prediction from that result
        #then return it
        return (self.feedForward(self.dataPreprocessing(words)).max(0)[1])

    def predict_all(self, all_ex_words: List[List[str]]) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :return:
        """
        return [self.predict(ex_words) for ex_words in all_ex_words]

    def dataPreprocessing(self, words):
        # take actual sentence from training data
        # tuple for each sentence in train data
        # may want to implement lemmatization or stemming here, but I'm not sure yet
        indexStore = []
        # indices in indexStore correspond to words
        wordEmbeddings = []
        for word in words:
            # self.e.word return where in word embeddings it is
            # self.e is map that has word and its index
            idx = self.w_e.word_indexer.index_of(word)
            if idx == -1:
                indexStore.append(1)
            else:
                indexStore.append(idx)
        # maxLength = 0
        # for sentence in
        # need to pick a constant larger enough because I don't want to get length of longest sentence
        # in trainn && dev
        # if token is UNK or PAD add 0s
        # print(indexStore)
        # wordEmbeddings is the new sentences
        for index in indexStore:
            # tensor_name.numpy to translate tensor to numpy array
            indexTensor = torch.tensor(index)
            embeddingTensor = self.e(indexTensor)
            # converting to numpy to get average latet
            wordEmbeddingArray = embeddingTensor.numpy()
            wordEmbeddings.append(wordEmbeddingArray)
            # print(wordEmbeddings)
        return torch.mean(torch.tensor(wordEmbeddings), 0)

def batchify(x, size):
    #x is either train data or dev data
    #we want to take that data  ze
    #if size was 50 there would be batches of size 50
    #we will return a list of batches
    #where each batch is a list of tensors
    batchList = []
    i = 0
    j = 0
    lastBatchSize = int(len(x) % size)
    numBatchesMinusOne = int(len(x) / size)
    for j in range(numBatchesMinusOne):
        batchList.append(x[i:(i+size)])
        i = i+size
    if lastBatchSize > 1:
        batchList.append(x[i:])
    return batchList


def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings) -> NeuralSentimentClassifier:
    """
    :param args: Command-line args so you can access them here
    :param train_exs: training examples
    :param dev_exs: development set, in case you wish to evaluate your model during training
    :param word_embeddings: set of loaded word embeddings
    :return: A trained NeuralSentimentClassifier model
    """
    #https://androidkt.com/batch-size-step-iteration-epoch-neural-network/
    #The above lnk says that a batch size of 32 is a good start
    #also 32,64,128 and 256 could work
    #I'll start with 32 and go from there
    DAN = deep_averaging_network(300,64,2,word_embeddings)
    op = opt.Adam(DAN.parameters(), lr=.0005)
    lossFn = torch.nn.CrossEntropyLoss()
    trainData = []
    devData = []
    trainingAccuracy = []
    devAccuracy = []
    index = 0
    sumLosses = 0.0

    while index < len(train_exs):
        trainData.append((DAN.dataPreprocessing(train_exs[index].words), train_exs[index].label))
        index = index + 1
    index = 0
    while index < len(dev_exs):
        devData.append((DAN.dataPreprocessing(dev_exs[index].words), dev_exs[index].label))
        index = index + 1

    i = 0
    while i <= 10:
        random.shuffle(trainData)
        batched = batchify(trainData,16)
        for batch in batched:
            ##now need to get the label (0 or 1)
            ##and input from each batch
            labels = []
            input = []
            for x in batch:
                labels.append(x[1])
                input.append(np.array(x[0], dtype=np.float32))
            inputTensor = torch.tensor(input)
            labelTensor = torch.tensor(labels)
            DAN.zero_grad()
            prediction = DAN.feedForward(inputTensor)
            loss = lossFn(prediction, labelTensor)
            sumLosses = sumLosses + loss
            k = 0
            while k < len(batch):
                if(prediction[k].max(0)[1] != labelTensor[k]):
                    trainingAccuracy.append(0)
                    k = k+1
                else:
                    trainingAccuracy.append(1)
                    k = k+1
            loss.backward()
            op.step()
        random.shuffle(devData)
        batched = batchify(devData, 16)
        for batch in batched:
            ##now need to get the label (0 or 1)
            ##and input from each batch
            labels = []
            input = []
            for x in batch:
                #print(x[1])
                labels.append(x[1])
                input.append(np.array(x[0], dtype=np.float32))
            inputTensor = torch.tensor(input)
            labelTensor = torch.tensor(labels)
            #############################
            prediction = DAN.feedForward(inputTensor)
            k = 0
            while k < len(batch):
                if (prediction[k].max(0)[1] != labelTensor[k]):
                    devAccuracy.append(0)
                    k = k+1
                else:
                    devAccuracy.append(1)
                    k = k+1
        if(i % 2 == 0):
            avgTrainAcc = np.mean(trainingAccuracy)
            avgDevAcc = np.mean(devAccuracy)
            print(f"sumLosses = {sumLosses} @ epoch {i}")
            print(f"avgTrainAcc = {avgTrainAcc} @ epoch {i}")
            print(f"avgDevAcc = {avgDevAcc} @ epoch {i}")
            #op = opt.Adam(DAN.parameters(), lr=(lr-.001))
        trainingAccuracy.clear()
        devAccuracy.clear()
        i = i+1
        sumLosses = 0
    return DAN