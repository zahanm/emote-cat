from datareaders import DataReader, KFoldDataReader
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
import feature_selection as fs

def run_nn(data):
    """
    ds = SupervisedDataSet(2, 1)
    ds.addSample((0, 0), (0,))
    ds.addSample((0, 1), (1,))
    ds.addSample((1, 0), (1,))
    ds.addSample((1, 1), (0,))
    """
    for fold in xrange(1, data.kfolds + 1):
        training_data = data.train(fold)
        test_data = data.test(fold)
        run_nn_fold(training_data, test_data)

def run_nn_fold(training_data, test_data):
    features, featureMap, labels, labelMap = fs.bernoulli(training_data)
    input_len = len(features[0])
    train_ds = SupervisedDataSet(input_len, 1)    
    for i in range(len(features)):
        train_ds.addSample(tuple(features[i]), (labels[i]))
    net = buildNetwork(input_len, 3, 1, bias=True, hiddenclass=TanhLayer)
    trainer = BackpropTrainer(net, train_ds)
    print "training until convergence..."
    trainer.trainUntilConvergence(maxEpochs=200)
    print "done. testing..."


    features, featureMap, labels, labelMap = fs.bernoulli(test_data)
    test_ds = SupervisedDataSet(input_len, 1)    
    for i in range(len(features)):
        print "adding", tuple(features[i]), (labels[i],)
        test_ds.addSample(tuple(features[i]), (labels[i]))

    print trainer.testOnClassData(test_ds)
        
