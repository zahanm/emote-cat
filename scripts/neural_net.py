from datareaders import DataReader, KFoldDataReader
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.datasets import SupervisedDataSet, ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer, SoftmaxLayer
from pybrain.supervised.trainers import BackpropTrainer
import feature_selection as fs
from pybrain.utilities import percentError


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
    test_features, featureMap, labels, labelMap = fs.bernoulli(training_data)

    input_len = len(test_features[0])
    num_classes = len(labelMap.keys())
    train_ds = ClassificationDataSet(input_len, 1,nb_classes=num_classes)
    for i in range(len(test_features)):
        train_ds.addSample(tuple(test_features[i]), (labels[i]))
    train_ds._convertToOneOfMany()
    net = buildNetwork(train_ds.indim, 3, train_ds.outdim, bias=True, outclass=SoftmaxLayer)
    trainer = BackpropTrainer(net, train_ds)
    print "training until convergence..."
    trainer.trainUntilConvergence(maxEpochs=200)
    print "done. testing..."


    test_ds = ClassificationDataSet(input_len, 1,nb_classes=num_classes)  

    for tweetinfo in test_data:
        featuresFound = tweetinfo["Tweet"]
        label = tweetinfo["Answer"]
        features = [0]*len(featureMap)
        for feat in featuresFound:
            if feat in featureMap:
                features[ featureMap[feat] ] = 1
        test_ds.addSample(tuple(features), (labelMap[label]))

    test_ds._convertToOneOfMany()
    tstresult = percentError( trainer.testOnClassData(
            dataset=test_ds ), test_ds['class'] )
    print tstresult

    
