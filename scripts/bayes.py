import nltk

word_features = set()

def document_features(feat_list): 
    document_words = set(feat_list) 
    features = {}
    
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def run(fold, data, result):
    for line in data.all():
        for feat in line:
            word_features.add(feat)

    training_data = data.train(fold)
    featuresets = [(document_features(x["Features"]), x["Answer"]) for x in training_data]
    train_set, test_set = featuresets[500:], featuresets[:500]


    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(5)

    classifier = nltk.DecisionTreeClassifier.train(train_set)
    print nltk.classify.accuracy(classifier, test_set)
