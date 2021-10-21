from BMA import ModelInfo, BMA_Ensamble, EnsambleStrategy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from pprint import pprint

classifiers = pickle.load(open('data/classifiers.p', 'rb'))
f1_scores = pickle.load(open('data/f1_scores.p', 'rb'))
x_val, y_val = pickle.load(open('data/val.p', 'rb'))

MIs = []
for index, classifier in enumerate(classifiers):
    MIs.append(ModelInfo(classifier, f1_scores[index]))

BMA_full = BMA_Ensamble((x_val, y_val), MIs)

BMA_full.create_ensamble(EnsambleStrategy.backward_local_maximum)

accuracy = dict()
for index, classifier in enumerate(classifiers):
    y_pred = classifier.predict(x_val)
    class_name = str(classifier.__class__).replace("<class '", "").replace("'>", "").split(".")[-1]
    accuracy[f"{index} - {class_name}"] = {
        'accuracy' : accuracy_score(y_val, y_pred), 
        'precision' : precision_score(y_val, y_pred),
        'recall' : recall_score(y_val, y_pred),
        'f1' : f1_score(y_val, y_pred)
    }

bma_pred = BMA_full.predict(x_val)
accuracy[f"BMA"] = {
    'accuracy' : accuracy_score(y_val, bma_pred), 
    'precision' : precision_score(y_val, bma_pred),
    'recall' : recall_score(y_val, bma_pred),
    'f1' : f1_score(y_val, bma_pred)
}

# accuracy = dict(sorted(accuracy.items(), key=lambda item: item[1]['accuracy']))

pprint(accuracy)