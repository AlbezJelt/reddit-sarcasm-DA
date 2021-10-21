from enum import Enum
from math import inf
from operator import __eq__, __ne__, itemgetter
from typing import Any, Iterable, List, Tuple

from numpy import argmax, ndarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


def iter_together(l1: list, l2: list):
    assert len(l1) == len(l2), "List must have same length!"
    for i in range(len(l1)):
        yield (i, (l1[i], l2[i]))


def Extract(lst, index):
    return [item[index] for item in lst]


def list_pop(l: list) -> list:
    for excluded in range(len(l)):
        li = l[:]
        del li[excluded]
        yield li


class EnsambleStrategy(Enum):
    backward_highest_accuracy = 0
    backward_local_maximum = 1
    backward_ratio = 2


class ModelInfo:

    acc = -1
    err = -1
    __one_hot_encoder: OneHotEncoder = None
    __val_prediction__ = []

    def __init__(self, model, f1_mean, one_hot_encoder: OneHotEncoder = None):
        if 'keras' in model.__module__:
            assert one_hot_encoder, "Keras model requires one_hot_encoder parameter!"
            self.__one_hot_encoder = one_hot_encoder
        self.model = model
        self.f1 = f1_mean

    def __predict__(self, X):
        if 'sklearn' in self.model.__module__:
            return self.model.predict(X)
        elif 'keras' in self.model.__module__:
            # Models from functional API return softmax vectors of probabilities
            # needs to be converted to class prediction
            predicted = argmax(self.model.predict(X), axis=-1)
            return predicted

    def __predict_proba__(self, X):
        if 'keras' in self.model.__module__:
            return self.model.predict(X)
        elif 'sklearn' in self.model.__module__:
            # Sklearn classifiers use predict_proba
            return self.model.predict_proba(X)

    def __predict_val__(self, val_data):
        # val_data is a tuple (X_val, Y_val)
        __val_prediction = self.__predict__(val_data[0])
        self.__val_prediction__ = __val_prediction
        # Compute accuracy and error rate
        self.acc = accuracy_score(val_data[1], __val_prediction)
        self.err = 1 - self.acc


class BMA_Ensamble:

    __val_data: Tuple[Any, Any] = None
    model_infos: List[ModelInfo] = None
    ACC = float(inf)

    def __init__(self, validation_data: Tuple[Any, Any], model_infos: Iterable[ModelInfo] = []):
        self.__val_data = validation_data
        self.model_infos = model_infos
        self.ensamble_mask = list()
        for model_id in range(len(model_infos)):
            self.__predict_validation(model_id)

    # Returns the number of instances (mis, __ne__)classified(__eq__) by i
    # over the number of instances (mis, __ne__)classified(__eq__) by j
    def __p_ij(self, model_i, op_i, model_j, op_j):

        # Instances by index (mis, __ne__)classified(__eq__) by j
        model_j_x = [index for index, (real, predicted) in iter_together(
            self.__val_data[1], self.model_infos[model_j].__val_prediction__) if op_j(real, predicted)]

        # Instances by index (mis, __ne__)classified(__eq__) by i over model_j_x
        model_i_x = [index for index in model_j_x if op_i(
            self.__val_data[1][index], self.model_infos[model_i].__val_prediction__[index])]

        return len(model_i_x) / len(model_j_x)
        # return len(model_i_x)

    def __contribution(self, classifier: int, current_ensamble: List[int]) -> float:
        correct = sum((self.__f_correct(classifier, j)
                       for j in current_ensamble if j != classifier))
        incorrect = sum((self.__f_incorrect(classifier, j)
                         for j in current_ensamble if j != classifier))
        return correct / incorrect

    def __f_correct(self, classifier, j): return (self.__p_ij(classifier, __eq__, j, __eq__) *
                                                  self.model_infos[j].acc) + (self.__p_ij(classifier, __eq__, j, __ne__) * self.model_infos[j].err)

    def __f_incorrect(self, classifier, j): return (self.__p_ij(classifier, __ne__, j, __eq__) *
                                                    self.model_infos[j].acc) + (self.__p_ij(classifier, __ne__, j, __ne__) * self.model_infos[j].err)

    def __local_maxima(self) -> List[int]:
        current_ensamble = list(range(len(self.model_infos)))
        def calc_ACC(ensamble): return sum(
            [self.__contribution(i, ensamble) for i in ensamble]) / len(ensamble)

        # Calculate starting Average Classifier Contribution
        self.ACC = calc_ACC(current_ensamble)

        while len(current_ensamble) > 2:
            # Backward step
            bACCs = [[next_ensamble, calc_ACC(next_ensamble)]
                     for next_ensamble in list_pop(current_ensamble)]
            best_bACC = max(bACCs, key=lambda ens: ens[1])

            # If ACC cannot optimize anymore
            if best_bACC[1] < self.ACC:
                break  # Stop the search

            current_ensamble = best_bACC[0]
            self.ACC = best_bACC[1]

        return current_ensamble

    def __predict_validation(self, model_id: int):
        self.model_infos[model_id].__predict_val__(self.__val_data)

    def create_ensamble(self, mode: EnsambleStrategy):
        if mode == EnsambleStrategy.backward_local_maximum:
            self.ensamble_mask = self.__local_maxima()

    def add_model(self, model_infos: Iterable[ModelInfo]):
        for model_id in range(len(model_infos)):
            self.model_infos.append(model_infos[model_id])
            self.__predict_validation(model_id)

    def predict(self, X):
        def __predict__(self, sample): return [ndarray.flatten(model.__predict_proba__(
            [sample]) * model.f1 * 0.5) for model in itemgetter(*self.ensamble_mask)(self.model_infos)]
        prediction_proba = ([sum(Extract(prediction, 0)), sum(Extract(
            prediction, 1))] for prediction in (__predict__(self, sample) for sample in X))
        prediction = [argmax(prediction, axis=-1)
                      for prediction in prediction_proba]
        return prediction

    def predict_proba(self, X):
        def __predict__(self, sample): return [ndarray.flatten(model.__predict_proba__(
            [sample]) * model.f1 * 0.5) for model in itemgetter(*self.ensamble_mask)(self.model_infos)]
        prediction_proba = ([sum(Extract(prediction, 0)), sum(Extract(
            prediction, 1))] for prediction in (__predict__(self, sample) for sample in X))
        return list(prediction_proba)
