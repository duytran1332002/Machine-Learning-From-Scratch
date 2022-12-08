import torch
import numpy as np
from scipy.stats import mode
from collections import Counter
from tqdm.notebook import tqdm


# CLASSIFICATION MODEL

# KNN model - Binary AND Multi-class - Euclidean, Manhattan, Minkowski - using torch
class KNN():
  def __init__(self, X_train, y_train, k , distance = 'euclidean', p = 2):
    # k is number of neightbors
    self.k = k
    self.distance = distance
    self.p = p
    self.X_train = X_train
    self.y_train = y_train

  def calc_distance(self, point1, point2):
    point1 = torch.tensor(point1, dtype=torch.float32).reshape((1, len(point2[0])))
    point2 = torch.tensor(point2, dtype=torch.float32)
    if self.distance == "euclidean":
      return torch.cdist(point1, point2, p = 2)
    elif self.distance == "mahattan":
      return torch.cdist(point1, point2, p = 1)
    elif self.distance == "minkowski":
      return torch.cdist(point1, point2, p = self.p)

  def fit_predict(self, item):
    """
      - Find distance between item and data (item/X_train)
        using D distance with D in [euclidean, manhattan, minkowski]
      - Sort the distance using argsort, it gives indices of the 
      - Find the majority label whose distance closest to each datapoint of y_train.

      item: tensors to be classified.
      return: predicted labels.
    """
    
    distance_score = self.calc_distance(item, self.X_train)
    k_neighbors = distance_score.argsort()[:, :self.k]
    neighbors_label = self.y_train[k_neighbors]
    nearest_class = mode(neighbors_label).mode[:, 0].item()
    return nearest_class

  def evaluate(self, X_test, y_test):
    test_predict = []
    for item in X_test:
      predict = self.fit_predict(item)
      test_predict.append(predict)
    test_predict = torch.tensor(test_predict)
    y_test = torch.tensor(y_test)
    return torch.sum(test_predict == y_test).item() / len(y_test) 

# Logistic Regression model - Binary AND Multi-class - using torch
class LogisticRegression():
  def __init__(self, batch_size, n_feature, type, num_class = None):
    self.batch_size = batch_size
    if type == "binary":
      self.theta = torch.rand((n_feature))
    elif type == "multi":
      self.theta = torch.rand((n_feature, num_class))
    self.type = type
    
  def sigmoid(self, z):
    return 1 / (1 + torch.exp(-z))

  def softmax(self, z):
    return torch.exp(z) / torch.exp(z).sum(axis=1, keepdims=True)
  

  def findBinaryLossEntropy(self, y, y_hat):
    y_hat = torch.clip(y_hat, 1e-6, 1 - 1e-6)
    return (- y * torch.log(y_hat) - (1 - y) * torch.log(1 - y_hat)).mean()
  
  def findCrossEntropyLoss(self, y, y_hat):
    y_hat = torch.clip(y_hat, 1e-6, 1 - 1e-6)
    return (- y * torch.log(y_hat)).mean()

  def gradient(self, X, y, y_hat):
    return torch.matmul(X.T, (y_hat - y)) / len(y)

  def predict(self, X):
    z = torch.matmul(X, self.theta)
    if self.type == "binary":
      return self.sigmoid(z)
    elif self.type == "multi":
      return self.softmax(z)
  
  def loss(self, y, y_hat):
    if self.type == "binary":
      return self.findBinaryLossEntropy(y, y_hat)
    elif self.type == "multi":
      return self.findCrossEntropyLoss(y, y_hat)

  def getBatchData(self, X, y, number_batch_size, last_size, ith_batch):
    if ith_batch == number_batch_size - 1 and last_size != 0:
      X_batch = X[ith_batch * self.batch_size : (ith_batch * self.batch_size) + last_size]
      y_batch = y[ith_batch * self.batch_size : (ith_batch * self.batch_size) + last_size]
    else:
      X_batch = X[ith_batch * self.batch_size : (ith_batch + 1) * self.batch_size]
      y_batch = y[ith_batch * self.batch_size : (ith_batch + 1) * self.batch_size]

    return X_batch, y_batch
  
  def accuracy(self, y_train, y_hat):
    y_hat = torch.round(y_hat)
    return (y_train == y_hat).float().mean()

  def fit(self, X_train, y_train, X_val, y_val, n_iters, learning_rate):
    history = {"losses_train": [], "accs_train": [], "losses_val": [], "accs_val": []}
    last_size = 0

    if len(X_train) % self.batch_size == 0:
      number_batch_size = int(len(X_train) / self.batch_size)
    else:
      number_batch_size = int(len(X_train) / self.batch_size) + 1
      last_size = len(X_train) - self.batch_size * (int(len(X_train) / self.batch_size))

    for epoch in tqdm(range(n_iters)):
      for ith_batch in range(0, number_batch_size):
        # get X_batch, y_batch
        X_train_batch, y_train_batch = self.getBatchData(X_train, y_train, number_batch_size, last_size, ith_batch)
        # predict
        y_train_batch_hat = self.predict(X_train_batch)

        # compute loss
        loss = self.loss(y_train_batch, y_train_batch_hat)

        # calculate the gradient
        gradient = self.gradient(X_train_batch, y_train_batch, y_train_batch_hat)
        
        # update theta
        self.theta -= learning_rate * gradient
        # for debug
        # calculate loss and accuracy of train sets
        history['losses_train'].append(loss)

        y_train_hat = self.predict(X_train)
        acc = self.accuracy(y_train, y_train_hat)
        history['accs_train'].append(acc)

        # calculate loss and accuracy of validation sets
        y_val_hat = self.predict(X_val)
        loss_val = self.loss(y_val, y_val_hat)
        history['losses_val'].append(loss_val)
        acc_val = self.accuracy(y_val, y_val_hat)
        history['accs_val'].append(acc_val)

      if (epoch + 1) % 10 == 0:
        print("Epoch: {}, loss: {}, acc: {}, loss_val: {}, acc_val: {}".format(epoch + 1, loss, acc, loss_val, acc_val))
    return history

# Bayes Classifier - Bernoulli and Gaussian - using numpy
class NaiveBayesClassifier():
    def __init__(self, typeBayes):

        self.features = list()
        self.prior = {}
        self.likelihood = {}
        self.pred_prior = {}
        
        self.train_size = int
        self.num_feats = int
        self.X_train = np.array
        self.y_train = np.array

        self.typeBayes = typeBayes

    def setClassPrior(self):
        """ P(c) - Prior class probability P(c) = count(c) / count(all) """
        for c in np.unique(self.y_train):
            self.prior[c] = np.sum(self.y_train == c) / self.train_size

    def setLikelihoodBernoulli(self):
        """ P(x|c) - Likelihood of feature given class P(x|c) = count(x|c) / count(c) """
        for feature in self.features:
            for c in np.unique(self.y_train):
                feat_likelihood = self.X_train[feature][self.y_train[self.y_train == c].index].value_counts().to_dict()
                for feat_val, count in feat_likelihood.items():
                    self.likelihood[feature][feat_val + "|" + c] = count / np.sum(self.y_train == c)
    
    def calculateGaussian(self, query, feature, c):
        """ Gaussian distribution"""
        return (1 / (np.sqrt(2 * np.pi) * self.likelihood[feature][c]['std'])) * np.exp(-0.5 * ((query - self.likelihood[feature][c]['mean']) / self.likelihood[feature][c]['std']) ** 2)
    
    def  setLikelihoodGaussian(self):
        """ P(x|c) - Gaussian distribution of feature given class P(x|c) = count(x|c) / count(c) """
        for feature in self.features:
            for c in np.unique(self.y_train):
                mean = np.mean(self.X_train[feature][self.y_train[self.y_train == c].index])
                std = np.std(self.X_train[feature][self.y_train[self.y_train == c].index])
                self.likelihood[feature][c]['mean'] = mean
                self.likelihood[feature][c]['std'] = std


    def setPredictPrior(self):
        for feature in self.features:
            feat_val = self.X_train[feature].value_counts().to_dict()

            for val, count in feat_val.items():
                self.pred_prior[feature][val] = count / self.train_size

    def setBernoulliBayes(self):
        for feature in self.features:
            self.likelihood[feature] = {}
            self.pred_prior[feature] = {}
            for feat_val in np.unique(self.X_train[feature]):
                self.pred_prior[feature].update({feat_val: 0})
                for c in np.unique(self.y_train):
                    self.likelihood[feature].update({feat_val+'|'+c:0})
                    self.prior.update({c: 0})
        self.setClassPrior()
        self.setLikelihoodBernoulli()
        self.setPredictPrior()

    def setGaussianBayes(self):
        for feature in self.features:
            self.likelihood[feature] = {}
            self.pred_prior[feature] = {}
            for c in np.unique(self.y_train):
                self.likelihood[feature].update({c:{}})
                self.prior.update({c: 0})

        self.setClassPrior()
        self.setLikelihoodGaussian()
        
    def predictBernoulliNaiveBayes(self, X):
        result = []
        X = np.array(X)
        for query in X:
            probs_outcome = {} # P(c|x) probability of outcome given query
            for c in np.unique(self.y_train):
                priors = self.prior[c]
                likelihood = 1
                evidence = 1
                for feature, feat_val in zip(self.features, query):
                   likelihood *= self.likelihood[feature][feat_val + "|" + c] # P(A, B | C) = P(A | C) * P(B | C)
                   evidence *= self.pred_prior[feature][feat_val]
                probs_outcome[c] = (likelihood * priors) / (evidence)
            result.append(max(probs_outcome, key=probs_outcome.get)) 
        return result

    def predictGaussianNaiveBayes(self, X):
        result = []
        X = np.array(X)
        for query in X:
            probs_outcome = {} # P(c|x) probability of outcome given query
            for c in np.unique(self.y_train):
                likelihood = 1
                evidence = 1
                probs_outcome[c] = self.prior[c]
                for feature, feat_val in zip(self.features, query):
                   likelihood *=  self.calculateGaussian(feat_val, feature, c) # P(A, B | C) = P(A | C) * P(B | C)
                probs_outcome[c] *= likelihood
            result.append(max(probs_outcome, key=probs_outcome.get)) 
        return result

    def fit(self, X, y):
        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        if self.typeBayes == 'Bernoulli':
            self.setBernoulliBayes()
        elif self.typeBayes == 'Gaussian':
            self.setGaussianBayes()

    def predict(self, X):
        if self.typeBayes == 'Bernoulli':
            return self.predictBernoulliNaiveBayes(X)
        elif self.typeBayes == 'Gaussian':
            return self.predictGaussianNaiveBayes(X)

    def findAccuracyScore(self, y_true, y_pred):
        """	score = (y_true - y_pred) / len(y_true) """
        return round(float(sum(y_pred == y_true))/float(len(y_true)) * 100 ,2)

    def printPredict(self, query):
        print("Query:- {} ---> {}".format(query, self.predict(query)))

# Decision Tree Classifier - gini and entropy - using torch

class Node():
    def __init__(self, feature=None, threshold=None, data_left=None, data_right=None, gain=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.data_left = data_left
        self.data_right = data_right
        self.gain = gain
        self.value = value


class DecisionTree():
    def __init__(self, min_sample=2, max_depth=5, impurity_function=None):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.root = None

    def _entropy(self, y):
        _, counts = torch.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return -(probs * torch.log2(probs)).sum()

    def _information_gain(self, y, y_left, y_right):
        p = y_left.shape[0] / y.shape[0]
        return self._entropy(y) - p * self._entropy(y_left) - (1 - p) * self._entropy(y_right)

    def _gini(self, y):
        _, counts = torch.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - (probs ** 2).sum()

    def _gini_gain(self, y, y_left, y_right):
        p = y_left.shape[0] / y.shape[0]
        return self._gini(y) - p * self._gini(y_left) - (1 - p) * self._gini(y_right)

    def _calc_gain(self, y, y_left, y_right):
        if self.impurity_function == "gini":
            return self._gini_gain(y, y_left, y_right)
        elif self.impurity_function == "entropy":
            return self._information_gain(y, y_left, y_right)

    def _best_split(self, X, y):
        best_gain = -1
        best_split = {}

        for feature in range(X.shape[1]):
            X_crr = X[:, feature]

            for threshold in torch.unique(X_crr):
                df = torch.concat((X, y.reshape(1, -1).T), dim=1)
                df_left = df[df[:, feature] <= threshold]
                df_right = df[df[:, feature] > threshold]
                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -1]
                    gain = self._calc_gain(y, df_left[:, -1], df_right[:, -1])
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {"feature": feature,
                                      "threshold": threshold,
                                      "data_left": df_left,
                                      "data_right": df_right,
                                      "gain": gain}
        return best_split

    def _build(self, X, y, depth=0):
        if X.shape[0] >= self.min_sample and depth <= self.max_depth:
            best = self._best_split(X, y)
            try:
                if best['gain'] > 0:
                    left = self._build(
                        best["data_left"][:, :-1], best["data_left"][:, -1], depth + 1)
                    right = self._build(
                        best["data_right"][:, :-1], best["data_right"][:, -1], depth + 1)
                    return Node(feature=best["feature"],
                                threshold=best["threshold"],
                                data_left=left,
                                data_right=right,
                                gain=best["gain"])
            except:
                pass
                #print("Can't split data at some branch, please check your min sample and max depth to have a better result")
        return Node(
            value=Counter(y).most_common(1)[0][0]
        )

    def fit(self, X, y):
        X = torch.tensor(X)
        y = torch.tensor(y)
        self.root = self._build(X, y)

    def _predict(self, X, tree):
        # return leaf value if we are at a leaf node
        if tree.value is not None:
            return tree.value
        # traverse the tree
        feature = tree.feature
        threshold = tree.threshold
        if X[feature] < threshold:
            # go left
            return self._predict(X, tree.data_left)
        # go right
        return self._predict(X, tree.data_right)

    def predict(self, X):
        return torch.tensor([self._predict(x, self.root) for x in X])

    def print_tree(self, current_node, list_feature, nameattr='feature', left_child='data_left', right_child='data_right', indent='', last='updown'):

        if hasattr(current_node, str(nameattr)):
            def name(node): return list_feature[getattr(node, nameattr)] + ", " + str(round(node.threshold.item(), 2)) if getattr(node, nameattr) is not None else getattr(node, "value")
        else:
            def name(node): return str(node)

        up = getattr(current_node, left_child)
        down = getattr(current_node, right_child)

        if up is not None:
            next_last = 'up'
            next_indent = '{0}{1}{2}'.format(
                indent, ' ' if 'up' in last else '|', ' ' * len(str(name(current_node))))
            self.print_tree(up, list_feature, nameattr, left_child,
                    right_child, next_indent, next_last)

        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'

        if up is not None and down is not None:
            end_shape = '┤'
        elif up:
            end_shape = '┘'
        elif down:
            end_shape = '┐'
        else:
            end_shape = ''

        print('{0}{1}{2}{3}'.format(
            indent, start_shape, name(current_node), end_shape))

        if down is not None:
            next_last = 'down'
            next_indent = '{0}{1}{2}'.format(
                indent, ' ' if 'down' in last else '|', ' ' * len(str(name(current_node))))
            self.print_tree(down, list_feature, nameattr, left_child,
                    right_child, next_indent, next_last)

# Random Forest Classifier - gini and entropy - using torch
class RandomForest():
    def __init__(self, n_trees=10, min_sample=2, max_depth=5, impurity_function="entropy"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.impurity_function = impurity_function
        # Store all decisions trees
        self.forest = []
    def _sample(self, X, y):
        '''
        Helper function used for boostrap sampling.
        
        :param X: np.array, features
        :param y: np.array, target
        :return: tuple (sample of features, sample of target)
        '''
        n_samples = X.shape[0]
        idxs = torch.randint(0, n_samples, size=(n_samples,))
        return X[idxs], y[idxs]

    def fit(self, X, y):
        '''
        Trains a Random Forest classifier.
        
        :param X: np.array, features
        :param y: np.array, target
        :return: None
        '''
        
        # reset forest
        if len(self.forest) > 0:
            self.forest = []

        for _ in range(self.n_trees):
            try:
                X_sample, y_sample = self._sample(X, y)
                tree = DecisionTree(min_sample=self.min_sample, max_depth=self.max_depth, impurity_function=self.impurity_function)
                tree.fit(X_sample, y_sample)
                self.forest.append(tree)
            except Exception as e:
                continue
    
    def predict(self, X):
        '''
        Predicts the target for a given set of features.
        
        :param X: np.array, features
        :return: np.array, predicted target
        '''
        y_pred = []
        for tree in self.forest:
            y_pred.append(tree.predict(X))
        y_pred = torch.stack(y_pred)
        return torch.mode(y_pred, dim=0)[0]

# AdaBoost Classifier - gini and entropy - using torch
class AdaBoost():
    def __init__(self, X, y, num_stump = 100):
        self.num_stump = num_stump
        self.stump_list = []
        self.alpha_list = []
        self.training_error_list = []
        self.X = torch.clone(X)
        self.y = torch.clone(y)
        self.w_i = torch.ones(len(y), dtype=torch.float32)  / len(y)
        
    def _compute_error(self, y, y_pred):
        return (torch.sum(self.w_i * (torch.not_equal(y, y_pred)).int())) / torch.sum(self.w_i)
    
    def _compute_alpha(self, error):
        return 0.5 * torch.log((1 - error) / error)
    
    def _update_w(self, w_i, alpha, y, y_pred):
        return w_i * torch.exp(alpha * (torch.not_equal(y, y_pred)).int())
    
    def fit(self):
        for m in range(self.num_stump):            
            # find a weak classifier and predict label
            stump = DecisionTree(max_depth = 0, impurity_function = "entropy")
            stump.fit(self.X, self.y)
            y_pred = torch.tensor(stump.predict(self.X))
            
            self.stump_list.append(stump) # Save the model to the list

            # compute the error
            error = self._compute_error(self.y, y_pred)
            self.training_error_list.append(error)

            # compute alpha - amount of say of each stump
            alpha = self._compute_alpha(error)
            self.alpha_list.append(alpha)

            # update the weight
            self.w_i = self._update_w(self.w_i, alpha, self.y, y_pred)

            # update dataset for next stump pay more attention to the misclassified data
            idx = torch.multinomial(self.w_i, len(self.y), replacement=True)
            self.X = self.X[idx]
            self.y = self.y[idx]

        assert len(self.stump_list) == len(self.alpha_list) == len(self.training_error_list)
    
    def predict(self, X):
        X = torch.tensor(X)
        y_pred = torch.zeros(len(X), dtype=torch.float32)
        for i in range(len(self.stump_list)):
            y_pred += self.alpha_list[i] * self.stump_list[i].predict(X)
        y_pred = torch.sign(y_pred)
        y_pred[y_pred < 0] = 0
        return y_pred

# XGBoost Classifier - gini and entropy - using torch
# XGBoost Tree
class XGBoostTree():
    def __init__(self, lambda_ = 1, gamma = 0 , min_sample=2, max_depth=5, impurity_function=None):
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.impurity_function = impurity_function
        self.lambda_ = lambda_
        self.gamma = gamma
        self.root = None
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def _compute_negative_gradient(self, y, p):
        return p - y
    
    def _compute_hessian(self, p):
        return p * (1 - p)
    
    def gain(self, sum_gradient, sum_hessian):
        return 0.5 * (sum_gradient ** 2 / (sum_hessian + self.lambda_))
    
    def _compute_gain(self, y, y_base_pred_left, y_base_pred_right, y_left, y_right):
        p_left = self.sigmoid(y_base_pred_left)
        sum_gradient_left = self._compute_negative_gradient(y_left, p_left).sum()
        sum_hessian_left = self._compute_hessian(p_left).sum()

        p_right = self.sigmoid(y_base_pred_right)
        sum_gradient_right = self._compute_negative_gradient(y_right, p_right).sum()
        sum_hessian_right = self._compute_hessian(p_right).sum()

        gain = (self.gain(sum_gradient_left, sum_hessian_left) + self.gain(sum_gradient_right, sum_hessian_right) 
                - self.gain((sum_gradient_left + sum_gradient_right), (sum_hessian_left + sum_hessian_right)) ) + self.gamma
        return gain



    def _best_split(self, X, y, y_base_pred):
        best_gain = -1
        best_split = {}

        for feature in range(X.shape[1]):
            X_crr = X[:, feature]

            for threshold in torch.unique(X_crr):
                df = torch.concat((X, y.reshape(1, -1).T, y_base_pred.reshape(1, -1).T), dim=1)
                df_left = df[df[:, feature] <= threshold]
                df_right = df[df[:, feature] > threshold]
                if len(df_left) > 0 and len(df_right) > 0:
                    y = df[:, -2]
                    gain = self._compute_gain(y, df_left[:, -1], df_right[:, -1], df_left[:, -2], df_right[:, -2])
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {"feature": feature,
                                      "threshold": threshold,
                                      "data_left": df_left,
                                      "data_right": df_right,
                                      "gain": gain}
        return best_split

    def _build(self, X, y, y_base_pred, depth=0):
        if X.shape[0] >= self.min_sample and depth <= self.max_depth:
            best = self._best_split(X, y, y_base_pred)
            try:
                if best['gain'] > 0:
                    left = self._build(
                        best["data_left"][:, :-2], best["data_left"][:, -2], best['data_left'][:, -1], depth + 1)
                    right = self._build(
                        best["data_right"][:, :-2], best["data_right"][:, -2], best['data_right'][:, -1], depth + 1)
                    return Node(feature=best["feature"],
                                threshold=best["threshold"],
                                data_left=left,
                                data_right=right,
                                gain=best["gain"])
            except:
                pass
        # compute leaf value
        grad = self._compute_negative_gradient(y, self.sigmoid(y_base_pred)).sum()
        hess = self._compute_hessian(self.sigmoid(y_base_pred)).sum()
        leaf_value = - grad / (hess + self.lambda_)
        return Node(

            value=leaf_value
        )

    def fit(self, X, y, y_base_pred):
        X = torch.tensor(X)
        y = torch.tensor(y)
        self.root = self._build(X, y, y_base_pred)

    def _predict(self, X, tree):
        # return leaf value if we are at a leaf node
        if tree.value is not None:
            return tree.value
        # traverse the tree
        feature = tree.feature
        threshold = tree.threshold
        if X[feature] < threshold:
            # go left
            return self._predict(X, tree.data_left)
        # go right
        return self._predict(X, tree.data_right)

    def predict(self, X):
        return torch.tensor([self._predict(x, self.root) for x in X])

    def print_tree(self, current_node, list_feature, nameattr='feature', left_child='data_left', right_child='data_right', indent='', last='updown'):

        if hasattr(current_node, str(nameattr)):
            def name(node): return list_feature[getattr(node, nameattr)] + ", " + str(round(node.threshold.item(), 2)) if getattr(node, nameattr) is not None else getattr(node, "value")
        else:
            def name(node): return str(node)

        up = getattr(current_node, left_child)
        down = getattr(current_node, right_child)

        if up is not None:
            next_last = 'up'
            next_indent = '{0}{1}{2}'.format(
                indent, ' ' if 'up' in last else '|', ' ' * len(str(name(current_node))))
            self.print_tree(up, list_feature, nameattr, left_child,
                    right_child, next_indent, next_last)

        if last == 'up':
            start_shape = '┌'
        elif last == 'down':
            start_shape = '└'
        elif last == 'updown':
            start_shape = ' '
        else:
            start_shape = '├'

        if up is not None and down is not None:
            end_shape = '┤'
        elif up:
            end_shape = '┘'
        elif down:
            end_shape = '┐'
        else:
            end_shape = ''

        print('{0}{1}{2}{3}'.format(
            indent, start_shape, name(current_node), end_shape))

        if down is not None:
            next_last = 'down'
            next_indent = '{0}{1}{2}'.format(
                indent, ' ' if 'down' in last else '|', ' ' * len(str(name(current_node))))
            self.print_tree(down, list_feature, nameattr, left_child,
                    right_child, next_indent, next_last)
# XGBoost
class XGBoost():
    def __init__(self):
        self.trees = []
    
    def fit(self, X, y, learning_rate, n_estimators=100, max_depth=5, min_sample=2, gamma=0, lambda_=1):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.gamma = gamma
        self.lambda_ = lambda_

        y_base_pred = torch.ones(y.shape[0]) 
        for i in range(self.n_estimators):
            booster = XGBoostTree(max_depth=self.max_depth, min_sample=self.min_sample, gamma=self.gamma, lambda_=self.lambda_)
            booster.fit(X, y, y_base_pred)
            self.trees.append(booster)
            y_base_pred = y_base_pred + learning_rate * booster.predict(X)
    
    def predict_prob(self, X):
        pred = torch.zeros(len(X))
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return torch.sigmoid(pred)

    def predict(self, X):
        pred = torch.zeros(len(X))
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return torch.round(torch.sigmoid(pred))

# REGRESSION MODEL

# Linear Regression - mse and mae - using torch
class LinearRegression():
    def __init__(self, n_feature, batch_size, loss_type = "mse"):
        self.batch_size = batch_size
        self.theta = torch.rand((n_feature))
        self.loss_type = loss_type
    
    def getBatchData(self, X, y, number_batch_size, last_size, ith_batch):
        if ith_batch == number_batch_size - 1 and last_size != 0:
            X_batch = X[ith_batch * self.batch_size : (ith_batch * self.batch_size) + last_size]
            y_batch = y[ith_batch * self.batch_size : (ith_batch * self.batch_size) + last_size]
        else:
            X_batch = X[ith_batch * self.batch_size : (ith_batch + 1) * self.batch_size]
            y_batch = y[ith_batch * self.batch_size : (ith_batch + 1) * self.batch_size]

        return X_batch, y_batch
    
    def predict(self, X):
        y_hat = torch.matmul(X, self.theta)
        return y_hat

    def loss(self, y, y_hat):
        if self.loss_type == "mse":
            loss = torch.sum((y_hat - y) ** 2) / len(y)
        elif self.loss_type == "mae":
            loss =torch.sum(torch.abs(y_hat - y)) / len(y)
        return loss
    
    def gradient(self, X, y, y_hat):
        if self.loss_type == "mse":
            gradient = torch.matmul(X.T, (y_hat - y)) / len(X)
        elif self.loss_type == "mae":
            gradient = torch.matmul(X.T, (y_hat - y) / torch.abs(y_hat - y)) / len(X)
        return gradient

    def fit(self, X_train, y_train, X_val, y_val, n_iters, learning_rate):
        history = {"losses_train": [], "losses_val": []}
        last_size = 0

        if len(X_train) % self.batch_size == 0:
            number_batch_size = int(len(X_train) / self.batch_size)
        else:
            number_batch_size = int(len(X_train) / self.batch_size) + 1
            last_size = len(X_train) - self.batch_size * (int(len(X_train) / self.batch_size))

        for epoch in range(n_iters):
            for ith_batch in range(0, number_batch_size):
                # get X_batch, y_batch
                X_train_batch, y_train_batch = self.getBatchData(X_train, y_train, number_batch_size, last_size, ith_batch)
                # predict
                y_train_batch_hat = self.predict(X_train_batch)

                # compute loss
                loss = self.loss(y_train_batch, y_train_batch_hat)

                # calculate the gradient
                gradient = self.gradient(X_train_batch, y_train_batch, y_train_batch_hat)
                
                # update theta
                self.theta -= learning_rate * gradient

                # for debug
                # calculate loss and accuracy of train sets
                history['losses_train'].append(loss)

                y_train_hat = self.predict(X_train)

                # calculate loss and accuracy of validation sets
                y_val_hat = self.predict(X_val)
                loss_val = self.loss(y_val, y_val_hat)
                history['losses_val'].append(loss_val)


            if (epoch + 1) % 10 == 0:
                print("Epoch: {}, loss: {}, loss_val: {}".format(epoch + 1, loss, loss_val))
        return history

# Genetic Algorithm - Regression and Classification - using torch
class GeneticAlgorithm():
    def __init__(self, X, y, individual_size, population_size, fitness_function, bounds, mutation_rate = 0.05, crossover_rate= 0.9, elitism = 2):
        self.X = X
        self.y = y
        self.bounds = bounds
        self.elitism = elitism
        self.individual_size = individual_size
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.fitness_function = fitness_function
        self.population = torch.stack([self._create_individual() for _ in range(self.population_size)])
    
    def _create_individual(self):
        return (torch.rand(self.individual_size) - 0.5) * self.bounds
    
    def _fitness(self, individual):
        return 1 / (self.fitness_function(individual, self.X, self.y) + 0.001)
    
    def _cross_over(self, parent1, parent2):
        child1 = parent1.clone()
        child2 = parent2.clone()
        for i in range(self.individual_size):
            if torch.rand(1) < self.crossover_rate:
                child1[:, i] = parent2[:, i]
                child2[:, i] = parent1[:, i]
        return child1.reshape((1, -1)), child2.reshape((1, -1))
    
    def _mutate(self, individual):
        for i in range(self.individual_size):
            if torch.rand(1) < self.mutation_rate:
                individual[:, i] = self._create_individual()[i]
        return individual
    
    def _select(self, sorted_population):
        idx1 = torch.randint(0, self.population_size, (1,))
        while True:
            idx2 = torch.randint(0, self.population_size, (1,))
            if idx1 != idx2:
                break
        individual_s = sorted_population[idx1]
        if idx2 < idx1:
            individual_s = sorted_population[idx2]
        return individual_s
    
    def _create_next_generation(self, gen):
        sorted_population = torch.stack(sorted(self.population, key=self._fitness, reverse=True))
        next_generation = sorted_population[:self.elitism]
        if gen % 10 == 0:
            print("Generation: ", gen, "Best fitness: ", sorted_population[0], "with fitness: ", self._fitness(sorted_population[0]))
        while next_generation.shape[0] < self.population_size:
            parent1 = self._select(sorted_population)
            parent2 = self._select(sorted_population)
            child1, child2 = self._cross_over(parent1, parent2)
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            next_generation = torch.cat((next_generation, child1, child2), dim=0)
        return next_generation
    
    def fit(self, n_generations):
        history = []
        for gen in range(n_generations):
            history.append(self._fitness(self.population[0]))
            self.population = self._create_next_generation(gen + 1)
        return history
    
    def get_best_individual(self):
        return self.population[0]
