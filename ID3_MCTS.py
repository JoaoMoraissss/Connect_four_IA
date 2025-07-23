import numpy as np
import pandas as pd
from collections import Counter
import random

# ------------------- Funções de Entropia -------------------
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(parent, left, right):
    weight_l = len(left) / len(parent)
    weight_r = len(right) / len(parent)
    return entropy(parent) - (weight_l * entropy(left) + weight_r * entropy(right))

# ------------------- Nó da Árvore -------------------
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# ------------------- Árvore de Decisão -------------------
class DecisionTree:
    def __init__(self, max_depth=10):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y)

    def _grow(self, X, y, depth=0):
        if len(set(y)) == 1 or depth == self.max_depth:
            return TreeNode(value=Counter(y).most_common(1)[0][0])

        best_gain = -1
        best_split = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t
                if len(y[left_mask]) == 0 or len(y[right_mask]) == 0:
                    continue
                gain = information_gain(y, y[left_mask], y[right_mask])
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature, t, left_mask, right_mask)

        if best_split is None:
            return TreeNode(value=Counter(y).most_common(1)[0][0])

        f, t, left_mask, right_mask = best_split
        left = self._grow(X[left_mask], y[left_mask], depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)
        return TreeNode(feature=f, threshold=t, left=left, right=right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)

    def print_tree(self, node=None, depth=0, feature_names=None):
        if node is None:
            node = self.root
        indent = "  " * depth
        if node.value is not None:
            print(f"{indent}Predict: {node.value}")
        else:
            fname = feature_names[node.feature] if feature_names else f"X[{node.feature}]"
            print(f"{indent}{fname} <= {node.threshold}")
            self.print_tree(node.left, depth + 1, feature_names)
            print(f"{indent}else:")
            self.print_tree(node.right, depth + 1, feature_names)

def extract_features(board, current_player):
    features = []
    for row in board:
        features.extend(row)
    features.append(current_player)
    return np.array(features).reshape(1, -1)  # reshape para 2D, pois predict espera matriz



def predict_connect4_move(tree, board, current_player, valid_moves):
    features = extract_features(board, current_player)
    pred = tree.predict(features)[0]
    # Verifica se o movimento previsto é válido, caso contrário escolhe um válido random
    if pred not in valid_moves:
        pred = random.choice(valid_moves)
    return pred


# ------------------- Avaliação -------------------
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    labels = np.unique(y_true)
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for i in range(len(y_true)):
        true = np.where(labels == y_true[i])[0][0]
        pred = np.where(labels == y_pred[i])[0][0]
        mat[true, pred] += 1
    return mat

# 1. Carregar dados
def train_tree():
    # 1. Carregar dados
    df = pd.read_csv("connect4_mcts_dataset.csv")
    df = df.dropna(subset=['move'])  # remover linhas com move = None

    X = df.drop(columns=['move']).values
    y = df['move'].astype(int).values   

    # 3. Dividir treino/teste
    split = int(0.7 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 4. Treinar árvore
    tree = DecisionTree(max_depth=10)
    tree.fit(X_train, y_train)

    # 5. Avaliar (opcional)
    y_pred = tree.predict(X_test)
    print(f"ID3 Accuracy: {accuracy(y_test, y_pred):.2f}")

    return tree