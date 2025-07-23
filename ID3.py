import numpy as np
import pandas as pd
from collections import Counter

# Entropy e Information Gain como definidos no seu código
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * np.log2(p) for p in probs if p > 0])

def information_gain(parent, left, right):
    weight_l = len(left) / len(parent)
    weight_r = len(right) / len(parent)
    return entropy(parent) - (weight_l * entropy(left) + weight_r * entropy(right))

# Nó da árvore e DecisionTree definidos no seu código
class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=10):  # max_depth aumentado para iris
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

# Função para acurácia
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

# --- Código principal ---

# 1. Carregar dataset iris.csv
df = pd.read_csv('iris.csv')

# 2. Preparar dados (sem discretização)
# Assumindo que colunas de atributos são todas menos 'ID' e 'class'
target = 'class'
features = [col for col in df.columns if col not in ['ID', target]]

X = df[features].values
y = df[target].values

# 3. Shuffle e split 80/20
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)

split_idx = int(0.8 * len(X))
train_idx = indices[:split_idx]
test_idx = indices[split_idx:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# 4. Treinar árvore
tree = DecisionTree(max_depth=7)
tree.fit(X_train, y_train)

# 5. Avaliar acurácia
y_pred = tree.predict(X_test)
print(f"Acurácia: {accuracy(y_test, y_pred):.2%}")

# Opcional: para visualizar a árvore
# tree.print_tree(feature_names=features)
