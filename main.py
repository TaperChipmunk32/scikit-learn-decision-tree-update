import sklearn.tree as tree
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

for i in range(1, 3):
    if i == 1:
        data = load_breast_cancer()
    elif i == 2:
        data = load_wine()
    X = data.data
    y = data.target
    print(f"Dataset {i}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
    unpruned_tree = tree.DecisionTreeClassifier(pruning=None, random_state=42, validation_split=0.3)
    unpruned_tree.fit(X_train, y_train)
    print(f'Decision Tree Accuracy: {unpruned_tree.score(X_test, y_test)}')
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(unpruned_tree, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    fig.savefig(f"tree_images/unpruned_decistion_tree_{i}.png")

    pruned_tree_rep = tree.DecisionTreeClassifier(pruning="rep", random_state=42, validation_split=0.3)
    pruned_tree_rep.fit(X_train, y_train)
    print(f'Reduced Error Pruned Decision Tree Accuracy: {pruned_tree_rep.score(X_test, y_test)}')
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(pruned_tree_rep, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    fig.savefig(f"tree_images/pruned_decistion_tree_rep_{i}.png")

    pruned_tree_ccp = tree.DecisionTreeClassifier(pruning="ccp", random_state=42, ccp_alpha=0.005, validation_split=0.3)
    pruned_tree_ccp.fit(X_train, y_train)
    print(f'Cost Complexity Pruned Decision Tree Accuracy: {pruned_tree_ccp.score(X_test, y_test)}')
    fig = plt.figure(figsize=(25,20))
    tree.plot_tree(pruned_tree_ccp, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    fig.savefig(f"tree_images/pruned_decistion_tree_ccp_{i}.png")
