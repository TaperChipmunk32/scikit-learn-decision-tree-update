import sklearn.tree as tree
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import numpy as np

rep_wins = 0
ccp_wins = 0
ties = 0
differences = []

for i in range(1, 3):
    if i == 1:
        data = load_breast_cancer()
    elif i == 2:
        data = load_wine()
    X = data.data
    y = data.target
    print(f"Dataset {i}")
    for j in range(0,100):
        print(f"Iteration {j}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
        unpruned_tree = tree.DecisionTreeClassifier(pruning=None, random_state=42, validation_split=0.3)
        unpruned_tree.fit(X_train, y_train)
        unpruned_score = unpruned_tree.score(X_test, y_test)
        print(f'Decision Tree Accuracy:\t\t\t\t {unpruned_score}')

        pruned_tree_rep = tree.DecisionTreeClassifier(pruning="rep", random_state=42, validation_split=0.3)
        pruned_tree_rep.fit(X_train, y_train)
        pruned_score_rep = pruned_tree_rep.score(X_test, y_test)
        print(f'Reduced Error Pruned Decision Tree Accuracy:\t {pruned_score_rep}')

        tree_ccp = tree.DecisionTreeClassifier(pruning="ccp", random_state=42, validation_split=0.3)
        pruned_tree_ccp = GridSearchCV(tree_ccp, {'ccp_alpha':[0.001, 0.0001, 0.005, 0.0005, 0.0025, 0.00025]})
        pruned_tree_ccp.fit(X_train, y_train)
        pruned_score_ccp = pruned_tree_ccp.score(X_test, y_test)
        print(f'Cost Complexity Pruned Decision Tree Accuracy:\t {pruned_score_ccp}')
        
        if pruned_score_rep > pruned_score_ccp:
            print("REP WINS")
            rep_wins += 1
        elif pruned_score_ccp > pruned_score_rep:
            print("CCP WINS")
            ccp_wins += 1
        elif pruned_score_rep == pruned_score_ccp:
            print("TIED")
            ties +=1
        differences.append(max([pruned_score_rep, pruned_score_ccp]) - min([pruned_score_rep, pruned_score_ccp]))
            
print(f"REP: {rep_wins}, CCP: {ccp_wins}, TIES: {ties}")
diffs = np.asarray(differences)
min_diff = min(diffs[np.nonzero(diffs)])
max_diff = max(differences)
avg_diff = sum(differences) / len(differences)
print(f"Min Diff: {min_diff}, Avg Diff: {avg_diff}, Max Diff: {max_diff}")