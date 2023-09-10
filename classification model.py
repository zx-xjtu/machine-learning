import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

data = pd.DataFrame(pd.read_excel("your_data.csv"))
# print(pd.DataFrame(data.iloc[1]))
num = list(range(len(data)))
print(num)
test_size = int(0.2*len(num))
print(test_size)
InputX = list(data)[a:b]
clf1 = LinearSVC(C=2, loss='squared_hinge', dual=False, tol=1e-4, multi_class=ovr)
clf2 = RandomForestClassifier(criterion="gini", n_estimators=400, max_depth=4, max_features=sqrt)
clf3 = KNeighborsClassifier(n_neighbors=11, algorithm=auto, leaf_size=30, weights=uniform, metric=minkowski)
clf4 = GaussianNB(priors=None, var_smoothing=1e-9)
clf5 = MLPClassifier(activation="relu", alpha=0.0001, early_stopping=False, shuffle=True,
                     hidden_layer_sizes=(64, 32, 16),
                     learning_rate="constant", learning_rate_init=0.01, solver="adam",
                     validation_fraction=0.1,
                     max_iter=10000)
estimator1 = [clf1, clf2, clf3, clf4, clf5]
for n in estimator1:
    for m in estimator1:
        accuracy = []
        nums = random.sample(range(0, 1000), 100)
        for i in nums:
                test_data = data.sample(n= test_size, random_state=i)
                train_data = data.drop(test_data.index)
                x_test_data = test_data[InputX]
                x_train_data = train_data[InputX]
                y_test_data = test_data["Output Y"]
                y_train_data = train_data["Output Y"]
                transfer = StandardScaler()
                x_train = transfer.fit_transform(x_train_data)
                x_test = transfer.fit_transform(x_test_data)

                estimator = StackingClassifier(estimators=[("estamitor1", n), ("estimator2", m)])
                estimator.fit(x_train, y_train_data)
                score = estimator.score(x_test, y_test_data)
                accuracy.append(score)
        print(accuracy)
        print("准确率是：\n", np.mean(accuracy))






