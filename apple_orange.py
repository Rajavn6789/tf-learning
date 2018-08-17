from sklearn import tree
features = [[140, 0], [130, 0], [150, 1], [170, 1], [120, 0], [140, 1]]  # 0 smooth 1 bumpy
labels = [1, 1, 2, 2, 1, 2]  # 1 apple  2 orange

# intialize decison tree
clf = tree.DecisionTreeClassifier()

# Find patterns in data
clf = clf.fit(features, labels)

# Check predictions
print(clf.predict([[150, 0]]))
print(clf.predict([[160, 1]]))
