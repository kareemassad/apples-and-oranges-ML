from sklearn import tree
# features = [[140,"smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
# bumpy = 0, smooth = 1
features = [[140,1], [130, 1], [150, 0], [170, 0]]

# labels = ["apple", "apple", "orange", "orange"]
# similar to above, apple = 0, orange = 1;
labels = ["0", "0", "1", "1"]

#creat classifier
classifier = tree.DecisionTreeClassifier()

classifier = classifier.fit(features, labels)

# testing an object that weighs 100 grams and is bumpy. 
# Expecting ['0'] output to signal an apple
print (classifier.predict([[100, 0]]))  