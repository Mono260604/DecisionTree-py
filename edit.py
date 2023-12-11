import pandas as pd
test_case = pd.read_csv("test.csv")
train_case = pd.read_csv("train.csv")
frame  = [train_case , test_case]
dataset = train_case
print(dataset)
dataset.corr()
print(dataset)
feature_columns = dataset.drop(columns=['price_range'])
target_column = dataset['price_range']
from sklearn.model_selection import train_test_split
X = feature_columns
y = target_column
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.33)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
predictions
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions, labels=[0,1])
from sklearn.metrics import precision_score
precision_score(y_test, predictions, average='weighted')
from sklearn.metrics import recall_score
recall_score(y_test, predictions, average='weighted')
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['accuracy_score', 'confusion_matrix', 'precision_score', 'recall_score']))
feature_names = X.columns
feature_names
feature_importance = pd.DataFrame(clf.feature_importances_, index=X.columns, columns=['Importance'])
feature_importance
feature_importance = feature_importance.sort_values('Importance', ascending=False)
from matplotlib import pyplot as plt
top_n = 10
plt.figure(figsize=(10, 6))
plt.barh(range(top_n), feature_importance['Importance'][:top_n], align='center')
plt.yticks(range(top_n), feature_importance.index[:top_n])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importances - Decision Tree Regressor')
plt.show()
from sklearn.tree import graphviz
from sklearn.tree import export_graphviz
target_names = ['accuracy_score', 'confusion_matrix', 'precision_score', 'recall_score']
dot_data = export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=target_names, filled=True, rounded=True)
# Visualize the decision tree using Graphviz
graph = graphviz.Source(dot_data)
graph.render("Price_decision", format="png")  
graph.view("Price_decision") 