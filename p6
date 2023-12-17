import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("/content/Program6dataset.csv", names=['message','label'])
df['labelnum'] = df.label.map({'pos' : 1, 'neg' : 0})
print(df)

X = df.message
Y = df.labelnum
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5)
print("Train size : ", x_train.shape[0])
print("Test size : ", x_test.shape[0])

count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(x_train)
xtest_dtm = count_vect.transform(x_test)
feature_df = pd.DataFrame(xtrain_dtm.toarray(), columns=count_vect.get_feature_names_out())
print(feature_df)

classifier = MultinomialNB()
classifier.fit(xtrain_dtm, y_train)
y_pred = classifier.predict(xtest_dtm)

for input, output in zip(x_test, y_pred):
  print("input : ", input)
  print("Predicted : ", output)
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred))
