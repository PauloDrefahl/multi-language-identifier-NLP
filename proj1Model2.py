import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Project 1 - Data Mining\n      Paulo \n ----------------------")
print("Amazon Alexa Reviews")

url = 'https://raw.githubusercontent.com/PauloDrefahl/TextMining/main/amazon_alexa.tsv'

data = pd.read_csv(url, sep='\t')

data.dropna(subset=["variation"], inplace=True)

reviews = data["variation"]
feedback = data["feedback"]

X_train, X_test, y_train, y_test = train_test_split(reviews, feedback, test_size=0.2, random_state=42)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

svm_classifier = SVC()

svm_classifier.fit(X_train_count, y_train)
y_pred_count = svm_classifier.predict(X_test_count)
accuracy_count = accuracy_score(y_test, y_pred_count)
print("Accuracy using word counts:", accuracy_count)

svm_classifier.fit(X_train_tfidf, y_train)
y_pred_tfidf = svm_classifier.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("Accuracy using TF-IDF:", accuracy_tfidf)

precision_count = precision_score(y_test, y_pred_count)
recall_count = recall_score(y_test, y_pred_count)

precision_tfidf = precision_score(y_test, y_pred_tfidf)
recall_tfidf = recall_score(y_test, y_pred_tfidf)

print("\nMetrics for word counts:")
print("Precision:", precision_count)
print("Recall:", recall_count)

print("\nMetrics for TF-IDF:")
print("Precision:", precision_tfidf)
print("Recall:", recall_tfidf)
