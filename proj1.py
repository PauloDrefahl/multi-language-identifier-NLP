import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

print("Project 1 - Data Mining\n      Paulo \n ----------------------")
print("Amazon Alexa Reviews")

url = 'https://raw.githubusercontent.com/PauloDrefahl/TextMining/main/amazon_alexa.tsv'

data = pd.read_csv(url, sep='\t')

data.dropna(subset=["verified_reviews"], inplace=True)

reviews = data["verified_reviews"]
feedback = data["feedback"]

X_train, X_test, y_train, y_test = train_test_split(reviews, feedback, test_size=0.2, random_state=42)

count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(X_train)
X_test_count = count_vectorizer.transform(X_test)

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

nb_classifier = MultinomialNB()

nb_classifier.fit(X_train_count, y_train)
y_pred_count = nb_classifier.predict(X_test_count)
accuracy_count = accuracy_score(y_test, y_pred_count)
print("Accuracy using word counts:", accuracy_count)

nb_classifier.fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_classifier.predict(X_test_tfidf)
accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
print("Accuracy using tfidf:", accuracy_tfidf)

accuracy_count = accuracy_score(y_test, y_pred_count)
precision_count = precision_score(y_test, y_pred_count)
recall_count = recall_score(y_test, y_pred_count)

accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
precision_tfidf = precision_score(y_test, y_pred_tfidf)
recall_tfidf = recall_score(y_test, y_pred_tfidf)

print("Metrics for word counts:")
print("Accuracy:", accuracy_count)
print("Precision:", precision_count)
print("Recall:", recall_count)

print("\nMetrics for TF-IDF:")
print("Accuracy:", accuracy_tfidf)
print("Precision:", precision_tfidf)
print("Recall:", recall_tfidf)
