import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('data/archive.zip', encoding='latin-1', header=None)
df = df[[0, 5]]
df.columns = ['polarity', 'text']
print(df.head())
#dataset

df = df[df.polarity != 2]
#only positive and negative

df['polarity'] = df['polarity'].map({0: 0, 4: 1})
#This part scans for 0 or 4 then lists 0 as 0 and 4 as 1.
#Then it lists 0 as negative and 1 as positive

print(df['polarity'].value_counts())

def clean_text(text):
    return text.lower()
    # this part right here turns all the text to lowercase

df['clean_text'] = df['text'].apply(clean_text)
# we apply the cleaning

print(df[['text', 'clean_text']].head())

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['polarity'],
    test_size=0.2,
    random_state=42
)
#splits the data into clean text and polarity and does training

print("Train size:", len(X_train))
print("Test size:", len(X_test))

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# it puts the training data and the test data into a vector
# and then returns the shape of the vector

print("TF-IDF shape (train):", X_train_tfidf.shape)
print("TF-IDF shape (test):", X_test_tfidf.shape)

bnb = BernoulliNB()
bnb.fit(X_train_tfidf, y_train)
#train the data using a Bernoulli Naive Bayes model

bnb_pred = bnb.predict(X_test_tfidf)
#predicts the test data

print("Bernoulli Naive Bayes Accuracy:", accuracy_score(y_test, bnb_pred))
print("\nBernoulliNB Classification Report:\n", classification_report(y_test, bnb_pred))

svm = LinearSVC(max_iter=1000)
svm.fit(X_train_tfidf, y_train)
#train the data using a Support Vector Machine (SVM) model

svm_pred = svm.predict(X_test_tfidf)
#predicts test labels

print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))

logreg = LogisticRegression(max_iter=100)
logreg.fit(X_train_tfidf, y_train)
#trains the data using a Logistic Regression model

logreg_pred = logreg.predict(X_test_tfidf)
#predicts labels for test data

print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
print("\nLogistic")

sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
sample_vec = vectorizer.transform(sample_tweets)
#makes the predictions

print("\nSample Predictions:")
print("BernoulliNB:", bnb.predict(sample_vec))
print("SVM:", svm.predict(sample_vec))
print("Logistic Regression:", logreg.predict(sample_vec))