import pandas as pd
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
from nltk.corpus import stopwords

data = pd.read_csv("spam.csv", names=['text', 'label'], encoding='latin-1')

def clean_text(text):
    text = str(text).lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

data['clean_text'] = data['text'].apply(clean_text)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


def predict_email(email):
    email_clean = clean_text(email)
    email_vector = vectorizer.transform([email_clean])
    prediction = model.predict(email_vector)
    return prediction[0]

# Example 
email = """
        URGENT! We are trying to contact you. Last
         weekends draw shows that you have won a å£900 prize
         GUARANTEED. Call 09061701939. Claim code S89.
           Valid 12hrs only.
    """
print("Prediction:", predict_email(email))
