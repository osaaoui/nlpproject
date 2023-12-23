import pickle
import joblib
from sklearn.feature_extraction.text import CountVectorizer




cv = pickle.load(open("vectorizer.pkl", 'rb'))
clf = joblib.load(open("models/model.joblib", 'rb'))
def model_predict(email):
    if email == "":
        return ""
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    print("Prediction: ", prediction)
    return prediction
