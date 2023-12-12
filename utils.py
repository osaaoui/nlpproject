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
    # If the email is spam prediction should be 1
    #if prediction == 0:
     #   prediction = 'Checking or savings account'
    #elif prediction == 1:
     #   prediction = 'Credit card or prepaid card'
    #elif prediction == 2:
     #   prediction = 'Credit reporting, credit repair services, or other personal consumer reports'
    #elif prediction == 3:
     #   prediction = 'Debt collection'
    #elif prediction == 4:
     #   prediction = 'Money transfer, virtual currency, or money service'
    #elif prediction == 5:
     #   prediction = 'Mortgage'
    #elif prediction == 6:
     #   prediction = 'Payday loan, title loan, or personal loan'
    #elif prediction == 7:
     #   prediction = 'Student loan'
    #else:
     #   prediction = 'Vehicle loan or lease'
    return prediction
