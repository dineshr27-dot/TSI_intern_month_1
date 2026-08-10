#TF-IDF + Logistic regression

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from fastapi import FastAPI
from pydantic import BaseModel

#insert the data in csv file

data = pd.read_csv("spam.csv",encoding = "latin-1")

# keep required columns

data = data[["v1", "v2"]]

#columns rename

data.columns =["label" , "message"]

X = data["message"]

y = data["label"] #replay the answer spam / ham

# Train the data and split test 

X_train, X_test , y_train , y_test = train_test_split(X,y, test_size=0.2 , random_state=42)

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

# print("\n Vocabulary:")
# print(vectorizer.get_feature_names_out()[:20])

# print("\n Training TF-IDF Shape:")
# print(X_train_tfidf.shape)

#transform its only convert # it use the vocabulary learned from X_train

X_test_tfidf = vectorizer.transform(X_test)

# model create logistic regression

model = LogisticRegression(max_iter= 1000)

#fit = Learn

model.fit(X_train_tfidf,y_train)

#predict the answer

predictions = model.predict(X_test_tfidf)

#check accuray

accuracy = accuracy_score( y_test , predictions)
# print("\nAccuracy:", accuracy)

#FastAPI connect

app = FastAPI()

#Request Body Model

class MessageRequest(BaseModel):
    message: str
    
# Home API

@app.get("/")
def home():
    
    return{
        "message":"SMS spam detection API"
    }
    
#prediction API

@app.post("/predict")
def predict_sms(
    request: MessageRequest
):
    #get sms form request
    message = request.message
    
    #convert the sms TF-IDF
    
    message_tfidf = vectorizer.transform([message])
    
    #predticon 
    
    prediction = model.predict(message_tfidf)[0]
    
    return{
        "message" : message,
        "prediction": prediction
        
    }


