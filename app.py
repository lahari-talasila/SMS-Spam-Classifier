from flask import Flask, redirect, url_for, render_template, request 
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

import string 
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
from sklearn.feature_extraction.text import  TfidfVectorizer


app = Flask(__name__)

model=pickle.load(open('model.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))  

@app.route('/')
def welcome(message=""):
    redirect('/')
    return render_template('index.html',message=message)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        print("\n\n\n\n\n\n\n")
        sentence=request.form.get('sentence')
        print('sentence===>',sentence)
       
        result=  preprocessAndPredict(request.form.get('sentence'))
        print("\n\n\n\n\n\n\n")
        if(result==1):
            return render_template('predict.html',text="The SMS is NOT spam")
        else:
            return render_template('predict.html',text="The SMS is SPAM")

def preprocessAndPredict(sentence):
    sentence = sentence.lower()
    sentence =nltk.word_tokenize(sentence)

    y =[]
    for i in sentence:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))
    sentence = " ".join(y)
    
    tfidf = TfidfVectorizer(max_features=3000)
    x = tfidf.fit_transform(sentence).toarray()
    return model.predict(x)[0]

if __name__ == '__main__':
    app.run(debug=True)
