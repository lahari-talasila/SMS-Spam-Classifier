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

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
        print("\n\n\n\n\n\n\n")
        sentence=request.form.get('sentence')
        print('sentence===>',sentence)
       
        result=  preprocessAndPredict(request.form.get('sentence'))
        print("\n\n\n\n\n\n\n")
        if(result==0):
            return render_template('index.html',text="The SMS is NOT spam")
        else:
            return render_template('index.html',text="The SMS is SPAM")

def preprocessAndPredict(sentence):
    # Convert to lowercase
    sentence = sentence.lower()
    
    # Tokenize the sentence
    words = nltk.word_tokenize(sentence)

    # Remove stopwords and punctuation, and apply stemming
    filtered_words = []
    for word in words:
        if word.isalnum():  # Filter out non-alphanumeric characters
            if word not in stopwords.words('english'):  # Remove stopwords
                filtered_words.append(ps.stem(word))  # Apply stemming

    # Join the processed words back into a single string
    processed_sentence = " ".join(filtered_words)

    # Transform the processed sentence using the pre-fitted TF-IDF vectorizer
    vectorized_input = tfidf_vectorizer.transform([processed_sentence])

    # Predict using the pre-trained model
    return model.predict(vectorized_input)[0]



if __name__ == '__main__':
    app.run(debug=True)
  