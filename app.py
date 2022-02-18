from flask import Flask, render_template, request
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


def clean_tokenize_stop(text):
    
    #Removes Unnecessary characters and only collects alphabets or numbers
    regexp = RegexpTokenizer('\w+')
    cleaned = regexp.tokenize(text.lower()) #transform text into lower cases
        
    #Removing stopwords or common words which dont add any meaning
    stopwords = nltk.corpus.stopwords.words("english")
    stopwords.extend(['movies', 'movie', 'im', 'film', 'br']) #extending my stopwords
    cleaned2 = [item for item in cleaned if item not in stopwords]
    
    #Performing lemmatization(reverting a word to its base form) which is better than stemming 
    wordnet_lem = WordNetLemmatizer()
    cleaned3 = [wordnet_lem.lemmatize(item) for item in cleaned2] # running runs
    
    cleaned4 = ' '.join([word for word in cleaned3]) 
                
    return cleaned4

model = joblib.load('model/finalized_model.pkl')
app = Flask(__name__)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/', methods=['GET','POST'])
def predict():
    text = request.form['review']
    text = clean_tokenize_stop(text)
    final_text = [text]
    output = model.predict(final_text)[0]
    
    if output == 0:
        out = 'The moview review is negative'
    else:
        out = 'The movie review is positive'
        
    return render_template('index.html', prediction_text = out)
     
  

if __name__ == '__main__':  
    app.run(debug=False)