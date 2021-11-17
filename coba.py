# rendering contact.html template and making JSON response
from flask import Flask, render_template, request, redirect, url_for, session, g
import pandas as pd
import numpy as np
import itertools
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string, re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


app = Flask(__name__, template_folder='template')
svm = pickle.load(open('svm6.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf6.pkl', 'rb'))

def cleansing(text):
    #case folding
    text= text.lower()
    #stopword removel
    #remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
    #remove no ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    #remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",text).split())
    #remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    #hapus puncuation
    text = text.translate(str.maketrans("","",string.punctuation))
    #remove number
    text = re.sub(r"\d+", "", text)
    #remove whitespace leading & trailing
    text = text.strip()
    #remove multiple whitespace into single whitespace
    text = re.sub('\s+',' ', text)
    #remove single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

	#filtering
    factory = StopWordRemoverFactory().get_stop_words()
    txt_stopword = pd.read_csv('stopword.txt', names=["stopwords"], header= None)
    txt_stopword= txt_stopword["stopwords"][0].split(' ')
    stopword = factory + txt_stopword
    dictionary = ArrayDictionary(stopword)
    stop_remover=StopWordRemover(dictionary)
    text = stop_remover.remove(text)
    
	#stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    return text

def cleansing2(text):
    #case folding
    text= text.lower()
    #stopword removel
    #remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
    #remove no ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    #remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ",text).split())
    #remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    #hapus puncuation
    text = text.translate(str.maketrans("","",string.punctuation))
    #remove number
    text = re.sub(r"\d+", "", text)
    #remove whitespace leading & trailing
    text = text.strip()
    #remove multiple whitespace into single whitespace
    text = re.sub('\s+',' ', text)
    #remove single char
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

	#filtering
    factory = StopWordRemoverFactory().get_stop_words()
    more_stopword= ['bpjs',' bpjs','Bpjs','BPJS',' bpjs ']
    stopword = pd.read_csv('stopword2.txt', names=["stopwords"], header= None)
    stopword= stopword["stopwords"][0].split(' ')
    stopword_txt = factory + stopword + more_stopword
    dictionary2 = ArrayDictionary(stopword_txt)
    stop_remover2=StopWordRemover(dictionary2)
    text = stop_remover2.remove(text)
    
    return text

def preprocessing(dt):
    dt = cleansing(dt)
    dt = vectorizer.transform([dt])
    return dt

@app.route("/scraping", methods=["POST", "GET"])
def scraping():
    if request.method == "POST":
        keyword = request.form["nm"]
        jumlah = request.form["jml"]
        
        return redirect(url_for("user", search=keyword, jml= jumlah))
    else:
        return render_template('scrape.html')

@app.route("/<search>/<jml>")
def user(search, jml):
    jml = int(jml)
    scraped_tweets = sntwitter.TwitterSearchScraper(search).get_items()
    sliced_scraped_tweets = itertools.islice(scraped_tweets, jml)
    df = pd.DataFrame(sliced_scraped_tweets)[['content']]

    review = []
    for index, row in df.iterrows():
        review.append(svm.predict(preprocessing(row['content'])))

    df['sentimen']= review

    df['sentimen'].replace(1, 'positif', inplace= True)
    df['sentimen'].replace(-1, 'negatif', inplace= True)
    df['sentimen'].replace(0, 'netral', inplace= True) 
    df.to_csv('df.csv')
    return render_template('tabel.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
    
    

@app.route('/bar_route')   
def bar_route():
    df= pd.read_csv('df.csv')
    df = df[['content', 'sentimen']]
    bar_chart = pygal.Bar()
    bar_chart.title = 'analisis sentimen'
    nf = 0
    pf = 0
    nt = 0
    for i, row in df.iterrows():
        if row['sentimen']== 'positif':
            pf= pf+1
        elif row['sentimen']=='negatif':
            nf = nf+1
        elif row['sentimen']== 'netral':
            nt = nt+1

    bar_chart.add('Positif', pf)
    bar_chart.add('Negatif', nf)
    bar_chart.add('Netral', nt)
    barchart_data=bar_chart.render_data_uri()
    return render_template('grafik.html',barchart_data=barchart_data)

def get_wordcloud(text):
    pil_img = WordCloud().generate(text=text).to_image()
    img= io.BytesIO()
    pil_img.save(img, "PNG")
    img.seek(0)
    img_b64= base64.b64encode(img.getvalue()).decode()
    return img_b64


@app.route('/wc')
def wc():
    df= pd.read_csv('df.csv')
    df= df[['content','sentimen']]
    review=[]
    for index, row in df.iterrows():
        review.append(cleansing2(row['content']))
    df['content']= review
    df.to_csv('wc.csv')
    data = df[df['sentimen']== 'negatif']
    data = ' '.join(word for word in data['content'])
    clouds=[]
    cloud = get_wordcloud(data)
    clouds.append(cloud)
    return render_template('wc.html', articles= clouds)

@app.route('/wc2')
def wc2():
    df= pd.read_csv('df.csv')
    df= df[['content','sentimen']]
    review=[]
    for index, row in df.iterrows():
        review.append(cleansing2(row['content']))
    df['content']= review
    data = df[df['sentimen']== 'positif']
    data = ' '.join(word for word in data['content'])
    clouds1=[]
    cloud1 = get_wordcloud(data)
    clouds1.append(cloud1)
    return render_template('wc2.html', article= clouds1)

@app.route('/wc3')
def wc3():
    df= pd.read_csv('df.csv')
    df= df[['content','sentimen']]
    review=[]
    for index, row in df.iterrows():
        review.append(cleansing2(row['content']))
    df['content']= review
    data = df[df['sentimen']== 'netral']
    data = ' '.join(word for word in data['content'])
    clouds2=[]
    cloud2= get_wordcloud(data)
    clouds2.append(cloud2)
    return render_template('wc3.html', article2= clouds2)
if __name__=='__main__':
	app.run(debug=True)
