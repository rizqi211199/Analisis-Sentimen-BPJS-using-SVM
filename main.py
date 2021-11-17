import pandas as pd
import pickle

dt_skripsi = pd.read_csv('data_result.csv')
dt_skripsi= dt_skripsi[['content','label']]
dt_skripsi

jml = dt_skripsi['label'].value_counts()
print("Data Sebelum Resampling: ")
print(jml)

#resampling data
dt1 = dt_skripsi[dt_skripsi['label']==0].sample(750, replace=True)
dt2 = dt_skripsi[dt_skripsi['label']==1].sample(750, replace=True)
dt3 = dt_skripsi[dt_skripsi['label']==-1].sample(750, replace=True)
dt_skripsi = pd.concat([dt1, dt2, dt3])

dt_skripsi.shape
jml= dt_skripsi['label'].value_counts()
print("Data setelah Resampling: ")
print(jml)

#PREPROCESSING DATA
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string, re
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
    

review = []
for index, row in dt_skripsi.iterrows():
    review.append(cleansing(row['content']))
    
dt_skripsi['content']= review
print("==============HASIL PREPROCESSING=============")
print(dt_skripsi)

#pembagian data training dan data test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dt_skripsi['content'], dt_skripsi['label'],
                                                   test_size=0.2, stratify=dt_skripsi['label'], random_state = 30)

#tf idf
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()


#cek dimensi data menggunkan sintaks di atas, data memiliki 1680 baris untuk data training.
#dan 420 baris untuk data testing.
X_train= vectorizer.fit_transform(X_train)
pickle.dump(vectorizer, open('tfidf6.pkl', 'wb'))
X_test = vectorizer.transform(X_test)

print(X_train.shape)
print(X_test.shape)


#klasifikasi svm
from sklearn import svm
from sklearn.model_selection import cross_val_score

clf = svm.SVC(kernel = 'linear').fit(X_train,y_train)
pickle.dump(clf, open('svm6.pkl', 'wb'))
#prediksi data test
prediksi = clf.predict(X_test)


#confution matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, prediksi))

from sklearn.metrics import classification_report
print(classification_report(y_test, prediksi))

