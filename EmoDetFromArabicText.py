import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Veri dosyalarının bulunduğu ana dizin yolu
main_path = "/Users/akadraydn/Desktop/arabic_sentiment"

# Eğitim ve test verilerini yükleyin
train_pos_path = pd.read_csv(main_path + "/train_pos.tsv", delimiter='\t', header=None)
train_neg_path = pd.read_csv(main_path + "/train_neg.tsv", delimiter='\t', header=None)
test_pos_path = pd.read_csv(main_path + "/test_pos.tsv", delimiter='\t', header=None)
test_neg_path = pd.read_csv(main_path + "/test_neg.tsv", delimiter='\t', header=None)

# Eğitim ve test verilerini birleştirin
train = pd.concat([train_pos_path, train_neg_path], axis=0, ignore_index=True, keys=['positive', 'negative'], names=['sentiment', 'index'])
train.columns = ['target', 'tweets']
test = pd.concat([test_pos_path, test_neg_path], axis=0, ignore_index=True, keys=['positive','negative'], names=['sentiment','index'])
test.columns = ['target', 'tweets']

# Eksik verileri kontrol edin ve giderin
train.isnull().sum()
test.isnull().sum()

# Tekrar eden verileri kontrol edin ve giderin
train.duplicated().sum()
test.duplicated().sum()
train.drop_duplicates(inplace=True)
test.drop_duplicates(inplace=True)

# Emojilerin kadlırılması
def remove_emoji(text):
  emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # duygular (emotions)
                           u"\U0001F300-\U0001F5FF"  # simgeler ve resimler (symbols and pictographs)
                           u"\U0001F680-\U0001F6FF"  # taşıma ve harita simgeleri (transport and map symbols)
                           u"\U0001F700-\U0001F77F"  # teknik simgeler (alım sonrası)
                           u"\U0001F780-\U0001F7FF"  # teknik simgeler (alt kısmı eksik)
                           u"\U0001F800-\U0001F8FF"  # simgeler (geometrik şekiller)
                           u"\U0001F900-\U0001F9FF"  # simgeler (çeşitli semboller)
                           u"\U0001FA00-\U0001FA6F"  # simgeler (spor)
                           u"\U0001FA70-\U0001FAFF"  # simgeler (yemekler)
                           u"\U00002702-\U000027B0"  # diğer simgeler
                           u"\U000024C2-\U0001F251" 
                           "]+", flags=re.UNICODE)
  return emoji_pattern.sub(r' ', text) 

#eğitim ve test verilerine emoji kaldırma fonksiyonunun uygulanması
train['tweets'] = train['tweets'].apply(lambda s: remove_emoji(s))
test['tweets'] = test['tweets'].apply(lambda s: remove_emoji(s))

import string

#noktalama işaretlerinin kaldırılması
def remove_punctuation(text):
  translator = str.maketrans(' ', ' ', string.punctuation)
  return text.translate(translator)

#eğitim ve test verilerine noktalama işaretlerinin kaldırılması fonksiyonunun uygulanması
train['tweets'] = train['tweets'].apply(lambda s: remove_punctuation(s))
test['tweets'] = test['tweets'].apply(lambda s: remove_punctuation(s))

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#Arapça stopwords'lerin kaldırılması
def remove_stopwords_arabic(text):
  stop_words = set(stopwords.words('arabic'))
  words = text.split()
  filtered_words = [word for word in words if word not in stop_words]
  return ' '.join(filtered_words)


#Stopwords kaldırma işleminin eğitim ve test verilerine uygulanması
train['tweets'] = train['tweets'].apply(lambda s: remove_stopwords_arabic(s))
test['tweets'] = test['tweets'].apply(lambda s: remove_stopwords_arabic(s))

mapping = {'neg':0, 'pos':1}   #hedef isimlerinin neg-pos'tan 0-1' e çevirilmesi

train['target'] = train['target'].map(mapping)
test['target'] = test['target'].map(mapping)



cv = CountVectorizer(binary=True) #metni vektörlere dönüştürme
cv.fit(train['tweets']) #eğitim
X = cv.transform(train['tweets']) #eğitim setindeki metin verilerini vektörize eder
X_test = cv.transform(test['tweets']) #test setindeki metin verilerini vektörize eder




# Logistic Regression modelini oluşturma ve eğitme
lr = LogisticRegression(C=10.0, penalty='l2')  

X_train, X_val, y_train, y_val = train_test_split(X, train['target'], train_size=0.7, shuffle=True)

lr.fit(X_train, y_train)

# Modelin doğruluğunu değerlendirme
y_pred = lr.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Model Doğruluğu:", accuracy)

# Tahmin yapma
y_pred_test = lr.predict(X_test)

# Doğruluk değerlendirmesi
accuracy_test = accuracy_score(test['target'], y_pred_test)
print("Test Doğruluğu:", accuracy_test)


def preprocess_text(text):
    text = remove_emoji(text)
    text = remove_punctuation(text)
    text = remove_stopwords_arabic(text)
    return text

# Arapça metin girişi
arabic_text = input("Buraya Arapça metnini girin: ")

# Metni ön işleme
processed_text = preprocess_text(arabic_text)

# Metni vektörleştirme
text_vector = cv.transform([processed_text])

# Tahmin yapma
predicted_label = lr.predict(text_vector)

# Tahmin sonucunu yazdırma
if predicted_label == 0:
    print("Metin negatif duyguya sahip.")
else:
    print("Metin pozitif duyguya sahip.")

