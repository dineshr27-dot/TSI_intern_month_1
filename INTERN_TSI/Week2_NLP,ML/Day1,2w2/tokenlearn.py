from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
import re
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#  1 tokenization , word , sentence , char

text = "She is always kind to animals.  This little black dress isn’t expensive. playing and studies"

token_1 = word_tokenize(text)

token_2 = sent_tokenize(text)

characters = list(text)

print("\n Word token :", token_1)

print("\n Sentence token :", token_2)

print("\n Character :" , characters)

 # 2 Stop words

stop_words = set(stopwords.words("english"))

words=(token_1)

filtered = []

for word in words:
    if word.lower() not in stop_words:
        filtered.append(word)
print("\n STOP WORDS :")
print(filtered)


# # 3 stemming 

steam = PorterStemmer()

for s in token_1:
    print("\n STEMMING : ",s, "->",steam.stem(s))

# # 4 Lemmatization 

lemma = WordNetLemmatizer()

for i in token_1:
    print("\n LEMMATIZATION")
    print("\n Lemmatization:",i, "->",lemma.lemmatize(i, "v") )
    
# REGEX 

text_2 = "Hello this MY number is 98832957987937 !!!@@@"

print(re.search("this",text_2))
print(re.match("Hello",text_2))
print(re.findall("is",text_2))
print(re.sub(r"[^\w\s]","" ,text_2))
print(re.sub(" ","+" ,text_2))

#spaCy

nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

for token in doc:
    print(token.text)
    
# BoW (Bag of words)
# convert text into numbers by counting word frequencies

docs = ["She is always kind to animals." ,
        "This little black dress isn't expensive. a playing and studies I am a"]

vectorizer = CountVectorizer(ngram_range=(1,2)) # using in N-gram

X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())

#vocabulary
#N-gram
#TF-IDF

vectorizer = TfidfVectorizer() # using in N-gram

X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
print(X.toarray())