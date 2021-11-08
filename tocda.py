import glob
import nltk
import spacy
import os
import string
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import matplotlib.pyplot as plt

os.chdir(r'C:/Users/adars/Desktop/TOC_Assignment_Instructions_Code/dataset')  #don't remove r and give extra space
myFiles = glob.glob('*.txt')
# print(myFiles)  #prints text file names

dict_words={}
nltk.download('punkt')
nltk.download('stopwords')

for filename in myFiles:
    # print(filename)
    file = open(filename, 'rt', encoding="utf8")
    text = file.read()
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word 
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    words=list(set(words))
    dict_words[filename]=words
    words=[]
    file.close()

#print(dict_words)

B = pd.read_csv('C:/Users/adars/Desktop/TOC_Assignment_Instructions_Code/BlacklistedFile1.csv')
#print(B)

value=B['Index'].tolist()
key=B['Sensitivity'].tolist()

#print(value)
#print(key)

Ref_dict=dict(zip(value,key))
#print(Ref_dict)      #Data Dictionary


# Clustering
less_sensitive=[]
avg_sensitive=[]
high_sensitive=[]

for name, words in dict_words.items():
    max=0
    for x in words:
        if(max==3):
            break
        for value, key in Ref_dict.items():
            if(x==value):
                if(max<key):
                    max=int(key)
                if(max==3):
                    print(name,"has High Senstive data")
                    high_sensitive.append(name)
                    break
    if(max==0):
        print(name,"has No Sensitive data")
    if(max==1):
        print(name,"has Less Sensitive data")
        less_sensitive.append(name)
    if(max==2):
        print(name,"has Average Sensitive data")
        avg_sensitive.append(name)
                
print()
print("CLUSTER LESS-SENSITIVE:",less_sensitive)
print("CLUSTER AVG-SENSITIVE:",avg_sensitive)
print("CLUSTER HIGH-SENSITIVE:",high_sensitive)

x=myFiles
y=[]
for i in x:
    if(i in less_sensitive):
        y.append(1)
    elif(i in avg_sensitive):
        y.append(2)
    elif(i in high_sensitive):
        y.append(3)
    else:
        y.append(None)
plt.yticks([1,2,3],["Less Sensitive","Avg Sensitive","High Sensitive"])
plt.xticks(rotation=45, ha='right')
plt.ylabel("SENSITIVITY")
plt.xlabel("FILES")
for i in range(0,len(x)):
    if(y[i]==3):
        plt.scatter(x[i],y[i],color="red")
    if(y[i]==2):
        plt.scatter(x[i],y[i],color="blue")
    if(y[i]==1):
        plt.scatter(x[i],y[i],color="green")

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree,tree2conlltags
from pprint import pprint
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()



def preprocess(sent):
    sent = word_tokenize(sent)
    sent = pos_tag(sent)
    return sent


for filename in high_sensitive:
    file = open(filename, 'rt', encoding="utf8")
    text = file.read()
    sent = preprocess(text)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    # pprint(iob_tagged)
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(text)))
    
    doc=nlp(text)
    # pprint([(X.text, X.label_) for X in doc.ents])
    # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
    len(doc.ents)
    labels = [x.label_ for x in doc.ents]
    Counter(labels)
    items = [x.text for x in doc.ents]
    print(Counter(items).most_common(3))
    
    sentences = [x for x in doc.sents]
    ttemp = len(sentences) - 1
    displacy.render(nlp(str(sentences[ttemp])),jupyter=True, style='ent')
    displacy.render(nlp(str(sentences[ttemp])), style='dep', jupyter = True, options = {'distance': 120})
    
    [(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[ttemp])) if not y.is_stop and y.pos_ != 'PUNCT']]
    
    dict([(str(x), x.label_) for x in nlp(str(sentences[ttemp])).ents])
    print([(x, x.ent_iob_, x.ent_type_) for x in sentences[ttemp]])
    file.close()

for filename in avg_sensitive:
    file = open(filename, 'rt', encoding="utf8")
    text = file.read()
    sent = preprocess(text)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    # pprint(iob_tagged)
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(text)))
    
    doc=nlp(text)
    # pprint([(X.text, X.label_) for X in doc.ents])
    # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
    len(doc.ents)
    labels = [x.label_ for x in doc.ents]
    Counter(labels)
    items = [x.text for x in doc.ents]
    print(Counter(items).most_common(3))
    
    sentences = [x for x in doc.sents]
    temp = len(sentences) - 1
    displacy.render(nlp(str(sentences[temp])),jupyter=True, style='ent')
    displacy.render(nlp(str(sentences[temp])), style='dep', jupyter = True, options = {'distance': 120})
    
    [(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[temp])) if not y.is_stop and y.pos_ != 'PUNCT']]
    
    dict([(str(x), x.label_) for x in nlp(str(sentences[temp])).ents])
    print([(x, x.ent_iob_, x.ent_type_) for x in sentences[temp]])
    file.close()

for filename in less_sensitive:
    file = open(filename, 'rt', encoding="utf8")
    text = file.read()
    sent = preprocess(text)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    # pprint(iob_tagged)
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(text)))
    
    doc=nlp(text)
    # pprint([(X.text, X.label_) for X in doc.ents])
    # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
    len(doc.ents)
    labels = [x.label_ for x in doc.ents]
    Counter(labels)
    items = [x.text for x in doc.ents]
    print(Counter(items).most_common(3))
    
    sentences = [x for x in doc.sents]
    temp = len(sentences) - 1
    displacy.render(nlp(str(sentences[temp])),jupyter=True, style='ent')
    displacy.render(nlp(str(sentences[temp])), style='dep', jupyter = True, options = {'distance': 120})
    
    [(x.orth_,x.pos_, x.lemma_) for x in [y for y in nlp(str(sentences[temp])) if not y.is_stop and y.pos_ != 'PUNCT']]
    
    dict([(str(x), x.label_) for x in nlp(str(sentences[temp])).ents])
    print([(x, x.ent_iob_, x.ent_type_) for x in sentences[temp]])
    file.close()
plt.ylabel("SENSITIVITY")
plt.xlabel("FILES")
for i in range(0,len(x)):
    if(y[i]==3):
        plt.scatter(x[i],y[i],color="red")
    if(y[i]==2):
        plt.scatter(x[i],y[i],color="blue")
    if(y[i]==1):
        plt.scatter(x[i],y[i],color="green")

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree,tree2conlltags
from pprint import pprint
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()



def preprocess(sent):
    sent = word_tokenize(sent)
    sent = pos_tag(sent)
    return sent


for filename in high_sensitive:
    file = open(filename, 'rt', encoding="utf8")
    text = file.read()
    sent = preprocess(text)
    pattern = 'NP: {<DT>?<JJ>*<NN>}'
    cp = nltk.RegexpParser(pattern)
    cs = cp.parse(sent)
    iob_tagged = tree2conlltags(cs)
    # pprint(iob_tagged)
    ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(text)))
    
    doc=nlp(text)
    # pprint([(X.text, X.label_) for X in doc.ents])
    # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
    len(doc.ents)
    labels = [x.label_ for x in doc.ents]
    Counter(labels)
    items = [x.text for x in doc.ents]
    print(Counter(items).most_common(3))
    
    sentences = [x for x in doc.sents]
    temp = len(sentences) - 1
    displacy.render(nlp(str(sentences[temp])),jupyter=True, style='ent')
    displacy.render(nlp(str(sentences[temp])), style='dep')
