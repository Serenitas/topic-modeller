import pymystem3 as mystem
from pprint import *

text = open(file='big_origin.txt', encoding='utf-8') .read()
out = open(file='big_lemmed.txt', encoding='utf-8', mode='w')
lemmer = mystem.Mystem()
lemmatized = lemmer.lemmatize(text)

signs = ['.', ',', '«', '»', '(', ')', '-', ':', ';', '?', '!', '@']

for lemma in lemmatized:
    if not signs.__contains__(lemma):
        out.write(lemma)
        out.write(' ')



