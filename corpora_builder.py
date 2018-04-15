import os, subprocess
import pymystem3 as mystem
import time, ngram_adapter
from sys import platform

out_file = 'corpora.txt'
out_dictionary = 'dict.txt'
out_lemmatized = 'lemmed.txt'
out_ngrams = 'ngrams.txt'
out_names = 'filenames.txt'

stopwords_file = 'stop_words.txt'

mystem_path = './mystem' #'./mystem.exe' для Windows
if platform == "win32":
    mystem_path = './mystem.exe'
args_lemmatize = [mystem_path, '-ld', out_file, out_lemmatized]
args_dictionary = [mystem_path, '-indl',  out_file, out_dictionary]

python_path = 'python' #'./venv/Scripts/python.exe' для Windows
if platform == "win32":
    python_path = './venv2/Scripts/python.exe'

turbotopics_path = './turbotopics/compute_ngrams.py'
args_turbotopics = [python_path, turbotopics_path, '--corpus=' + out_lemmatized, '--pval=0.001', '--min-count=30',
                    '--out=' + out_ngrams]


def clean_text(text):
    stopwords = open(stopwords_file, mode='r', encoding='utf-8').read()
    stopwords = stopwords.split('\n')

    cur_word = ''
    result = ''
    for c in text:
        c = c.lower()
        if c.isalpha() or (cur_word != '' and c == '-'):
            cur_word = cur_word + c
        elif c.isdigit():
            cur_word = ''
        elif cur_word != '':
            if not cur_word in stopwords:
                result = result + ' ' + cur_word.strip()
            cur_word = ''
    return result


def print_ngrams_by_doc(outfile, ngram_file, corpora_file):
    out = open(outfile, 'w', encoding='utf-8')
    ngrams = open(ngram_file, 'r', encoding='utf-8').read().split('\n')
    corpora = open(corpora_file, 'r', encoding='utf-8').readlines()

    dict = ngram_adapter.build_noun_dictionary(out_dictionary)
    adict = ngram_adapter.build_adj_dictionary(out_dictionary)

    for text in corpora:
        found = []
        text = text.strip('\n').split(' ')
        while '' in text:
            text.remove('')
        if len(text) < 1:
            continue
        out.write(text[0] + ':')
        for i in range(1, len(text) - 1):
            ngram = text[i] + ' ' + text[i + 1]
            if ngram in ngrams and ngram not in found:
                adapted_ngram = ngram_adapter.adapt_ngram(ngram, dict, adict)
                out.write(adapted_ngram + '|')
                found.append(ngram)
            if (i < len(text) - 2):
                ngram = ngram + ' ' + text[i + 2]
                if ngram in ngrams and ngram not in found:
                    adapted_ngram = ngram_adapter.adapt_ngram(ngram, dict, adict)
                    out.write(adapted_ngram + '|')
                    found.append(ngram)
            if (i < len(text) - 3):
                ngram = ngram + ' ' + text[i + 3]
                if ngram in ngrams and ngram not in found:
                    adapted_ngram = ngram_adapter.adapt_ngram(ngram, dict, adict)
                    out.write(adapted_ngram + '|')
                    found.append(ngram)
        out.write('\n')

def build_corpora(directory):
    signs = ['.', ',', '«', '»', '(', ')', '-', ':', ';', '?', '!', '@']

    out = open(out_file, mode='w', encoding='utf-8')
    out_lem = open(file=out_lemmatized, encoding='utf-8', mode='w')
    out_filenames = open(out_names, encoding='utf-8', mode='w')

    #printing corpora
    print("Printing corpora")
    for file in os.scandir("./" + directory):
        if os.DirEntry.is_file(file):
            text = open(file, mode='r', encoding='utf-8').read()
            out.write(file.name + clean_text(text) + '\n')
            out_filenames.write(file.name + '\n')
            #printing lemmed
            print("Printing lemmatized corpora")
            lemmer = mystem.Mystem()
            lemmatized = lemmer.lemmatize(clean_text(text))
            out_lem.write(file.name + ' ')
            for lemma in lemmatized:
                if not lemma in signs:
                    out_lem.write(lemma)
                    out_lem.write(' ')

    #printing dictionary
    print("Printing morphological dictionary")
    subprocess.run(args_dictionary)
    #printing ngrams
    print("Printing ngrams")
    subprocess.run(args_turbotopics, stdout=None)

