import os, subprocess
import pymystem3 as mystem
import time

out_file = 'corpora.txt'
out_dictionary = 'dict.txt'
out_lemmatized = 'lemmed.txt'
out_ngrams = 'ngrams.txt'
out_names = 'filenames.txt'

stopwords_file = 'stop_words.txt'

mystem_path = './mystem' #'./mystem.exe' для Windows
args_lemmatize = [mystem_path, '-ld', out_file, out_lemmatized]
args_dictionary = [mystem_path, '-indl',  out_file, out_dictionary]

python_path = 'python' #'./venv/Scripts/python.exe' для Windows
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


def build_corpora(directory):
    signs = ['.', ',', '«', '»', '(', ')', '-', ':', ';', '?', '!', '@']

    out = open(out_file, mode='w', encoding='utf-8')
    out_lem = open(file=out_lemmatized, encoding='utf-8', mode='w')
    out_filenames = open(out_names, encoding='utf-8', mode='w')

    #printing corpora
    for file in os.scandir("./" + directory):
        if os.DirEntry.is_file(file):
            text = open(file, mode='r', encoding='utf-8').read()
            out.write(file.name + clean_text(text) + '\n')
            out_filenames.write(file.name + '\n')
            #printing lemmed

            lemmer = mystem.Mystem()
            lemmatized = lemmer.lemmatize(clean_text(text))
            for lemma in lemmatized:
                if not lemma in signs:
                    out_lem.write(lemma)
                    out_lem.write(' ')

    #printing dictionary
    subprocess.run(args_dictionary)
    #printing ngrams
    subprocess.run(args_turbotopics)