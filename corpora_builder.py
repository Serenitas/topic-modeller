import os, subprocess
import pymystem3 as mystem

directory = 'texts'
out_file = 'corpora.txt'
out_dictionary = 'dict.txt'
out_lemmatized = 'lemmed.txt'
out_ngrams = 'ngrams.txt'

mystem_path = './mystem.exe'
args_lemmatize = [mystem_path, '-ld', out_file, out_lemmatized]
args_dictionary = [mystem_path, '-indl',  out_file, out_dictionary]

python_path = './venv2/Scripts/python.exe'
turbotopics_path = './turbotopics/compute_ngrams.py'
args_turbotopics = [python_path, turbotopics_path, '--corpus=' + out_lemmatized, '--pval=0.01', '--min-count=10',
                    '--out=' + out_ngrams]


def clean_text(text):
    cur_word = ''
    result = ''
    for c in text:
        c = c.lower()
        if c.isalpha() or (cur_word != '' and c == '-'):
            cur_word = cur_word + c
        elif c.isdigit():
            cur_word = ''
        elif cur_word != '':
            result = result + ' ' + cur_word.strip()
            cur_word = ''
    return result


def build_corpora(directory):
    signs = ['.', ',', '«', '»', '(', ')', '-', ':', ';', '?', '!', '@']

    out = open(out_file, mode='w', encoding='utf-8')
    out_lem = open(file=out_lemmatized, encoding='utf-8', mode='w')

    #printing corpora
    for file in os.scandir("./" + directory):
        if os.DirEntry.is_file(file):
            text = open(file, mode='r', encoding='utf-8').read()
            out.write(file.name + clean_text(text) + '\n')

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

build_corpora(directory)