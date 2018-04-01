import pymystem3 as mystem
import time

signs = ['.', ',', '«', '»', '(', ')', '-', ':', ';', '?', '!', '@']
signs_str = '.,@"«»()-_:;?!#$%^*+=-/\|{}[]\' '

infile = open(file='big_ngrams.txt', encoding='utf-8').read()
origin_0 = open(file='big_origin.txt', encoding='utf-8').read()
origin = ''
for char in origin_0:
    if char.isalnum() or char == ' ':
        origin = origin + char.lower()

lemmer = mystem.Mystem()

lemmed = lemmer.lemmatize(origin)
while lemmed.__contains__(' '):
    lemmed.remove(' ')
while lemmed.__contains__('  '):
        lemmed.remove('  ')
while lemmed.__contains__('\n'):
    lemmed.remove('\n')
origin = origin.split(' ')
while origin.__contains__(''):
    origin.remove('')

outfile = open(file='big_adapted_ngrams.txt', encoding='utf-8', mode='w')



ngrams = infile.split('\n')
adapted_ngrams = []


start = time.time()
for ngram in ngrams:
    value = ngram.split('|')[0]
    if len(value) > 1:
        if value.find(' ') == -1:
            adapted_ngrams.append(value)
            adapted_ngrams.append('\n')
        else:
            lem_word_1 = value.split(' ')[0]
            lem_word_2 = value.split(' ')[1]
            for i in range(len(origin) - 1):
                word_1 = origin[i].strip(signs_str).lower()
                word_2 = origin[i + 1].strip(signs_str).lower()
                lem1 = lemmed[i]
                lem2 = lemmed[i + 1]
                if lem1 == lem_word_1 and lem2 == lem_word_2:
                        if word_1 == lem_word_1 or word_2 == lem_word_2:
                            adapted_ngrams.append(word_1 + ' ' + word_2)
                            adapted_ngrams.append('\n')
                            print(word_1 + ' ' + word_2)
                            break
                        else:
                            print('Вариант фразы: ' + word_1 + ' ' + word_2)
print("Time: ", time.time() - start, ' sec.')

for ngram in adapted_ngrams:
    outfile.write(ngram)



