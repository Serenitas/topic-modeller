import os


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


directory = 'texts'
out_file = 'corpora.txt'

out = open(out_file, mode='w', encoding='utf-8')
for file in os.scandir("./" + directory):
    if os.DirEntry.is_file(file):
        text = open(file, mode='r', encoding='utf-8').read()
        out.write(file.name + clean_text(text) + '\n')

