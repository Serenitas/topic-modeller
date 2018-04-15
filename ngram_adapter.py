import adjective_stemmer as stemmer

adj_endings_male_hard = ['ый', 'ого', 'ому', 'ым', 'ом', 'ой']
adj_endings_male_soft = ['ий', 'его', 'ему', 'им', 'ем']
adj_endings_female_hard = ['ая', 'ой', 'ую']
adj_endings_female_soft = ['яя', 'ей', 'юю']
adj_endings_neuter_hard = ['ое', 'ого', 'ому', 'ым']
adj_endings_neuter_soft = ['ее', 'его', 'ему', 'им', 'ем']


def adj_to_gender(adjective, gender):
    stem = stemmer.stem(adjective)
    ending = adjective[len(stem):]
    if gender == 'м':
        return adjective
    if ending in adj_endings_male_hard:
        if gender == 'ж':
            return stem + adj_endings_female_hard[0]
        return stem + adj_endings_neuter_hard[0]
    if ending in adj_endings_male_soft:
        if stem[-1:] in ['г', 'к', 'х', 'ц']:
            if gender == 'ж':
                return stem + adj_endings_female_hard[0]
            return stem + adj_endings_neuter_hard[0]
        if gender == 'ж':
            return stem + adj_endings_female_soft[0]
        return stem + adj_endings_neuter_soft[0]
    return adjective


def noun_to_genitive(noun, gender):
    stem = noun[:-1]
    ending = noun[-1:]
    if ending == 'ь':
        if gender == 'м':
            return stem + 'я'
        if gender == 'ж':
            return stem + 'и'
    if ending == 'а':
        if stem[-1:] in ['г', 'к', 'х', 'ч', 'щ', 'ш']:
            return stem + 'и'
        return stem + 'ы'
    if ending == 'я':
        if stem[-1:] == 'м':
            return stem + 'ени'
        return stem + 'и'
    if ending == 'е' or ending == 'й':
        return stem + 'я'
    if ending == 'о':
        return stem + 'а'
    return noun + 'а'


def adj_to_genitive(adjective, gender):
    stem = stemmer.stem(adjective)
    ending = adjective[len(stem):]
    if ending in adj_endings_male_hard:
        if gender == 'м':
            return stem + adj_endings_male_hard[1]
        elif gender == 'ж':
            return stem + adj_endings_female_hard[1]
        return stem + adj_endings_neuter_hard[1]
    if ending in adj_endings_male_soft:
        if stem[-1:] in ['г', 'к', 'х', 'ц']:
            if gender == 'м':
                return stem + adj_endings_male_hard[1]
            elif gender == 'ж':
                return stem + adj_endings_female_hard[1]
            return stem + adj_endings_neuter_hard[1]
        if gender == 'м':
            return stem + adj_endings_male_soft[1]
        elif gender == 'ж':
            return stem + adj_endings_female_soft[1]
        return stem + adj_endings_neuter_soft[1]
    return adjective


def build_noun_dictionary(filename):
    dict = {}
    file = open(file=filename, encoding='utf-8').readlines()
    for str in file:
        str = str.strip('\n')
        str = str.split('=')
        if len(str) < 2:
            continue
        lemma = str[0]
        info = str[1]
        if 'S' in info and 'SPRO' not in info:
            if 'муж' in info:
                dict[lemma] = 'м'
            if 'жен' in info:
                dict[lemma] = 'ж'
            if 'сред' in info:
                dict[lemma] = 'с'
    return dict


def build_adj_dictionary(filename):
    dict = {}
    file = open(file=filename, encoding='utf-8').readlines()
    for str in file:
        str = str.strip('\n')
        str = str.split('=')
        if len(str) < 2:
            continue
        lemma = str[0]
        info = str[1]
        if 'A' in info and 'APRO' not in info:
            if 'муж' in info:
                dict[lemma] = 'м'
            if 'жен' in info:
                dict[lemma] = 'ж'
            if 'сред' in info:
                dict[lemma] = 'с'
    return dict


def adapt_ngram(ngram, dictionary, adj_dictionary):
    ngram = ngram.split('|')[0]
    if ngram.count(' ') == 0:
        return ngram
    if ngram.count(' ') == 1:
        word1 = ngram.split(' ')[0]
        word2 = ngram.split(' ')[1]
        if word1 in dictionary and word2 in dictionary:
            word2 = noun_to_genitive(word2, dictionary[word2])
        elif word1 in adj_dictionary and word2 in dictionary:
            word1 = adj_to_gender(word1, dictionary[word2])
        return word1 + ' ' + word2
    if ngram.count(' ') == 2:
        word1 = ngram.split(' ')[0]
        word2 = ngram.split(' ')[1]
        word3 = ngram.split(' ')[2]
        if word1 in dictionary and word2 in dictionary and word3 in dictionary:
            word2 = noun_to_genitive(word2, dictionary[word2])
            word3 = noun_to_genitive(word3, dictionary[word3])
        if word1 in dictionary and word2 in adj_dictionary and word3 in dictionary:
            word2 = adj_to_genitive(word2, dictionary[word3])
            word3 = noun_to_genitive(word3, dictionary[word3])
        return word1 + ' ' + word2 + ' ' + word3


def adapt_ngrams(ngrams_file, dict_file, result_file):
    dictionary = build_noun_dictionary(dict_file)
    adj_dictionary = build_adj_dictionary(dict_file)

    ngrams = open(file=ngrams_file, encoding='utf-8').read().split('\n')
    adapted_ngrams = []
    multiword_only = []
    multiword_adapted = []

    for ngram in ngrams:
        if len(ngram) < 3:
            continue
        ngram = ngram.split('|')[0]
        if ngram.count(' ') == 0:
            adapted_ngrams.append(ngram)
        if ngram.count(' ') == 1:
            multiword_only.append(ngram)
            word1 = ngram.split(' ')[0]
            word2 = ngram.split(' ')[1]
            if word1 in dictionary and word2 in dictionary:
                word2 = noun_to_genitive(word2, dictionary[word2])
                multiword_adapted.append(word1 + ' ' + word2)
            elif word1 in adj_dictionary and word2 in dictionary:
                word1 = adj_to_gender(word1, dictionary[word2])
                multiword_adapted.append(word1 + ' ' + word2)
            adapted_ngrams.append(word1 + ' ' + word2)
        if ngram.count(' ') == 2:
            multiword_only.append(ngram)
            word1 = ngram.split(' ')[0]
            word2 = ngram.split(' ')[1]
            word3 = ngram.split(' ')[2]
            if word1 in dictionary and word2 in dictionary and word3 in dictionary:
                word2 = noun_to_genitive(word2, dictionary[word2])
                word3 = noun_to_genitive(word3, dictionary[word3])
                multiword_adapted.append(word1 + ' ' + word2 + ' ' + word3)
            if word1 in dictionary and word2 in adj_dictionary and word3 in dictionary:
                word2 = adj_to_genitive(word2, dictionary[word3])
                word3 = noun_to_genitive(word3, dictionary[word3])
                multiword_adapted.append(word1 + ' ' + word2 + ' ' + word3)
            adapted_ngrams.append(word1 + ' ' + word2 + ' ' + word3)

    out = open(file=result_file, mode='w', encoding='utf-8')
    out2 = open(file='adapted.txt', mode='w', encoding='utf-8')
    out3 = open(file='multiword_only.txt', mode='w', encoding='utf-8')
    for w in adapted_ngrams:
        #print(w)
        out.write(w + '\n')
    for w in multiword_adapted:
        out2.write(w + '\n')
    for w in multiword_only:
        out3.write(w + '\n')