import math, collections, operator


def build_td_idf_dict(corpora_file):
    corpora_origin = open(corpora_file, mode='r', encoding='utf-8').readlines()
    corpora = []
    for text in corpora_origin:
        text = text.split(' ')
        res = []
        for w in text:
            res.append(w.strip('\n'))
        corpora.append(res)
    documents_list = []
    for text in corpora:
        tf_idf_dictionary = {}
        computed_tf = compute_tf(text)
        for word in computed_tf:
            word = word.strip('\n')
            tf_idf_dictionary[word] = computed_tf[word] * compute_idf(word, corpora)
        sorted_dict = sorted(tf_idf_dictionary.items(), key=operator.itemgetter(1), reverse=True)
        documents_list.append(sorted_dict)
    return documents_list


def compute_tf(text):
    tf_text = collections.Counter(text)
    for i in tf_text:
        tf_text[i] = tf_text[i] / float(len(text))
    return tf_text


def compute_idf(word, corpus):
    return math.log(len(corpus)/sum([1.0 for i in corpus if word in i]))


print(build_td_idf_dict('test.txt'))