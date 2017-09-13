"""
    Natural Language Processing Lab
    Naive Bayes classifier for word sense disambiguation (WSD)
    Due: 05.05.2017
"""
from collections import defaultdict
from porter import PorterStemmer
import re


def print_dict(dict):
    for key, value in dict.items():
        print(str(key) + " " + str(value))


def print_txt(item, path):
    f = open(path, "w")
    for i in item:
        f.write(i + "\n")
    f.close()


def read_vocabulary(path):
    f = open(path)
    vocab = f.read()
    words = vocab.strip().split(", ")
    vocab = []
    stemmer = PorterStemmer()
    for word in words:
        vocab.append(stemmer.stem(word, 0, len(word)-1))
    return vocab


def read_documents(path, vocab):
    lines = [line.rstrip('\n') for line in open(path)]
    docs = {}
    temp = 0
    for line in lines:
        if line.isdigit():
            temp = line
        else:
            docs[temp] = docs.get(temp, "") + str(line)
    return revise_documents(docs, vocab)


def revise_documents(docs, vocab):
    stemmer = PorterStemmer()
    senses = {}     # {reference:sense}
    for ref, text in docs.items():
        words = re.findall(r"[\w']+", text)
        word_list = []
        for w in words:
            if w == "tag":
                continue
            if w.isdigit() and int(w) > 100000:
                senses[ref] = w
                continue
            if w in vocab:
                word_list.append(stemmer.stem(w.lower(), 0, len(w) - 1))
        docs[ref] = word_list
    return docs, senses


def get_sense_list(senses):     # Converts from { ref1:sense1, ref2:sense2 }
    dict = defaultdict(list)    # to { sense1:[ref1,ref2], sense3:[ref3,ref4] }
    for ref, sense in senses.items():
        dict[sense].append(ref)
    return dict


def get_v(docs):    # Returns count of vocabulary in train set
    count = 0
    for ref, text in docs.items():
        count += len(text)
    return count


def get_total_count_of_sense(reflist, train_docs):  # Returns word count of given sense
    count = 0
    for ref in reflist:
        count += len(train_docs.get(ref))
    return count


def get_count_of_word_in_sense(word, reflist, train_docs):
    count = 0       # Returns count of given word in given sense
    for ref in reflist:       # [ 800001, 800011 ..]
        for w in train_docs.get(ref):     # ['cent','cut','cut']
            if w == word:
                count += 1
    return count


def naive_bayes(sense_ref_pairs, train_docs, test_docs, path):
    output = []
    for ref, test in test_docs.items():
        senses = get_sense_list(sense_ref_pairs)        # {sense:[ref1,ref2]}
        V = get_v(train_docs)       # Total count of words in train set
        probabilities = {}

        for sense, reflist in senses.items():       # Check every sense and add prob to list
            # P(sense|words) = P(word1|sense) P(word2|sense) .. P(words)
            prob = len(senses.get(sense)) / len(sense_ref_pairs)  # P(class) prior

            for word in test:       # P(word|sense) = count(word,sense) + 1 / count(sense) + V
                # selected word count in given sense + 1 / all word count in given sense
                prob *= (get_count_of_word_in_sense(word, reflist, train_docs) + 1) / (get_total_count_of_sense(reflist, train_docs) + V)

            probabilities[sense] = prob

        best = max(probabilities, key=probabilities.get)
        print(str(ref) + " " + str(best))
        output.append(str(ref) + " " + str(best))
    print_txt(output, path)


if __name__ == "__main__":
    vocab = read_vocabulary("./io/generous_vocabulary.txt")
    train_docs, senses = read_documents('./io/Generous_train.txt', vocab)
    test_docs, none = read_documents('./io/Generous_test.txt', vocab)
    print("--- TASK 1: NAIVE BAYES ---")
    naive_bayes(senses, train_docs, test_docs, "output.txt")
