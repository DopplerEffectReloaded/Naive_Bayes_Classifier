# import math
import numpy as np
# import matplotlib
import os
from sklearn.naive_bayes import MultinomialNB
import re

# path_ham = "raw_data/enron1/ham/"
path_mails = "raw_data/enron1/mails/"
path_test_mails = "raw_data/enron2/mails/"
# ham_mail_names = [i for i in os.listdir(path_ham)]
mail_names = [i for i in os.listdir(path_mails)]
# words_list_ham = []
mail_test_names = [i for i in os.listdir(path_test_mails)]

def obtain_words(path, mail_names):
    word_list = []
    for mail in mail_names:
        f = open(path+mail, 'r')
        try:
            output = f.read().replace('\n', ' ')
            mail = output.split(' ')
            word_list.extend(mail)
        except UnicodeDecodeError:
            continue
        word_list.append("\n")
    return word_list

# words_list_ham = obtain_words(path_ham, ham_mail_names)
words_list = obtain_words(path_mails, mail_names)
# words_list_ham_set = {elem for elem in list(set(words_list_ham)) if not elem.isdigit()}
words_list = [elem for elem in list(set(words_list)) if not elem.isdigit()]

for index, word in enumerate(words_list):
    if (word.isalpha() == False) or (len(word)==1):
        del words_list[index]

def build_features(mail_names, path, words_list):
    mail_names.sort()
    features_matrix = np.zeros((len(mail_names), len(words_list)))
    for email_index, email in enumerate(mail_names):
        f = open(os.path.join(path, email))
        try:
            for line_index, line in enumerate(f):
                if line_index == 2:
                    words = line.split()
                    for word_index, word in enumerate(words_list):
                        features_matrix[email_index, word_index] = words.count(word)
        except UnicodeDecodeError:
            continue
    return features_matrix

def build_labels(mail_names):
    mail_names.sort()
    labels_matrix = np.zeros(len(mail_names))

    for index, email in enumerate(mail_names):
        labels_matrix[index] = 1 if re.search('spam*', email) else 0
    
    return labels_matrix

print("Building training features and labels")
features_train = build_features(mail_names, path_mails,words_list)
labels_train = build_labels(mail_names)

classifier = MultinomialNB()
print("Training the classifier")
classifier.fit(features_train, labels_train)

print("Building test features and labels")
features_test = build_features(mail_test_names, path_test_mails, words_list)
labels_test = build_labels(mail_test_names)

print("Calcuating accuracy of the tested classifier")
accuracy = classifier.score(features_test, labels_test)
print("Accuracy: ", accuracy*100)