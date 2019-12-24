#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: srmanj - Sri Harsha Manjunath; vbmaigur - Vijaylaxmi Maigur; dtalreja - Disha Talreja
#
#

from glob import iglob
import numpy as np
import os
import re
import sys


def read_data(spam_d, ham_d, test_d):
    test_data = []
    data = []

    for files in iglob(os.path.join(spam_d, '*')):
        with open(files, encoding="utf8", errors='ignore') as file:
            for line in file:
                # Filter for sentences that start with the keyword subject
                if line.startswith("Subject:"):
                    subject = re.sub("^Subject: ", "", line).strip()
                    data.append((subject, True))

    for files in iglob(os.path.join(ham_d, '*')):
        with open(files, encoding="utf8", errors='ignore') as file:
            for line in file:
                # Filter for sentences that start with the keyword subject
                if line.startswith("Subject:"):
                    subject = re.sub("^Subject: ", "", line).strip()
                    data.append((subject, False))

    for files in iglob(os.path.join(test_d, '*')):
        with open(files, encoding="utf8", errors='ignore') as file:
            for find_subject in file:
                if find_subject.startswith("Subject:"):
                    subject = re.sub("^Subject: ", "", find_subject).strip()
                    test_data.append(subject)
    return data, test_data


def generate_bag(mail):
    mail = mail.lower()
    bagged_mail = re.findall("[a-z]+", mail)
    return set(bagged_mail)


def predict(file, word_prob):
    # Generate the bag of words for a given file
    bag_of_words = generate_bag(file)
    spam_prob = ham_prob = 0

    for word, spam, ham in word_prob:
        if word in bag_of_words:  # Seen words
            spam_prob += np.log(spam)
            ham_prob += np.log(ham)
        else:  # Incase of unseen words
            spam_prob += np.log(1.0 - spam)
            ham_prob += np.log(1.0 - ham)
    p_spam = np.exp(spam_prob) / (np.exp(spam_prob) + np.exp(ham_prob))
    return p_spam


def create_output_string(test_d, word_prob):
    #print(word_prob)
    string = ''
    for files in iglob(os.path.join(test_d, '*')):
        with open(files, encoding="utf8", errors='ignore') as file:
            filename = file.name[len(test_d):len(file.name)]
            for line in file:
                if line.startswith("Subject:"):
                    subject = re.sub("^Subject: ", "", line).strip()
                    if predict(subject, word_prob) > 0.5:
                        string = string + '\n'+filename+' '+'spam'
                    else:
                        string = string + '\n'+filename+' '+'notspam'
    string = string[1:]
    return string


if __name__ == "__main__":
    if(len(sys.argv) != 4):
        raise Exception(
            "usage: ./spam.py training-directory testing-directory output-file")

    spam_d = sys.argv[1]+'/spam/'
    ham_d = sys.argv[1]+'/notspam/'
    test_d = sys.argv[2]+'/'

    # Read the training and testing data
    train_data, test_data = read_data(spam_d, ham_d, test_d)

    # Train the Naive Bayes Model
    # Step 1 - Counting the number of spams and non spams
    total_spam = total_ham = 0

    for key, value in train_data:
        if value:
            total_spam += 1
        else:
            total_ham += 1

    # Step 2 - Count the number of times a word occurred as a spam and a ham
    word_count = {}
    for key, val in train_data:
        for word in generate_bag(key):
            if word not in word_count.keys():
                word_count[word] = [0, 0]
            if val:
                word_count[word][0] += 1
            else:
                word_count[word][1] += 1

    # Step 3 - Compute probabilities using the counts
    word_prob = []
    for w, (spam, not_spam) in word_count.items():
        # compute the prob of spam and ham
        word_prob.append((w, (0.5 + spam) / (1 + total_spam),
                          (0.5 + not_spam) / (1 + total_ham)))

    output_str = create_output_string(test_d, word_prob)

    with open(sys.argv[3], "w") as file:
        print(output_str, file=file)
