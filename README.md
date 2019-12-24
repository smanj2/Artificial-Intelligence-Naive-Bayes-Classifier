# B551 Assignment 3: Probability and Statistical Learning for NLP
##### Submission by Sri Harsha Manjunath - srmanj@iu.edu; Vijayalaxmi Bhimrao Maigur - vbmaigur@iu.edu; Disha Talreja - dtalreja@iu.edu
###### Fall 2019

## Part 3: Spam classification

Let’s consider a straightforward document classification problem: deciding whether or not an e-mail is spam.
We’ll use a bag-of-words model, which means that we’ll represent a document in terms of just an unordered
“bag” of words instead of modeling anything about the grammatical structure of the document. If, for
example, there are 100,000 words in the English language, then a document can be represented as a 100,000-
dimensional vector, where the entries in the vector corresponds to a binary value — 1 if the word appears
in the document and 0 otherwise. Of course, most vectors will be sparse (most entries are zero).
Implement a Naive Bayes classifier for this problem. For a given document D, we’ll need to evaluate
P(S = 1|w1, w2, ..., wn), the posterior probability that a document is spam given the features (words) in
that document. Make the Naive Bayes assumption, which says that for any i 6= j, wi
is independent from
wj given S. (It may be more convenient to evaluate the likelihood (or “odds”) ratio of P (S=1|w1,...wn)
P (S=0|w1,...,wn)
, and
compare that to a threshold to decide if a document is spam or non-spam.)
To help you get started, we’ve provided a dataset in your repo of known spam and known non-spam emails,
split into a training set and a testing set. Your program should accept command line arguments like this:
./spam.py training-directory testing-directory output-file
The training-directory can be assumed to contain two subdirectories called spam and notspam, containing
email files that can be used to estimate the needed probabilities of the model. The testing-directory contains
test emails, one per file; your program should output a output-file in a format like this:
00393.85c9cd10122736d443e69db6fce3ad3f spam
01064.50715ffeb13446500895836b77fcee09 notspam
and so on, where the first part of each line is a filename and the second is predicted class (spam or notspam).

## Solution 
The solution implements the following sections as explained - 

#### Reads the input files - 
- Since the HTML mail files needed considerable amount of cleaning, we chose to only consider the subject of the mails for the bag of words model. Although we did end up cleaning the entire e-mails and run that through the Naive Bayes, we did not see any significant improvement from it.
- All special characters and numerical characters were also eleminated when creating the words/tokens (in the bag of words)

#### Naive Bayes Model - 
- The total number of spam and ham messages were counted and generated corresponding probabilities
- Counted the number of times a word occurred under a spam mail and a non-spam mail and generated corresponding probabilities
- When given a new file for evaluation, the model sums up the log probabilities all the words/tokens in the given mail using the spam probabilities and ham probabilities
- In case of a word that has not been seen before, a "generic" probability is put in place P(A given mail is Spam) or P(A given mail is Ham)
- Finally, we compute P(Spam)/ P(Spam + Ham). If this is greater than 0.5 it is classified as spam, else ham

#### Output - 
- The implementation carries out the above operations on the contents of the test folder and stores the results in the expected format in the user mentioned text file

####  Accuracy - 
We have observed an accuracy 90% on the test dataset provided as part of the assignment.
