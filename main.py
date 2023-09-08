import pandas as pd
import os
import EmailCleaner
import TermFrequencyInverseDocumentFrequency
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from os import listdir

if __name__ == '__main__':
    corpus = []
    for email in os.listdir('emails'):
        email_text = "emails/" + email
        returnString = EmailCleaner.email_reduction(email_text)
        corpus.append(returnString)
        TermFrequencyInverseDocumentFrequency.tf_idf(corpus)

