import pandas as pd
import EmailCleaner
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm

if __name__ == '__main__':
    email_text = "email.txt"
    returnString = EmailCleaner.email_reduction(email_text)
    print(returnString)

