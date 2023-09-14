import os
import EmailCleaner
import TermFrequencyInverseDocumentFrequency
import SupportVectorMachine
import numpy as np
from joblib import load


def create_model():
    corpus = []
    labels = []
    vals = []

    print('In Progress...')
    for email in os.listdir('scam_emails'):
        email_text = "scam_emails/" + email
        return_string = EmailCleaner.email_reduction(email_text)
        corpus.append(return_string)
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email
        vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list
        labels.append(0)

    print('In Progress...')
    for email in os.listdir('nonscam_emails'):
        email_text = 'nonscam_emails/' + email
        print(email_text)
        return_string = EmailCleaner.email_reduction(email_text)
        corpus.append(return_string)
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email
        vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list
        labels.append(1)

    max_email_length = get_max_len(vals)
    vals = feature_engineering(vals, max_email_length)

    SupportVectorMachine.svm(vals, labels)


def run_model():
    # TODO: FILL THIS OUT
    print('Do nothing')


def get_max_len(vals):
    max_email_length = max(len(preprocessed_email) for preprocessed_email in vals)
    return max_email_length


def feature_engineering(vals, max_email_length):
    feature_vectors = []

    for email in vals:
        email_length = len(email)

        # Pad the email to max_email_length with zeros (or any padding value)
        padding = np.zeros(max_email_length - email_length)
        padded_email = np.concatenate((email, padding))

        # Create binary padding indicators (1 for data, 0 for padding)
        padding_indicators = np.concatenate((np.ones(len(email)), np.zeros(len(padding))))

        # Combine the email tokens and padding indicators into a single list
        combined_features = np.concatenate((padded_email, padding_indicators))

        # Append the combined feature vector to the list
        feature_vectors.append(combined_features)

    return feature_vectors


if __name__ == '__main__':
    create_model()


