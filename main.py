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
    # Preprocess scam emails then convert them to number matrices and add it to final list of values to be run on.
    for email in os.listdir('scam_emails'):
        email_text = "scam_emails/" + email
        return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
        corpus.append(return_string)  # Add the reduced email to a list of emails.
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email.
        vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list.
        labels.append(1)  # 1 for scam.

    print('In Progress...')
    # Preprocess nonscam emails then convert them to number matrices and add it to final list of values to be run on.
    for email in os.listdir('nonscam_emails'):
        email_text = 'nonscam_emails/' + email
        return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
        corpus.append(return_string) # Add the reduced email to a list of emails.
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string])  # TF-IDF for a single email.
        vals.append(val[0])  # TF-IDF matrix is a list of lists, so take the first list.
        labels.append(0)  # 0 for nonscam.

    max_email_length = get_max_len(vals)  # Need to get the longest email length.
    vals = feature_engineering(vals, max_email_length)

    SupportVectorMachine.svm(vals, labels)  # create model.


def run_model():
    corpus = []
    vals = []
    emails_to_predict = []
    loaded_svm_model = load('model/svm_model.joblib')
    for email in os.listdir('unassigned_emails'):
        email_text = 'unassigned_emails/' + email
        with open(email_text, "r") as file:
            text = file.read()
        emails_to_predict.append(text)  # Used later to show the email content.
        return_string = EmailCleaner.email_reduction(email_text)  # Preprocess the email.
        corpus.append(return_string)  # Add the reduced email to a list of emails that will be labeled.
        val = TermFrequencyInverseDocumentFrequency.tf_idf([return_string]) # TF-IDF for a single email.
        vals.append(val[0]) # TF-IDF matrix is a list of lists, so take the first list.
    feature_vectors = feature_engineering(vals, 194)  # 194 because it was the longest length email in training set.

    predictions = loaded_svm_model.predict(feature_vectors)  # Get prediction.
    for i, prediction in enumerate(predictions):
        email_text = emails_to_predict[i]  # Load email content to be printed.
        print(f"Email {i + 1}: Predicted Label - {prediction}, Email Text: \n{email_text}")
        print("\n" + "----------------------------------------------------------" + "\n")


def get_max_len(vals):
    max_email_length = max(len(preprocessed_email) for preprocessed_email in vals)  # Get the max length out of all emails.
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
    run_model()
    # create_model()
