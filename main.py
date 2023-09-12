import os
import EmailCleaner
import TermFrequencyInverseDocumentFrequency


if __name__ == '__main__':
    corpus = []
    labels = []
    print('In Progress...')
    for email in os.listdir('scam_emails'):
        email_text = "scam_emails/" + email
        returnString = EmailCleaner.email_reduction(email_text)
        corpus.append(returnString)
        TermFrequencyInverseDocumentFrequency.tf_idf(corpus)
        labels.append([0])

    print('In Progress...')
    for email in os.listdir('nonscam_emails'):
        email_text = 'nonscam_emails/' + email
        returnString = EmailCleaner.email_reduction(email_text)
        corpus.append(returnString)
        TermFrequencyInverseDocumentFrequency.tf_idf(corpus)
        labels.append([1])

    print(len(labels))
    print(labels)
