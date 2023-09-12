from sklearn.svm import SVC
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# Corpus contains tf-idf values, while labels contains binary numbers with 0 being a scam and 1 being a non-scam
def svm(corpus, labels):
    svm_classifier = SVC(kernel='linear')
    data, labels = shuffle_data_and_labels(corpus, labels)
    stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Split data into testing and training sets
    for train_index, test_index in stratified_splitter.split(data, labels):
        x_train, x_test = [data[i] for i in train_index], [data[i] for i in test_index]
        y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

    svm_classifier.fit(x_train, y_train)
    y_pred = svm_classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    print(report)
    print(confusion)


def shuffle_data_and_labels(data, labels):
    if len(data) != len(labels):
        raise ValueError('Data and labels must have the same length')

    data_and_labels = list(zip(data, labels))
    random.shuffle(data_and_labels)
    shuffled_data, shuffled_labels = zip(data_and_labels)
    return list(shuffled_data), list(shuffled_labels)