from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(corpus):
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(corpus)
    result = result.toarray()
    print(result)
    return result



