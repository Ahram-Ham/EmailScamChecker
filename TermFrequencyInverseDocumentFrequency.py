from sklearn.feature_extraction.text import TfidfVectorizer

def tf_idf(corpus):
    tfidf = TfidfVectorizer()
    result = tfidf.fit_transform(corpus)
    result = result.toarray()
    print(result)
    
    """print('\nidf values:')
    for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
        print(ele1,  ':', ele2)"""
