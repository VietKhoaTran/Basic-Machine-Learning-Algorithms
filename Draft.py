import gensim
from gensim import corpora
import nltk
from gensim.models import Word2Vec, LdaModel

documents = [
    "Human machine interface for lab abc computer applications",
    "A survey of user opinion of computer system response time",
    "The EPS user interface management system",
    "System and human system engineering testing of EPS",
    "Relation of user perceived response time to error measurement"
]

texts = [nltk.word_tokenize(doc.lower()) for doc in documents]
print(texts)
dictionary = corpora.Dictionary(texts)
#dictionary.filter_extremes(no_below=2, no_above=0.5)
corpus = [dictionary.doc2bow(text) for text in texts]
print(corpus)
print(dictionary.token2id)

model = Word2Vec(texts, vector_size=100, window=5, min_count=1, workers=4)
