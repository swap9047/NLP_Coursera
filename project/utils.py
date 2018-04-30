import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'starspace_embedding.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    good_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = good_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
#     embed={}
#     with open(embeddings_path) as f:
#         for line in f:
#             (key,value)=line.split()
#             embed[key]=value
#         embed_dim=len(value)
#     import gensim
#     from gensim.models import KeyedVectors
#     wv_embeddings = word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, limit=500000,binary=True) 
#     ls=word_vectors.index2word
#     embed={}
#     embed={word:word_vectors[word] for word in word_vectors.index2word }
#     embed_dim=len(next(iter(embed.values())))
    import csv
    embed={}

    with open(embeddings_path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            embed[row[0]]=np.array(row[1:])
        embed_dim=len(row)-1
        
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embedings.

    ########################
    #### YOUR CODE HERE ####
    ########################

    return (embed,embed_dim) 

def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    # Hint: you have already implemented exactly this function n the 3rd assignment.

    ########################
    #### YOUR CODE HERE ####
    ########################
    mean_vec=np.zeros(dim,dtype=np.float64)
    j=0
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    for i in question.split(' '):
        if i in embeddings:
            mean_vec+=np.array(embeddings[i],dtype=np.float64)
            j+=1
    if j==0:
        return 0
    else:
        return (mean_vec/j)



def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
