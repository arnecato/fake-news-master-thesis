from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import pandas as pd
from nltk.tokenize import word_tokenize
import time

chunksize = 1000  # Adjust based on memory capacity

def tokenize_text(text):
    # Tokenizes and lowercases the text
    return word_tokenize(text.lower())

def create_model():
    # Initialize the Word2Vec model
    model = Word2Vec(vector_size=100, window=5, min_count=1, workers=10)

    # Build vocabulary and train incrementally
    with pd.read_csv('dataset/ISOT/True_Fake.csv', chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            print(i)
            # Assuming the text data is in a column named 'title' and 'text'
            text = chunk['title'].str.cat(chunk['text'], sep=' ')
            sentences = text.apply(tokenize_text).tolist()
            #print(sentences)
            # If this is the first chunk, build the vocabulary
            if i == 0:
                model.build_vocab(sentences)
            else:
                # Update the vocabulary for the next chunks
                model.build_vocab(sentences, update=True)
            
            # Train the model on the current chunk
            model.train(sentences, total_examples=len(sentences), epochs=model.epochs)

    # Save the model
    model.save("model/word2vec.model")

model = Word2Vec.load("model/word2vec.model")
#for value in model.wv.index_to_key:
#    print(value)
time0 = time.perf_counter() * 1000
vector = model.wv['guns']
print(time.perf_counter() * 1000 - time0)
sims = model.wv.most_similar('guns', topn=10)
print(vector)
print(sims)