import io
import fasttext
import fasttext.util
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex

#fasttext.util.download_model('en', if_exists='ignore') 

def create_vector_db(model, type):
    vector_dim = 100
    t = AnnoyIndex(vector_dim, type)

    # Add vectors to the index
    for i, word in enumerate(model.get_words()):
        t.add_item(i, model.get_word_vector(word))

    t.build(20)  # 10 trees
    t.save('model/annoy_index.ann')
    return t

def get_model(language, vector_size):    
    return fasttext.load_model(f'model/cc.{language}.{vector_size}.bin')

def get_document_vector(model, text):
    sentence_vectors = []
    for sentence in text.split(' . '):
        sentence_vectors.append(model.get_sentence_vector(sentence))
    print(sentence_vectors)
    return np.mean(sentence_vectors, axis=0) # IMPORTANT TO CHANGE? ***********************

def main():
    dim = 100
    text1 = "positive" # "The X was a New York newspaper published from 1833 until 1950 . It was considered a serious paper"
    text2 = "great" # "The Washington Post locally known as the Post and is an American daily newspaper published in Washington . It is the most widely circulated newspaper in the Washington metropolitan area and has a national audience"
    text3 = "negative" #"I love this . I enjoyed every moment and I feel so positive when I think about everything we were able to do"
    
    model = get_model('en', dim)
    
    text1_v = get_document_vector(model, text1)
    text2_v = get_document_vector(model, text2)
    text3_v = get_document_vector(model, text3)
    print(np.linalg.norm(text1_v - text2_v), np.linalg.norm(text1_v - text3_v), np.linalg.norm(text2_v - text3_v))
    print(cosine_similarity([text1_v], [text2_v])[0][0], cosine_similarity([text1_v], [text3_v])[0][0], cosine_similarity([text2_v], [text3_v])[0][0])
    print(model.get_subwords('positive'))
    word_a = model.get_word_vector('king')
    word_b = model.get_word_vector('man')
    combined_word1 = word_a - word_b
    word_c = model.get_word_vector('woman')
    combined_word2 = word_c + combined_word1
    #create_vector_db(model, 'angular')
    time0 = time.perf_counter()
    vector_db = AnnoyIndex(dim, 'angular')
    vector_db.load('model/annoy_index.ann')
    print(time.perf_counter() - time0)
    time0 = time.perf_counter()
    nearest_neighbor_index = vector_db.get_nns_by_vector(combined_word2, 1)[0]
    print(time.perf_counter() - time0)
    print('annoy:', model.get_words()[nearest_neighbor_index])
    time0 = time.perf_counter()
    print('ft:', model.get_analogies('king', 'man', 'woman'))
    print(time.perf_counter() - time0)
    
    '''chunksize = 50
    with pd.read_csv('dataset/ISOT/True_cleaned.csv', chunksize=chunksize) as reader:
        print(reader)
        idx = 0
        for batch in reader:
            for row in range(idx*chunksize,idx*chunksize + chunksize):
                print(batch.iloc[row]['text'])
            print('hey')'''
if __name__ == '__main__':
    main()





