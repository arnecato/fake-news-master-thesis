import io
import fasttext
import fasttext.util
import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#fasttext.util.download_model('en', if_exists='ignore') 

def get_model(language, vector_size):    
    return fasttext.load_model(f'model/cc.{language}.{vector_size}.bin')

def get_document_vector(model, text):
    sentence_vectors = []
    for sentence in text.split(' . '):
        sentence_vectors.append(model.get_sentence_vector(sentence))
    print(sentence_vectors)
    return np.mean(sentence_vectors, axis=0)

def main():
    text1 = "The X was a New York newspaper published from 1833 until 1950 . It was considered a serious paper . "
    text2 = "The Washington Post locally known as the Post and is an American daily newspaper published in Washington . It is the most widely circulated newspaper in the Washington metropolitan area and has a national audience"
    text3 = "I love this . I enjoyed every moment and I feel so positive when I think about everything we were able to do . "
    
    model = get_model('en', 100)
    model.get_sentence_vector
    text1_v = get_document_vector(model, text1)
    text2_v = get_document_vector(model, text2)
    text3_v = get_document_vector(model, text3)
    print(np.linalg.norm(text1_v - text2_v), np.linalg.norm(text1_v - text3_v), np.linalg.norm(text2_v - text3_v))
    print(cosine_similarity([text1_v], [text2_v])[0][0], cosine_similarity([text1_v], [text3_v])[0][0], cosine_similarity([text2_v], [text3_v])[0][0])
    
    

    
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





