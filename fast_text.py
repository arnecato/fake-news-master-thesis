import io
import fasttext
import fasttext.util
import time

#fasttext.util.download_model('en', if_exists='ignore') 

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def get_model(language, vector_size):    
    ft = fasttext.load_model(f'model/cc.{language}.{vector_size}.bin')
    return ft

model_small = get_model('en', 100)
model_large = get_model('en', 300)
for i in range(10):
    time0 = time.perf_counter()
    model_small.get_word_vector('king')
    print('small', time.perf_counter() - time0)
    time0 = time.perf_counter()
    model_large.get_word_vector('king')
    print('large', time.perf_counter() - time0)
    





