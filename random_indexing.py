import random
import pandas as pd
import numpy as np

# Step 1: Initialize Parameters
D = 100  # Dimensionality of the random index vectors
C = 5
     # Context window size (number of words before and after the target word)
N = 10    # Number of non-zero elements in the random index vectors

# Step 2: Initialize Data Structures
word_index_vectors = {}   # Dictionary to store the random index vectors for each word
word_context_vectors = {} # Dictionary to store the context vectors for each word

# Step 3: Create a Random Index Vector
def create_random_index_vector(D, N):
    vector = np.zeros(D)
    non_zero_positions = random.sample(range(D), N)  # Select N random positions
    for pos in non_zero_positions:
        vector[pos] = random.choice([-1, 1])  # Assign either +1 or -1 to these positions
    return vector

# Step 4: Update Context Vectors
def update_context_vectors(word_context_vectors, target_word, context_words, word_index_vectors):
    if target_word not in word_context_vectors:
        word_context_vectors[target_word] = [0] * D  # Initialize context vector if not exists
    
    for context_word in context_words:
        if context_word not in word_index_vectors:
            word_index_vectors[context_word] = create_random_index_vector(D, N)
        
        # Update the target word's context vector
        word_context_vectors[target_word] = [a + b for a, b in zip(word_context_vectors[target_word], word_index_vectors[context_word])]

# Step 5: Process Documents
def process_documents(documents):
    for document in documents:
        words = document.split()  # Assuming documents are provided as strings of words
        for i, word in enumerate(words):
            # Define the context window
            context_start = max(0, i - C)
            context_end = min(len(words), i + C + 1)
            context_words = words[context_start:i] + words[i+1:context_end]  # Exclude the target word itself
            update_context_vectors(word_context_vectors, word, context_words, word_index_vectors)
 
# Step 6: Output the Word Context Vectors
def get_word_context_vectors():
    return word_context_vectors

def cosine_similarity(vector_a, vector_b):
    # Calculate the dot product
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the magnitudes (norms) of the vectors
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # Calculate cosine similarity
    cosine_similarity = dot_product / (magnitude_a * magnitude_b)
    return cosine_similarity

def main():
    #df, df_tfidfvect = generate_vocabulary_and_word_matrix()
    # load directly
    #true = pd.read_csv('dataset/ISOT/True.csv')
    #fake = pd.read_csv('dataset/ISOT/Fake.csv')
    #df = pd.concat([true, fake])
    #df.to_csv('dataset/ISOT/True_Fake.csv')
    chunksize = 50
    with pd.read_csv('dataset/ISOT/True_Fake.csv', chunksize=chunksize) as reader:
        print(reader)
        idx = 0
        for batch in reader:
            #print(batch)
            process_documents(batch['text'])
            print(f'Processing {idx}')
            idx += 1
            # check vectors
            context_vectors = get_word_context_vectors()
            if context_vectors.get('capture') != None and context_vectors.get('verdict') != None:
                a1 = np.array(context_vectors['ruling'])
                a2 = np.array(context_vectors['verdict'])
                b1 = np.array(context_vectors['seizure'])
                b2 = np.array(context_vectors['capture'])

                #print('business', business, 'budget', budget, 'abortion', abortion, 'woman', women)        
                #print(np.dot(a1, a2), np.dot(b1, b2), np.dot(a1, b1), np.dot(a1, b2), '|', np.dot(a2, b1), np.dot(a2, b2))
                print(cosine_similarity(a1, a2), cosine_similarity(b1, b2), cosine_similarity(a1, b1), cosine_similarity(a1, b2))
                print(np.linalg.norm(a1 - a2), np.linalg.norm(b1 - b2), np.linalg.norm(a1 - b2))
if __name__ == '__main__':
    main()