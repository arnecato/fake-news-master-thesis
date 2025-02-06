from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel
import torch
import time
import pandas as pd
import numpy as np
import argparse
import os

torch.set_num_threads(8)

class BERTVectorFactory():
    def __init__(self, model_name='roberta-base'):
        if model_name == 'roberta-base':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        elif model_name == 'roberta-large':
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
            self.model = RobertaModel.from_pretrained(model_name)
        elif model_name == 'bert-base-cased':
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertModel.from_pretrained(model_name)
        elif model_name == 'distilbert-base-cased':
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            self.model = DistilBertModel.from_pretrained(model_name)
        
        #self.model = DistilBertModel.from_pretrained('distilbert-base-cased')
    
    def document_vector(self, text):
        encoded_input = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        #print(type(outputs.last_hidden_state[:, 0, :][0]))
        return np.array(outputs.last_hidden_state[:, 0, :][0], dtype=np.float32)

    def document_vector_batch(self, texts, batch_size=32):
        # Tokenize input texts in batches, padding and truncating to a maximum length of 512 tokens
        encoded_input = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        
        vectors = []

        # Batch process with no gradients
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_input = {key: value[i:i+batch_size] for key, value in encoded_input.items()}
                outputs = self.model(**batch_input)
                # Extract [CLS] token embeddings for the entire batch
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.extend(batch_vectors)

        return np.array(vectors, dtype=np.float32)
    
    def vectorize_dataframe_using_batches(self, load_filepath, save_filepath, columns, batch_size=32):
        # Load the dataframe and concatenate the selected columns into a 'text' field
        df = pd.read_csv(load_filepath)
        df = df[columns]
        df['text'] = df.apply(lambda row: ' . '.join(row.values.astype(str))[:512], axis=1) 
        df = df[['text']]
        time0 = time.perf_counter()
        # Get the list of texts from the DataFrame
        texts = df['text'].tolist()
        # Perform batch encoding using the batch version of document_vector
        encoded_vectors = self.document_vector_batch(texts, batch_size=batch_size)
        # Store the vectors back into the DataFrame as a new column
        df['vector'] = list(encoded_vectors)
        print('Time:', time.perf_counter() - time0)
        # Optionally, save the resulting DataFrame with vectors to a file
        df.to_hdf(save_filepath, key='df', mode='w')

    def vectorize_dataframe_first_characters(self, load_filepath, save_filepath, columns, first_characters):
        df = pd.read_csv(load_filepath)
        df = df[columns]
        df['text'] = df.apply(lambda row: ' . '.join(row.values.astype(str)), axis=1)
        df = df[['text']]
        time0 = time.perf_counter()

        df['vector'] = df['text'].apply(self.document_vector)
        print('Time:', time.perf_counter() - time0)
        df.to_hdf(save_filepath, key='df', mode='w')
    
    def load_vectorized_dataframe(self, load_filepath):
        return pd.read_hdf(load_filepath, key='df')
    
def main():
    parser = argparse.ArgumentParser(description='BERT Vectorization')
    parser.add_argument('--model_name', type=str, required=True, choices=['bert-base-cased','distilbert-base-cased', 'roberta-base', 'roberta-large'], help='Path to the true news CSV file')
    args = parser.parse_args()
    bert_vfac = BERTVectorFactory(model_name=args.model_name)
    #bert_vfac.vectorize_dataframe_first_characters('dataset/ISOT/True.csv', 'dataset/ISOT/True_256_BERT.h5', ['title', 'text'], 256)
    #bert_vfac.vectorize_dataframe_first_characters('dataset/ISOT/Fake.csv', 'dataset/ISOT/Fake_256_BERT.h5', ['title', 'text'], 256)
    true_file_path = f'dataset/ISOT/True_{args.model_name}.h5'
    fake_file_path = f'dataset/ISOT/Fake_{args.model_name}.h5'

    if os.path.exists(true_file_path):
        print(f"{true_file_path} already exists. Skipping vectorization for True news.")
    else:
        bert_vfac.vectorize_dataframe_using_batches(f'dataset/ISOT/True.csv', true_file_path, ['title', 'text'])
    if os.path.exists(fake_file_path):
        print(f"{fake_file_path} already exists. Skipping vectorization for Fake news.")
    else:
        bert_vfac.vectorize_dataframe_using_batches(f'dataset/ISOT/Fake.csv', fake_file_path, ['title', 'text'])

    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased')
    '''tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    model = DistilBertModel.from_pretrained('distilbert-base-cased')
    text = "Trump wants Postal Service to charge 'much more' for Amazon shipments. SEATTLE/WASHINGTON (Reuters) - President Donald Trump called on the U.S. Postal Service on Friday to charge “much more” to ship packages for Amazon (AMZN.O), picking another fight with an online retail giant he has criticized in the past.     “Why is the United States Post Office, which is losing many billions of dollars a year, while charging Amazon and others so little to deliver their packages, making Amazon richer and the Post Office dumber and poorer? Should be charging MUCH MORE!” Trump wrote on Twitter.  The president’s tweet drew fresh attention to the fragile finances of the Postal Service at a time when tens of millions of parcels have just been shipped all over the country for the holiday season.  The U.S. Postal Service, which runs at a big loss, is an independent agency within the federal government and does not receive tax dollars for operating expenses, according to its website.  Package delivery has become an increasingly important part of its business as the Internet has led to a sharp decline in the amount of first-class letters. The president does not determine postal rates. They are set by the Postal Regulatory Commission, an independent government agency with commissioners selected by the president from both political parties. That panel raised prices on packages by almost 2 percent in November.  Amazon was founded by Jeff Bezos, who remains the chief executive officer of the retail company and is the richest person in the world, according to Bloomberg News. Bezos also owns The Washington Post, a newspaper Trump has repeatedly railed against in his criticisms of the news media. In tweets over the past year, Trump has said the “Amazon Washington Post” fabricated stories. He has said Amazon does not pay sales tax, which is not true, and so hurts other retailers, part of a pattern by the former businessman and reality television host of periodically turning his ire on big American companies since he took office in January. Daniel Ives, a research analyst at GBH Insights, said Trump’s comment could be taken as a warning to the retail giant. However, he said he was not concerned for Amazon. “We do not see any price hikes in the future. However, that is a risk that Amazon is clearly aware of and (it) is building out its distribution (system) aggressively,” he said. Amazon has shown interest in the past in shifting into its own delivery service, including testing drones for deliveries."
    encoded_input = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    # Forward pass, get hidden states
    with torch.no_grad():
        time0 = time.perf_counter()
        outputs = model(**encoded_input)
        time1 = time.perf_counter()

    # Get the embeddings for the [CLS] token
    cls_embeddings = outputs.last_hidden_state[:, 0, :]

    print(cls_embeddings)
    print('Time:', time1 - time0)'''

if __name__ == '__main__':
    main()

