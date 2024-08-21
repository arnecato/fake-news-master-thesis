import pandas as pd
import re


isot_true = pd.read_csv('dataset/ISOT/True.csv')
isot_fake = pd.read_csv('dataset/ISOT/Fake.csv')

removal_pattern = r'[,“”’‘():\'-]'
whitespace_pattern = r'\s+'

def apply_and_save(df, filename):
    df['title'] = df['title'].apply(clean_text)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(filename)    

def clean_text(text):
    text = text.lower()
    text = re.sub(removal_pattern, '', text)
    text = text.replace('...', ' . ')
    text = text.replace('. ', ' . ')
    text = re.sub(whitespace_pattern, ' ', text)
    text = text.replace(' . . ', ' . ')
    return text

apply_and_save(isot_true, 'dataset/ISOT/True_cleaned.csv')
apply_and_save(isot_fake, 'dataset/ISOT/Fake_cleaned.csv')
