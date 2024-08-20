import pandas as pd
import string

isot_true = pd.read_csv('dataset/ISOT/True-short.csv')

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)        

text = isot_true.iloc[0]['title'] + ". " + isot_true.iloc[0]['text']
print(text)
text = text.lower()
print('---\n', text)
text = remove_punctuation(text)
print('---\n', text)

