from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import nltk
#nltk.download('words') # needs to be run once
#nltk.download('stopwords') # # needs to be run once
from nltk.corpus import words
english_words = set(words.words())

def clean_words_in_df(df):
    ''' cleans the dataframe for any non-English and stop words '''
    for column in df.columns:
        df[column] = df[column].apply(lambda x: ' '.join(word for word in str(x).split() if word.lower() in english_words and word.lower() not in nltk.corpus.stopwords.words('english')))

def generate_vocabulary_and_word_matrix():
    print('Reading datasets...')
    df_true = pd.read_csv('dataset/ISOT/True-short.csv')
    clean_words_in_df(df_true)
    df_fake = pd.read_csv('dataset/ISOT/Fake-short.csv')
    clean_words_in_df(df_fake)
    print(len(df_true), len(df_fake))
    df = pd.concat([df_true, df_fake])

    df['title_text'] = df['title'] + ' ' + df['text']
    documents = df['title_text'].tolist()

    tfidf_vectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english', token_pattern='(?u)\\b[a-zA-Z]{2,}\\b')

    # convert into matrix
    tfidf_wm = tfidf_vectorizer.fit_transform(documents)

    # Getting feature names
    tfidf_tokens = tfidf_vectorizer.get_feature_names_out()

    # Creating DataFrame for TfidfVectorizer
    df_tfidfvect = pd.DataFrame(data=tfidf_wm.toarray(), columns=tfidf_tokens)
    #df_tfidfvect.to_csv('dataset/tfidf_matrix.csv', index=False)
    print("\nTF-IDF Matrix:\n", df_tfidfvect)
    return df, df_tfidfvect

def visualize_t_sne(df, texts):
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_tfidf = tsne_model.fit_transform(df)

    # Create a DataFrame for the TSNE results
    tsne_df = pd.DataFrame({
        'Dimension 1': tsne_tfidf[:, 0],
        'Dimension 2': tsne_tfidf[:, 1],
        'Text': texts
    })

    # Create a scatter plot
    fig = px.scatter(tsne_df, x='Dimension 1', y='Dimension 2', hover_data=['Text'])
    fig.update_layout(title='t-SNE visualization of TF-IDF data', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    fig.show()

def show_most_important_words(row_index, df):
    row_index = 0
    non_zero_words = df.loc[row_index][df.loc[row_index] != 0]
    sorted_words = sorted(non_zero_words.items(), key=lambda x: x[1], reverse=True)
    for key, element in sorted_words:
        print(key, element)

def main():
    df, df_tfidfvect = generate_vocabulary_and_word_matrix()
    show_most_important_words(0, df_tfidfvect)
    print(df.head())
    #visualize_t_sne(df_tfidfvect, df['title_text'].tolist())

if __name__ == '__main__':
    main()
    