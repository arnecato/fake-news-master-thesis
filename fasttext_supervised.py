import pandas as pd
import fasttext
import os

def prepare_training_file(file1, file2, label1, label2, output_file):
    with open(output_file, 'w', encoding='utf-8') as fout:
        for filename, label in [(file1, label1), (file2, label2)]:
            with open(filename, 'r', encoding='utf-8') as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(f"__label__{label} {line}\n")
# model file
model_file = "model/fasttext_supervised_model_300.bin"
if not os.path.exists(model_file):    
    print(f"{model_file} does not exist. Training the model...")  
    # File paths and labels
    file1 = 'dataset/ISOT/True.csv'
    file2 = 'dataset/ISOT/Fake.csv'

    # Load data using pandas
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Save the data to temporary text files
    df1.to_csv('dataset/ISOT/True_fasttext.txt', index=False, header=False)
    df2.to_csv('dataset/ISOT/Fake_fasttext.txt', index=False, header=False)

    # Update file paths to point to the new text files
    file1 = 'dataset/ISOT/True_fasttext.txt'
    file2 = 'dataset/ISOT/Fake_fasttext.txt'

    training_file = 'dataset/ISOT/fasttext_training.txt'
    print(f"Preparing training file: {training_file}")
    prepare_training_file(file1, file2, 'class1', 'class2', training_file)
    print("Starting training...")
    # Train the supervised fastText model
    model = fasttext.train_supervised(
        input=training_file,
        epoch=25,
        lr=0.1,
        wordNgrams=2,
        verbose=2,
        minCount=1,
        dim=300
    )
    model.save_model(model_file)
else:
    print(f"{model_file} already exists. Skipping training.")
    hdf = "dataset/ISOT/Fake_fasttext_supervised.h5"
    df = pd.read_hdf(hdf)
    pd.set_option('display.max_colwidth', None)
    print(df['vector'].head(10))
