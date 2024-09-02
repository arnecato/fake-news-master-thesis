# Plan
- Datasets
    - batch-loading of dataset
    - sparse tf-idf objects (index[],)
    - DVC
    - versioning and storing of datasets
    - ELT TF-IDF
    - automation loading and converting
    - automation running experiments, measuring and plotting results
- Implementation baseline
    - Plotting of results
- Implementation MO NSA wo/feature selection
- Implementation MO NSA w/embedded feature selection + VLC
- Results

# Notes
Word vectors - retrieved from pretrained model (word2vec or FastText)
Sentence vectors - average of word vectors in sentence
Document vectors - average of sentence vectors in document

Use spacy to tokenize documents. Retrieve word, sentence and document vectors.

OK Check how spacy tokenizes words, sentences and documents. Does it keep the numbers or nouns?
OK Proper document vector. Not average of all words. Average of all sentence vectors instead.

Spacy is slow when using sentence segmentation. Could I just split on periods?
Investigate Sentencizer

Next is to create vector representations of documents in datasets.

# Baseline single-objective NSA
Similar to work done in (Ripon et al., n.d.) by Ripon et al.

Objective 1: Maximizing RR
Setting RR next to nearest self (incl other detectors)
5-fold training and test set 75%/25% and measure:

1. Avg. precision
2. Avg. recall

# Multi-objective NSA without feature selection
Objective 1: Maximizing RR

Objective 2: Minimizing overlap with self (incl. other detectors)

Full feature vectors used.

5-fold training and test set 75%/25% and measure:
1. Avg. precision
2. Avg. recall

# Multi-objective NSA with embedded feature selection and VLC
Same as above, except with embedded feature selection and variable-length chromosome method.
Detector encoding has variable length. Selects which subset of features to use in the detector, e.g. [49,355,1433,254]. Mutation and recombination operators will add and subtract feature subsets.

5-fold training and test set 75%/25% and measure:
1. Avg. precision
2. Avg. recall

# Algorithm
1. Initialize new antibody population. 0.5 of them randomly, 0.5 of them based on an antigen position. Set RR to border of closest self.
2. Evolve antibodies with feature values and RR:
    - Maximize RR
    - Minimize overlap between Antibody ab and antigen ag and ab and mature antibody mab (penalizing self sample overlap more than mature detector):
$$𝑂=(1+ΣOverlapab, ag)^2+(1+ΣOverlapab, mab)−2$$

3. MO convergence based on diversity or spacing TBD.
4. Evaluate F1-score of mature detector set on training dataset. Let precision and recall guide weighting between objectives for selecting new detector from pareto-front of MO run. E.g if lower recall than precision, emphasize picking a new detector by weighting objective 1 (RR and coverage) higher. Use softmax to get weight from precision and recall metrics. If no mature detector yet, then weight 0.5/0.5 between objectives.
5. New detector must have no overlap with self samples.
6. Find F1-score of mature detector set with new detector
7. on validation set:
    - If F1 has not converged yet: Go back to 1 and re-initialize population to evolve new detector
    - If converged: record best F1 score achieved on validation set


