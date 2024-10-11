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

OK - Check how spacy tokenizes words, sentences and documents. Does it keep the numbers or nouns?
OK - Proper document vector. Not average of all words. Average of all sentence vectors instead.

Spacy is slow when using sentence segmentation. Could I just split on periods?
Investigate Sentencizer

OK - Next is to create vector representations of documents in datasets and save it for easy retrieval.

OK - Next is to create detectors based on offspring vectors, then mutate, then compare, sort and refill population
OK - then check detectors against self before adding to mature detector set
OK - split training, validation, test set
OK - train one detector on training set before checking on validation
OK define save object for a detector
OK save mature detectors
OK load mature detectors
OK test detector set on fake news
f1 check against validation set to check for stopping of adding detectors

Precision = TP (fake news) / (TP + FP)
Recall = TP (fake news) / (TP + FN)

11.09.24
Fake sample inititalization did not work
bert vectors did not work
only self sample and only detector checks did not work
changing mutation rates etc did not work
PCA reduction 10,50,100,150,200 did not work

distance * 1.15 = Precision: 0.9513274336283186 Recall 0.13735093696763204
distance * 1.75 = Precision: 0.9188405797101449 Recall 0.20251277683134583
distanace * 1.75 (2nd detector) Precision: 0.5245164192532613 Recall 0.9931856899488927

Hypothesis: document representation and NSA not a good match. OR, bug in code.
Observation: detector vector has extreme values

Options ahead:
- different document representation (but bert did not work either)
- VLC
- TF-IDF
- Completely different representation. Word importance, or word existence? 
- Tolerance level, taking the nth closest instead of the closest as max distance9


word importance in document?
- count words 

Initialize detectors on/around self, at least a portion of them
Try to control the initialization so it covers the search space

Looks like high dimensions was an issue.
PCA helped.
Also, generation of detectors in 3d is stuck to two orthogonal planes. Inefficient distribution throughout the problem space.

IDEA:
Since MO cannot be based on checking non-self:
Measure total negative space covered and overlap between detectors

Or, be close to selves and cover most negative space (distance)
So: in the beginning it will prioritize large coverage detectors far from selves, and towards end it will prioritize detectors closer to selves

What about prioritizing checking against detectors first? First maximum distance from other detector. Then check against self. Can choose orthogonal distance from existing detector?

15.09.24:
Dim 4: 0.01 self region
    Detectors: 100
    0.9328575999825262
    Precision: 1.0 Recall 0.297
    True/Real detected: 0 Total real/true: 1000 Fake detected: 297 Total fake: 1000

    Detectors: 200
    1.366883099952247
    Precision: 1.0 Recall 0.337
    True/Real detected: 0 Total real/true: 1000 Fake detected: 337 Total fake: 1000

    Detectors: 1500
    2.3514501999597996
    Precision: 1.0 Recall 0.475
    True/Real detected: 0 Total real/true: 200 Fake detected: 95 Total fake: 200
Dim 4:
    detectors_test_4dim_2.json
    initiating detector far from existing detectors
    Detectors: 1000
    1.597175500006415
    Precision: 1.0 Recall 0.45
    Validation! True/Real detected: 0 Total real/true: 200 Fake detected: 90 Total fake: 200

Dim 3:
    1108
    Detectors: 1108
    1.7588314999593422
    Precision: 1.0 Recall 0.48
    Validation True/Real detected: 0 Total real/true: 200 Fake detected: 96 Total fake: 200

16.09.24:
Observing that for 3 dim and large number of detectors (1110) precision is 100% and recall is 48%. Incl initiating new detectors far from existing. 
Really slow at this point. More detectors than true training samples (600). Increasing detector radius with minimal distance only drops precision dramatically and negligble improvement in recall.
Euclidean distance.

What about higher dimensions? Will precision stay and recall improve? Will it take longer to fill the negative space?
Dim 10 did not seem to work: 50 detectors 0 recall. 117 for 0.2% recall.
Detectors: 120
0.7631791000021622
Precision: 1.0 Recall 0.002
True/Real detected: 0 Total real/true: 1000 Fake detected: 2 Total fake: 1000

Dim 2:
    500
    Detectors: 500
    3.300890700018499
    Precision: 1.0 Recall 0.56
    True/Real detected: 0 Total real/true: 1000 Fake detected: 560 Total fake: 1000
    Detectors: 610
    610
    Detectors: 610
    0.9198585000121966
    Precision: 1.0 Recall 0.615
    True/Real detected: 0 Total real/true: 200 Fake detected: 123 Total fake: 200
    Detectors: 1100
    1100
    Detectors: 1100
    6.851127800007816
    Precision: 1.0 Recall 0.729
    True/Real detected: 0 Total real/true: 1000 Fake detected: 729 Total fake: 1000
    Detectors: 1110
    1.5798142999992706
    Precision: 1.0 Recall 0.75
    Validation!: True/Real detected: 0 Total real/true: 200 Fake detected: 150 Total fake: 200
    Detectors: 1400
    9.487312500015832
    Precision: 1.0 Recall 0.754
    True/Real detected: 0 Total real/true: 1000 Fake detected: 754 Total fake: 1000
    Detectors: 1610
    2.510504400008358
    Precision: 1.0 Recall 0.79
    Validation: True/Real detected: 0 Total real/true: 200 Fake detected: 158 Total fake: 200

2 dim BERT: Can get 100% precision and 75% recall on validation set (200). Similar to training (event bit better).
What about making a self region around self and maximize coverage of self while minimizing amount of detectors. Chromosome consist of detectors

Dim 3 BERT:
    Detectors: 1221
    2.27325110003585
    Precision: 1.0 Recall 0.52
    Validation: True/Real detected: 0 Total real/true: 200 Fake detected: 104 Total fake: 200
    Detectors: 2221
    3.369496800005436
    Precision: 1.0 Recall 0.58
    True/Real detected: 0 Total real/true: 200 Fake detected: 116 Total fake: 200
    OPS! From 1221 to 2221 was done without initializing detectors furthers away from existing detectors.

    No overlap
    Detectors: 2465
    7.625135299982503
    Precision: 1.0 Recall 0.51
    True/Real detected: 0 Total real/true: 400 Fake detected: 204 Total fake: 400

Dim 2 BERT UMAP:
    Detectors: 1000
    1.500557999999728
    Precision: 1.0 Recall 0.735
    Validation: True/Real detected: 0 Total real/true: 200 Fake detected: 147 Total fake: 200

    Allowing some overlap between detectors
    Detectors: 621
    0.943705299985595
    Precision: 1.0 Recall 0.635
    Validation! True/Real detected: 0 Total real/true: 200 Fake detected: 127 Total fake: 200

    Allowing some overlap
    Detectors: 2299
    5.577829800080508
    Precision: 1.0 Recall 0.925
    True/Real detected: 0 Total real/true: 400 Fake detected: 370 Total fake: 400

    No overlap
    Detectors: 3063
    7.746635599993169
    Precision: 1.0 Recall 0.8625
    True/Real detected: 0 Total real/true: 400 Fake detected: 345 Total fake: 400

18.09.24:
Realized that model was trained on full true set. Changed to true training set.

2dim, some overlap, 0.0001 self region rate (of avgrandom vector distance form self):
    Detectors: 1127
    1.3616040999768302
    Precision: 0.5672082717872969 Recall 0.96
    True/Real detected: 293 Total real/true: 400 Fake detected: 384 Total fake: 400

    Self region: 0.06852963038506088
    Detectors: 1709
    3.5012864000163972
    Precision: 0.6585903083700441 Recall 0.7475
    True/Real detected: 155 Total real/true: 400 Fake detected: 299 Total fake: 400

2dim, some overlap, 0.02 self region rate
    Detectors: 100
    0.2388986999867484
    Precision: 0.8702290076335878 Recall 0.855
    True/Real detected: 51 Total real/true: 400 Fake detected: 342 Total fake: 400

    700 neighbours 0.1 min_dist
    Detectors: 100
    0.14662949997000396
    Precision: 0.9635854341736695 Recall 0.86
    True/Real detected: 13 Total real/true: 400 Fake detected: 344 Total fake: 400

    Self region: 0.11915704995836396
    Detectors: 200
    0.4207805000478402
    Precision: 0.9363867684478372 Recall 0.92
    True/Real detected: 25 Total real/true: 400 Fake detected: 368 Total fake: 400

2dim, some overlap allowed, uniform initialization of detectors, 
    Detectors: 50
    0.08149080001749098
    Precision: 0.9973753280839895 Recall 0.95
    True/Real detected: 1 Total real/true: 400 Fake detected: 380 Total fake: 400

    Detectors: 594
    0.3333925000624731
    Precision: 0.9702233250620348 Recall 0.9775
    True/Real detected: 12 Total real/true: 400 Fake detected: 391 Total fake: 400

    area - overlap single optimization
    Detectors: 40
    0.09705620002932847
    Precision: 0.9949367088607595 Recall 0.9825
    True/Real detected: 2 Total real/true: 400 Fake detected: 393 Total fake: 400

    Detectors: 100
    0.22057200002018362
    Precision: 0.9876237623762376 Recall 0.9975
    True/Real detected: 5 Total real/true: 400 Fake detected: 399 Total fake: 400
    

21.090.24:
IDEA: measure density per feature value, most dense prioritized first, then set min/max caps and cover negative space outside. E.g  -1 < x < 1 in all true cases. Negative space captures everything above 1+selfregion , and everything below -1 - selfregion.
    divide up chromosome, try dimensional reduction on sections, then apply nsa on each


23.09.24:
- implement DVC, automate dataset 
- implement M.O NSA

07.10.24:
Embedded Feature Selection
Reducing dimensions
Filtering true data. Splitting them around the cluster.

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


