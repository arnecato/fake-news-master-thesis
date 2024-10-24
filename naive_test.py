import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
import time
dataset = "dataset/ISOT/True_Fake_bert_umap_3dim_3600_6000.h5"
true_training_df = pd.read_hdf(dataset, key='true_training')
true_validation_df = pd.read_hdf(dataset, key='true_validation')
true_test_df = pd.read_hdf(dataset, key='true_test')
true_test_df = pd.concat([true_validation_df, true_test_df])

fake_training_df = pd.read_hdf(dataset, key='fake_training')
fake_validation_df = pd.read_hdf(dataset, key='fake_validation')
fake_test_df = pd.read_hdf(dataset, key='fake_test')
fake_test_df = pd.concat([fake_validation_df, fake_test_df])

distances = []
self_region = 0.25 #0.05613950275754954 # 3d: 0.10658193345223824 # 2d: 0.05613950275754954
false_positive = 0
true_positive = 0
false_negative = 0
true_negative = 0

start_time = time.time()
for row in true_training_df.itertuples(index=False):
    tp = True
    for f_row in fake_test_df.itertuples(index=False):
        distance = np.linalg.norm(row.vector - f_row.vector)
        if distance < self_region:
            false_negative += 1
            tp = False
            break
    if tp:
        true_positive += 1
    
    fp = True
    for t_row in true_test_df.itertuples(index=False):
        distance = np.linalg.norm(row.vector - t_row.vector)
        if distance < self_region:
            true_negative += 1
            fp = False
            break
    if fp:
        false_positive += 1
print(time.time() - start_time, 'samples:', len(true_test_df) + len(fake_test_df))
print('True positive:', true_positive, 'False negative:', false_negative, 'False positive:', false_positive, 'True negative:', true_negative, 'Precision tp/(tp+fp)', true_positive/(true_positive+false_positive), 'Recall tp/(tp+fn):', true_positive/(true_positive+false_negative))
        

# 2D results:
#
# 68.68291282653809 samples: 4800
# True positive: 3571 False negative: 29 False positive: 399 True negative: 3201 Precision tp/(tp+fp) 0.8994962216624686 Recall tp/(tp+fn): 0.9919444444444444

# 58.311758518218994 samples: 4800
# True positive: 3520 False negative: 80 False positive: 79 True negative: 3521 Precision tp/(tp+fp) 0.9780494581828285 Recall tp/(tp+fn): 0.9777777777777777

# 59.24174761772156 samples: 4800
# True positive: 3457 False negative: 143 False positive: 19 True negative: 3581 Precision tp/(tp+fp) 0.9945339470655926 Recall tp/(tp+fn): 0.9602777777777778

# 3D results:
# 92.69389867782593 samples: 4800
# True positive: 3597 False negative: 3 False positive: 2069 True negative: 1531 Precision tp/(tp+fp) 0.6348393928697493 Recall tp/(tp+fn): 0.9991666666666666

# 66.1022834777832 samples: 4800
# True positive: 3563 False negative: 37 False positive: 447 True negative: 3153 Precision tp/(tp+fp) 0.8885286783042394 Recall tp/(tp+fn): 0.9897222222222222

# 57.88345813751221 samples: 4800
# True positive: 3509 False negative: 91 False positive: 146 True negative: 3454 Precision tp/(tp+fp) 0.9600547195622435 Recall tp/(tp+fn): 0.9747222222222223

# 55.831087827682495 samples: 4800
# True positive: 3393 False negative: 207 False positive: 40 True negative: 3560 Precision tp/(tp+fp) 0.9883483833381882 Recall tp/(tp+fn): 0.9425

# NSA 2d, 100 detectors, Precision: 0.9268492031058438 Recall 0.945. 0.86/0.96
# Precision: 0.976558837318331 Recall 0.8679166666666667
# setting radius against detectors too Precision: 0.8823061630218688 Recall 0.9245833333333333
# 130 detectors, pop 50 Precision: 0.9733624454148472 Recall 0.92875
# 160 detectors, pop 50 Precision: 0.9679075738125802 Recall 0.9425
# 220 detectors, pop 50 Precision: 0.9585097375105842 Recall 0.9433333333333334
# 280 detectors, pop 50 Precision: 0.9488352745424293 Recall 0.9504166666666667
# 400 detectors, pop 50 Precision:Precision: 0.9344660194174758 Recall 0.9625
# 430 Precision: 0.9323671497584541 Recall 0.965

# NSA 2D voronoi 16 pop
# 110 detectors, Precision: 0.9792677547419497 Recall 0.925 first example
# 100 Precision: 0.9793001478560868 Recall 0.8279166666666666 2nd example (cutting away the overflowing ones)
# 300 Precision: 0.9492119089316988 Recall 0.9033333333333333
# 648 Precision: 0.911102172164119 Recall 0.94375

# NSA 2D mutation 10 pop
# 408 Precision: 0.947912360479537 Recall 0.9554166666666667
# 608 Precision: 0.9321486268174475 Recall 0.9616666666666667
