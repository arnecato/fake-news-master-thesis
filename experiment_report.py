import os
file_identification = 'experiment_result'
path = 'model/ISOT'
files = [f for f in os.listdir(path) if file_identification in f]
print(files)