import os
import json
import csv
import statistics
# algorithm, dataset, dim, experiment no.
'''results = {
    'ga': {
        'roberta-base': {
            '1': [
                {
                    "precision": 0.9366197183098591,
                    "recall": 0.7657952069716776,
                    "true_detected": 333,
                    "true_total": 6426,
                    "fake_detected": 4921,
                    "fake_total": 6426,
                    "negative_space_coverage": 22.23471745952702,
                    "time_to_build": 746.4579837999918,
                    "detectors_count": 1225,
                    "time_to_infer": 48.32926679999218,
                    "self_region": 0.000512362349768473
                }
            ]
        }
    }
}'''
results = {}


file_identification = 'experiment_result'
path = 'model/detector'
files = [f for f in os.listdir(path) if (file_identification in f and '-1' not in f)]
for f in files:
    if 'nsgaii' in f:
        algo = 'nsgaii'
    else:
        algo = 'ga'
    f_split = f.split('_')

    with open(os.path.join(path, f), 'r') as file:
        data = json.load(file)
        dataset = f_split[1]
        dim = f'{f_split[2][0]}D'
        experiment_no = f_split[3].split('.')[0]

        if algo not in results:
            results[algo] = {}
        if dataset not in results[algo]:
            results[algo][dataset] = {}
        if dim not in results[algo][dataset]:
            results[algo][dataset][dim] = []

        results[algo][dataset][dim].append({
            "precision": data["precision"],
            "recall": data["recall"],
            "true_detected": data["true_detected"],
            "true_total": data["true_total"],
            "fake_detected": data["fake_detected"],
            "fake_total": data["fake_total"],
            "negative_space_coverage": data["negative_space_coverage"],
            "time_to_build": data["time_to_build"],
            "detectors_count": data["detectors_count"],
            "time_to_infer": data["time_to_infer"],
            "self_region": data["self_region"]
        })
        # TODO: remove this when I have more than 1 experiment!
        results[algo][dataset][dim].append({
            "precision": data["precision"],
            "recall": data["recall"],
            "true_detected": data["true_detected"],
            "true_total": data["true_total"],
            "fake_detected": data["fake_detected"],
            "fake_total": data["fake_total"],
            "negative_space_coverage": data["negative_space_coverage"],
            "time_to_build": data["time_to_build"],
            "detectors_count": data["detectors_count"],
            "time_to_infer": data["time_to_infer"],
            "self_region": data["self_region"]
        })
print(results)   

# Prepare CSV file
csv_file = 'results/averaged_results.csv'
csv_columns = [
    'algorithm', 'dataset', 'dimension', 'precision_avg', 'precision_stdev', 'recall_avg', 'recall_stdev',
    'accuracy_avg', 'accuracy_stdev', 'f1_avg', 'f1_stdev', 'true_detected_avg', 'true_detected_stdev', 'true_total_avg', 'true_total_stdev',
    'fake_detected_avg', 'fake_detected_stdev', 'fake_total_avg', 'fake_total_stdev',
    'negative_space_coverage_avg', 'negative_space_coverage_stdev', 'time_to_build_avg', 'time_to_build_stdev',
    'detectors_count_avg', 'detectors_count_stdev', 'time_to_infer_avg', 'time_to_infer_stdev', 'self_region_avg',
    'self_region_stdev'
]

with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    writer.writeheader()

    for algo, datasets in results.items():
        for dataset, dims in datasets.items():
            for dim, experiments in dims.items():
                precision_list = [exp["precision"] for exp in experiments]
                recall_list = [exp["recall"] for exp in experiments]
                true_detected_list = [exp["true_detected"] for exp in experiments]
                true_total_list = [exp["true_total"] for exp in experiments]
                fake_detected_list = [exp["fake_detected"] for exp in experiments]
                fake_total_list = [exp["fake_total"] for exp in experiments]
                negative_space_coverage_list = [exp["negative_space_coverage"] for exp in experiments]
                time_to_build_list = [exp["time_to_build"] for exp in experiments]
                detectors_count_list = [exp["detectors_count"] for exp in experiments]
                time_to_infer_list = [exp["time_to_infer"] for exp in experiments]
                self_region_list = [exp["self_region"] for exp in experiments]

                true_positive_list = [exp["fake_detected"] for exp in experiments]
                false_negative_list = [exp["fake_total"] - exp["fake_detected"] for exp in experiments]
                false_positive_list = [exp["true_detected"] for exp in experiments]
                true_negative_list = [exp["true_total"] - exp["true_detected"] for exp in experiments]

                accuracy_list = [(tp + tn) / (tp + tn + fp + fn) for tp, tn, fp, fn in zip(true_positive_list, true_negative_list, false_positive_list, false_negative_list)]
                f1_list = [2 * (p * r) / (p + r) for p, r in zip(precision_list, recall_list)]

                writer.writerow({
                    'algorithm': algo,
                    'dataset': dataset,
                    'dimension': dim,
                    'precision_avg': statistics.mean(precision_list),
                    'precision_stdev': statistics.stdev(precision_list),
                    'recall_avg': statistics.mean(recall_list),
                    'recall_stdev': statistics.stdev(recall_list),
                    'accuracy_avg': statistics.mean(accuracy_list),
                    'accuracy_stdev': statistics.stdev(accuracy_list),
                    'f1_avg': statistics.mean(f1_list),
                    'f1_stdev': statistics.stdev(f1_list),
                    'true_detected_avg': statistics.mean(true_detected_list),
                    'true_total_avg': statistics.mean(true_total_list),
                    'fake_detected_avg': statistics.mean(fake_detected_list),
                    'fake_total_avg': statistics.mean(fake_total_list),
                    'negative_space_coverage_avg': statistics.mean(negative_space_coverage_list),
                    'time_to_build_avg': statistics.mean(time_to_build_list),
                    'time_to_build_stdev': statistics.stdev(time_to_build_list),
                    'detectors_count_avg': statistics.mean(detectors_count_list),
                    'detectors_count_stdev': statistics.stdev(detectors_count_list),
                    'time_to_infer_avg': statistics.mean(time_to_infer_list),
                    'time_to_infer_stdev': statistics.stdev(time_to_infer_list),
                    'self_region_avg': statistics.mean(self_region_list),
                    'self_region_stdev': statistics.stdev(self_region_list)
                })

# format
{
    "precision": 0.9366197183098591,
    "recall": 0.7657952069716776,
    "true_detected": 333,
    "true_total": 6426,
    "fake_detected": 4921,
    "fake_total": 6426,
    "negative_space_coverage": 22.23471745952702,
    "time_to_build": 746.4579837999918,
    "detectors_count": 1225,
    "precision_list": [
        0.9699416778824586,
        0.9640625,
        0.9626068376068376,
        0.9611017153901908,
        0.9570757880617036,
        0.9542372881355933,
        0.9498767460969597,
        0.9462668544978869,
        0.944136291600634,
        0.9416471506635441,
        0.9394640447272026,
        0.9366197183098591
    ],
    "recall_list": [
        0.33644568938686586,
        0.4800809212573918,
        0.5608465608465608,
        0.6190476190476191,
        0.6661998132586368,
        0.7009025832555245,
        0.7195767195767195,
        0.7317149081854964,
        0.7416744475568005,
        0.7508558979147214,
        0.7583255524431995,
        0.7657952069716776
    ],
    "true_detected_list": [
        67,
        115,
        140,
        161,
        192,
        216,
        244,
        267,
        282,
        299,
        314,
        333
    ],
    "fake_detected_list": [
        2162,
        3085,
        3604,
        3978,
        4281,
        4504,
        4624,
        4702,
        4766,
        4825,
        4873,
        4921
    ],
    "negative_space_coverage_list": [
        17.36491933465004,
        18.70008644508198,
        19.557814925909042,
        20.17026789439842,
        20.671858738991432,
        21.05321857362287,
        21.319802812940907,
        21.54024770911201,
        21.73632528631424,
        21.906399990519276,
        22.07209271829197,
        22.19928125572187
    ],
    "time_to_infer": 48.32926679999218,
    "self_region": 0.000512362349768473
}