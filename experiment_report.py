import os
import json
import csv
import statistics
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
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
            "precision": data["test_precision"],
            "recall": data["test_recall"],
            "true_detected": data["test_true_detected"],
            "true_total": data["test_true_total"],
            "fake_detected": data["test_fake_detected"],
            "fake_total": data["test_fake_total"],
            "negative_space_coverage": data["test_negative_space_coverage"],
            "time_to_build": data["test_time_to_build"],
            "detectors_count": data["test_detectors_count"],
            "time_to_infer": data["test_time_to_infer"],
            "self_region": data["self_region"],
            "stagnation": data["stagnation"]
        })
        # TODO: remove this when I have more than 1 experiment!
        results[algo][dataset][dim].append({
            "precision": data["test_precision"],
            "recall": data["test_recall"],
            "true_detected": data["test_true_detected"],
            "true_total": data["test_true_total"],
            "fake_detected": data["test_fake_detected"],
            "fake_total": data["test_fake_total"],
            "negative_space_coverage": data["test_negative_space_coverage"],
            "time_to_build": data["test_time_to_build"],
            "detectors_count": data["test_detectors_count"],
            "time_to_infer": data["test_time_to_infer"],
            "self_region": data["self_region"],
            "stagnation": data["stagnation"]
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
    'self_region_stdev', 'stagnation'
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
                    'self_region_stdev': statistics.stdev(self_region_list),
                    'stagnation': statistics.mean([exp["stagnation"] for exp in experiments])
                })

                

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Drop all columns that contain 'stdev'
df = df.loc[:, ~df.columns.str.contains('stdev')]

# Save the modified DataFrame to a new CSV file
excel_file = 'results/averaged_results_nostdev.xlsx'

# Round all numbers to max 3 decimals
df = df.round(3)
df.to_excel(excel_file, index=False)

# Display the DataFrame as a table
print(df.to_string(index=False))

# format to read frp, experiment files
{
    "test_precision": 0.8910860012554928,
    "test_recall": 0.6626984126984127,
    "test_f1": 0.7601070950468541,
    "test_true_detected": 347,
    "test_true_total": 4284,
    "test_fake_detected": 2839,
    "test_fake_total": 4284,
    "test_negative_space_coverage": 270.56578311376575,
    "test_time_to_build": 1565.372218000004,
    "test_detectors_count": 1475,
    "test_time_to_infer": 41.928929599991534,
    "validation_precision_list": [
        0.9918962722852512,
        0.9725050916496945,
        0.9648814749780509,
        0.9608626198083067,
        0.9585571757482733,
        0.9541984732824428,
        0.9518613607188704,
        0.9522946859903382,
        0.9486594409583571,
        0.9456404736275565,
        0.9422476586888657,
        0.9402390438247012,
        0.939143135345667,
        0.9391345696623871,
        0.9377901578458682,
        0.936986301369863,
        0.9363028953229399,
        0.9319877139096094,
        0.9297110823630875,
        0.9277518062048449,
        0.9250104733975701,
        0.9237147595356551,
        0.9212146081247435,
        0.9193154034229829,
        0.9160951996772893,
        0.9145231499802137,
        0.9136522753792299,
        0.9118436182445382,
        0.9115749525616699,
        0.9104252916823485,
        0.9089891831406192,
        0.9070453707119144,
        0.906318082788671,
        0.90465872156013,
        0.90379113018598,
        0.9035802906770649,
        0.9026798307475318,
        0.8998599439775911,
        0.899025069637883,
        0.8975778546712803,
        0.897038567493113,
        0.8959260527216707,
        0.8935516888433982,
        0.8930027173913043,
        0.8921568627450981,
        0.891545975075783,
        0.8894472361809045,
        0.8894789246598075,
        0.8883751651254954,
        0.8869908015768725,
        0.8860510805500982,
        0.8861709067188519,
        0.8859649122807017,
        0.8858713223407695,
        0.8849871134020618,
        0.8854667949951877,
        0.88512,
        0.8848852040816326,
        0.8845419847328244
    ],
    "validation_recall_list": [
        0.14289049731496614,
        0.22297455054868084,
        0.2565958440345552,
        0.2808778893299089,
        0.2916180247490077,
        0.3210366565491478,
        0.3462526266635536,
        0.3681998599112771,
        0.38827924352089654,
        0.4102264767686201,
        0.422834461825823,
        0.4408125145925753,
        0.45038524398785895,
        0.46112537940695775,
        0.47163203362129347,
        0.47910343217371004,
        0.49077749241186086,
        0.4959140789166472,
        0.5033854774690637,
        0.5096894699976652,
        0.5155265001167406,
        0.5201961242120009,
        0.5241653046929722,
        0.5267335979453655,
        0.5302358160168107,
        0.5395750642073313,
        0.5484473499883259,
        0.5554517861312165,
        0.5608218538407658,
        0.564791034321737,
        0.5689936960074714,
        0.5741302825122577,
        0.5827690870884894,
        0.5848704179313565,
        0.5900070044361428,
        0.5951435909409293,
        0.5977118841933224,
        0.6000466962409526,
        0.6028484706981088,
        0.605650245155265,
        0.6082185384076582,
        0.6110203128648144,
        0.6114872752743404,
        0.6138220873219706,
        0.6161568993696007,
        0.6180247490077049,
        0.619892598645809,
        0.6257296287648845,
        0.6280644408125146,
        0.6303992528601448,
        0.6318001400887229,
        0.634368433341116,
        0.6367032453887462,
        0.6397385010506654,
        0.6413728694840065,
        0.6444081251459257,
        0.6458090123745038,
        0.647910343217371,
        0.6493112304459491
    ],
    "validation_true_detected_list": [
        5,
        27,
        40,
        49,
        54,
        66,
        75,
        79,
        90,
        101,
        111,
        120,
        125,
        128,
        134,
        138,
        143,
        155,
        163,
        170,
        179,
        184,
        192,
        198,
        208,
        216,
        222,
        230,
        233,
        238,
        244,
        252,
        258,
        264,
        269,
        272,
        276,
        286,
        290,
        296,
        299,
        304,
        312,
        315,
        319,
        322,
        330,
        333,
        338,
        344,
        348,
        349,
        351,
        353,
        357,
        357,
        359,
        361,
        363
    ],
    "validation_fake_detected_list": [
        612,
        955,
        1099,
        1203,
        1249,
        1375,
        1483,
        1577,
        1663,
        1757,
        1811,
        1888,
        1929,
        1975,
        2020,
        2052,
        2102,
        2124,
        2156,
        2183,
        2208,
        2228,
        2245,
        2256,
        2271,
        2311,
        2349,
        2379,
        2402,
        2419,
        2437,
        2459,
        2496,
        2505,
        2527,
        2549,
        2560,
        2570,
        2582,
        2594,
        2605,
        2617,
        2619,
        2629,
        2639,
        2647,
        2655,
        2680,
        2690,
        2700,
        2706,
        2717,
        2727,
        2740,
        2747,
        2760,
        2766,
        2775,
        2781
    ],
    "validation_negative_space_coverage_list": [
        259.48605386565526,
        261.29029620974,
        262.24610246800916,
        262.9723768039717,
        263.4899222962909,
        263.93205061895844,
        264.3892279970134,
        264.76774944505866,
        265.1232306254379,
        265.4222467193124,
        265.7031065066221,
        265.94344325732146,
        266.1812622296741,
        266.4052230876976,
        266.592270818914,
        266.7775119520199,
        266.95840297376105,
        267.13337532655123,
        267.29829186610004,
        267.4433786244009,
        267.5973643638148,
        267.74360000344757,
        267.8720574357275,
        267.9983792332089,
        268.1048935351691,
        268.2230676839261,
        268.3506488553551,
        268.4523089961173,
        268.5504099696273,
        268.64020197779337,
        268.7246146039649,
        268.80976623508417,
        268.9063922025935,
        268.9932635621042,
        269.0895828002238,
        269.18382919385823,
        269.2547932011949,
        269.3230107536787,
        269.402200337639,
        269.4795069043911,
        269.55063731602024,
        269.62130819141436,
        269.690128806862,
        269.7516169922243,
        269.8126480170372,
        269.8688231408898,
        269.92621821089693,
        269.9865038354926,
        270.04558574274563,
        270.09845049729597,
        270.15235540368883,
        270.20615218640233,
        270.2601108999397,
        270.31722705966865,
        270.36561634312244,
        270.41191024741966,
        270.46931921414125,
        270.5170165444157,
        270.56578311376575
    ],
    "self_region": 0.019760372698923515,
    "stagnation": 0
}