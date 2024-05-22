#########################################################
#    Imports    #
#########################################################
import json

import tensorflow as tf

# List all physical devices and configure them before any other operations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth on the GPU to true
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print("Memory growth set")
            print("GPU Device:", gpu, "\n")
    except RuntimeError as e:
        # Memory growth must be set before initializing the GPUs
        print("RuntimeError in setting up GPU:", e)

    try:
        # Optional: Set a memory limit
        memory_limit = 8000  # e.g., 4096 MB for 4GB
        config = tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [config])
        print(f"Memory limit set to {memory_limit}MB on GPU {gpus[0].name}")
    except RuntimeError as e:
        print(f"Failed to set memory limit: {e}")
else:
    print("No GPU devices found.")

import numpy as np
import pandas as pd
import math
import glob
import random
from tqdm import tqdm
#from IPython.display import clear_output
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
from ydata_profiling import ProfileReport

import sklearn.cluster as cluster
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, precision_recall_fscore_support

# Print versions and device configurations after ensuring GPU settings
print("TensorFlow version:", tf.__version__)
# print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
# print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
# print(tf.config.list_physical_devices(), "\n", tf.config.list_logical_devices(), "\n")
# print(tf.config.list_physical_devices('GPU'), "\n")

#########################################################
#    Loading GAN Model and Generating Data              #
#########################################################


########## inputs #################
# gan variants
synth_categories = ['CWGANGP', 'CGAN', 'CTGAN', 'WGANGP']
model = "CTGAN"  # input

evalautor_types = ['XGBoost', 'LogisticRegression', 'Perceptron', 'AdaBoost', 'RandomForest', 'DeepNeuralNetwork', 'KNearestNeighbor']

label_classes = ['33+1', '7+1', '1+1']
labelClass = "33+1"  # input

if model == "CWGANGP":
    if labelClass == '33+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cwgangp_model_2_specific.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCWS_20240515185435.pkl')  # input
    if labelClass == '7+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cwgangp_model_2_general.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCWG_20240515183824.pkl')  # input
    if labelClass == '1+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cwgangp_model_2_binary1.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCWB_20240515182814.pkl')  # input

if model == "CGAN":
    if labelClass == '33+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cgan_model_1_specific.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCS_20240515224149.pkl')  # input
    if labelClass == '7+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cgan_model_1_general.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCG_20240516014417.pkl')  # input
    if labelClass == '1+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_cgan_model_1_binary.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCWB_20240515182814.pkl')  # input

if model == "CTGAN":
    if labelClass == '33+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_ctgan_model_3_specific2.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCTS_20240521194349.pkl')  # input
    if labelClass == '7+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_ctgan_model_3_general.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScaler_ctGANGeneral.pkl')  # input
    if labelClass == '1+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/cyberattack_ctgan_model_3_binary.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerCTB_20240517000909.pkl')  # input

if model == "WGANGP":
    if labelClass == '33+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/attack_wgangp_model_specific.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerWS_20240516120126.pkl')  # input
    if labelClass == '7+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/attack_wgangp_model_general.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerWG_20240516120817.pkl')  # input
    if labelClass == '1+1':
        print(model, " ", labelClass)
        # loading GAN
        synth = RegularSynthesizer.load('../GAN_models/attack_wgangp_model_binary.pkl')  # input
        # loading Scaler
        scaler = joblib.load('../scalar_models/MinMaxScalerWB_20240516134008.pkl')  # input


# specifying the samples per class
samples_per_synth_class = 10000  # input

real_test_data_sample_size = 10000  # input

#########################


# dictionary to decode label
if labelClass == "33+1":
    print("Synthesizing synthetically generated Specific labels...")
    synth_label_mapping = {0: 'Backdoor_Malware', 1: 'BenignTraffic', 2: 'BrowserHijacking', 3: 'CommandInjection',
                           4: 'DDoS-ACK_Fragmentation', 5: 'DDoS-HTTP_Flood', 6: 'DDoS-ICMP_Flood',
                           7: 'DDoS-ICMP_Fragmentation', 8: 'DDoS-PSHACK_Flood', 9: 'DDoS-RSTFINFlood',
                           10: 'DDoS-SYN_Flood', 11: 'DDoS-SlowLoris', 12: 'DDoS-SynonymousIP_Flood', 13: 'DDoS-TCP_Flood',
                           14: 'DDoS-UDP_Flood', 15: 'DDoS-UDP_Fragmentation', 16: 'DNS_Spoofing',
                           17: 'DictionaryBruteForce', 18: 'DoS-HTTP_Flood', 19: 'DoS-SYN_Flood', 20: 'DoS-TCP_Flood',
                           21: 'DoS-UDP_Flood', 22: 'MITM-ArpSpoofing', 23: 'Mirai-greeth_flood', 24: 'Mirai-greip_flood',
                           25: 'Mirai-udpplain', 26: 'Recon-HostDiscovery', 27: 'Recon-OSScan', 28: 'Recon-PingSweep',
                           29: 'Recon-PortScan', 30: 'SqlInjection', 31: 'Uploading_Attack', 32: 'VulnerabilityScan',
                           33: 'XSS'}
if labelClass == "7+1":
    print("Synthesizing synthetically generated General labels...")
    synth_label_mapping = {0: 'DDOS', 1: 'DOS', 2: 'Mirai', 3: 'Recon',
                            4: 'Spoofing', 5: 'Benign', 6: 'Web', 7: 'BruteForce'}
if labelClass == "1+1":
    print("Synthesizing synthetically generated Binary labels...")
    synth_label_mapping = {0: 'Attack', 1: 'Benign'}

print("Synth labels mapping:", synth_label_mapping)

# Create inverse mapping from string labels to integers
inverse_synth_label_mapping = {v: k for k, v in synth_label_mapping.items()}

# Create an array that contains the class code repeated for the number of samples per class
# conditions = []
# for code in synth_label_mapping.values():
#     conditions.extend([code] * samples_per_class)
conditions = []
for label in synth_label_mapping.values():
    conditions.extend([inverse_synth_label_mapping[label]] * samples_per_synth_class)

# Optionally shuffle the conditions to randomize the order
# np.random.shuffle(conditions)

# Create a DataFrame for these conditions
cond_array = pd.DataFrame(conditions, columns=['label'])

# Start the training timer
start_time_gen = time.time()
print("Start Generating...\n")

# Generating synthetic samples
if model == "WGANGP" or model == "CTGAN":
    synth_data = synth.sample(samples_per_synth_class)  # for non cgans
else:
    synth_data = synth.sample(cond_array)  # # This uses the condition array


# End the training timer
generation_time = time.time() - start_time_gen
print("Finished Generating...\n")

# print data to review
print(synth_data.head(), "\n")

# Apply mapping to decode
# synth_train_data['label'] = synth_train_data['label'].map(synth_label_mapping)
# print(synth_train_data)

#########################################################
#    Analyzing the Synthetic Data   #
#########################################################

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = synth_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

class_counts = synth_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(synth_data.head())

#########################################################
#    Loading Real Data                                  #
#########################################################
DATASET_DIRECTORY = '../../archive/'

# List the files in the dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
print(csv_filepaths)

# If there are more than X CSV files, randomly select X files from the list
sample_size = 1
if len(csv_filepaths) > sample_size:
    csv_filepaths = random.sample(csv_filepaths, sample_size)
    print(csv_filepaths)
csv_filepaths.sort()

# l\ist of csv files used for training data sets
training_data_sets = csv_filepaths

# Mapping Features
num_cols = [
    'flow_duration', 'Header_Length',  'Duration',
    'Rate', 'Srate', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
    ]

# Mapping Labels
cat_cols = [
    'Protocol Type', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC',
    ]

dict_7classes = {'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
                 'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
                 'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
                 'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
                 'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
                 'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
                 'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
                 'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing',
                 'MITM-ArpSpoofing': 'Spoofing', 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web',
                 'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web',
                 'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'
                 }

dict_2classes = {'DDoS-RSTFINFlood': 'Attack', 'DDoS-PSHACK_Flood': 'Attack', 'DDoS-SYN_Flood': 'Attack',
                 'DDoS-UDP_Flood': 'Attack', 'DDoS-TCP_Flood': 'Attack', 'DDoS-ICMP_Flood': 'Attack',
                 'DDoS-SynonymousIP_Flood': 'Attack', 'DDoS-ACK_Fragmentation': 'Attack',
                 'DDoS-UDP_Fragmentation': 'Attack', 'DDoS-ICMP_Fragmentation': 'Attack', 'DDoS-SlowLoris': 'Attack',
                 'DDoS-HTTP_Flood': 'Attack', 'DoS-UDP_Flood': 'Attack', 'DoS-SYN_Flood': 'Attack',
                 'DoS-TCP_Flood': 'Attack', 'DoS-HTTP_Flood': 'Attack', 'Mirai-greeth_flood': 'Attack',
                 'Mirai-greip_flood': 'Attack', 'Mirai-udpplain': 'Attack', 'Recon-PingSweep': 'Attack',
                 'Recon-OSScan': 'Attack', 'Recon-PortScan': 'Attack', 'VulnerabilityScan': 'Attack',
                 'Recon-HostDiscovery': 'Attack', 'DNS_Spoofing': 'Attack', 'MITM-ArpSpoofing': 'Attack',
                 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Attack', 'Backdoor_Malware': 'Attack', 'XSS': 'Attack',
                 'Uploading_Attack': 'Attack', 'SqlInjection': 'Attack', 'CommandInjection': 'Attack',
                 'DictionaryBruteForce': 'Attack'
                 }

# Extracting data from csv to input into data frame
real_data = pd.DataFrame()
for data_set in training_data_sets:
    print(f"data set {data_set} out of {len(training_data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, data_set)
    df = pd.read_csv(data_path)
    real_data = pd.concat([real_data, df])

# Relabel the 'label' column using dict_7classes
if labelClass == "7+1":
    print("General Classes")
    real_data['label'] = real_data['label'].map(dict_7classes)

if labelClass == "1+1":
    print("Binary Classes")
    # Relabel the 'label' column using dict_2classes
    real_data['label'] = real_data['label'].map(dict_2classes)

# Shuffle data
real_data = shuffle(real_data, random_state=1)

#########################################################
#    Analyzing the Synthetic Data                       #
#########################################################
# x and y split
X = {}
y = {}

for _class in label_classes:
    X[_class] = synth_data[_class].drop('label', axis=1)
    y[_class] = synth_data[_class]['label']

print(f'X: {X[label_classes[0]].shape}, y: {y[label_classes[0]].shape}')

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels_synth = synth_data['label'].unique()

# Print the number of unique labels
# print(f"There are {unique_labels_synth} unique labels in the Synthetic dataset.")
num_unique_labels_synth = len(unique_labels_synth)
print(f"There are {num_unique_labels_synth} unique labels in the Synthetic dataset.")

# print the amount of instances for each label
class_counts_synth = synth_data['label'].value_counts()
print(class_counts_synth)

# Display the first few entries to verify the changes
print("Synthetic dataset:")
print(synth_data.head())

all_data_generated = {}
for _class in label_classes:
    all_data_generated[_class] = pd.concat([X[_class], y[_class]], axis=1)

# prints an instance of each class
print("Each Instance in the Synthetic Training Dataset:")
for label in unique_labels_synth:
    instances = synth_data[synth_data['label'] == label]
    if not instances.empty:
        print(f"First instance of {label}:")
        print(instances.iloc[0])
    else:
        print(f"No instances found for label {label}")



#########################################################
#         Saving Metrics and Results                     #
#########################################################

# Directory to save classification report text files
report_dir = "./synth_data_reports"
os.makedirs(report_dir, exist_ok=True)

# If there's no sampled_dataset_metrics.json, make a new one and store the unsampled dataset metrics
try:
    df_label_counts = pd.read_json(path_or_buf=report_dir + '/sampling_label_counts.json', orient='index')

except FileNotFoundError:
    # schema:   Synth | Label Classes | 0 | 1 | 2 | 3 | ... | 31 | 32 | 33
    df_label_counts = pd.DataFrame(columns=['Synth', 'Label Classes'] + [str(i) for i in range(34)])


def save_results():
    # Get the current timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")

    # Update dataframe
    for label_class in label_classes:

        row_index = df_label_counts.index[
            (df_label_counts['Synth'] == model) & (df_label_counts['Label Classes'] == label_class)]
        row_index = row_index.tolist()

        value_counts = all_data_generated[label_class]['label'].value_counts()
        value_counts.sort_index(inplace=True)

        match len(row_index):
            case 0:  # No previous record
                row_index = len(df_label_counts.index)
                df_label_counts.loc[row_index, 'Synth'] = model
                df_label_counts.loc[row_index, 'Label Classes'] = label_class

                for i in range(len(value_counts)):
                    df_label_counts.loc[row_index, str(i)] = value_counts[i]

            case 1:  # Update previous record
                for i in range(len(value_counts)):
                    df_label_counts.loc[row_index, str(i)] = value_counts[i]

            case _:
                assert False, f'ERROR: {model} / {label_class} is duplicated. This should NOT happen.'

    # Update file
    df_label_counts.to_json(path_or_buf=report_dir + '/generated_label_counts.json', orient='index')
    print("GAN reports saved successfully.")


# Call save_results with the evaluator_metrics list
save_results()


# Save the synthetic data to a CSV file
synth_data.to_csv('./results/synthetic_TEST_data.csv', index=False)

#########################################################
#         Making Graphs, Documents, and Diagrams        #
#########################################################


def plot_class_distribution(data, title):
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='label', data=data)
    plt.title(title)
    plt.ylabel('Count')
    plt.xlabel('Class Label')
    # Rotate labels to prevent overlap
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust rotation and font size as needed
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()


def plot_feature_comparison(real_data, synth_data, feature1, feature2):
    plt.figure(figsize=(12, 6))

    # Plotting the real data
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=feature1, y=feature2, data=real_data, alpha=0.5)
    plt.title('Real Data')
    plt.xlabel(feature1)  # Ensure the feature names are readable
    plt.ylabel(feature2)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust rotation and font size as needed

    # Plotting the synthetic data
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=feature1, y=feature2, data=synth_data, alpha=0.5)
    plt.title('Synthetic Data')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust rotation and font size as needed

    plt.suptitle(f'Comparison of {feature1} vs {feature2}')
    plt.tight_layout()  # Adjust layout to make room for label rotation
    plt.show()


print(f"Generation time for Balanced Synthetic Dataset: {generation_time:.20f} seconds")

# Plot class distribution for both real and synthetic data
plot_class_distribution(real_data, 'Real Data Class Distribution')
plot_class_distribution(synth_data, 'Synthetic Data Class Distribution')

# Plot feature comparisons (adjust 'feature1' and 'feature2' to your dataset's features)
plot_feature_comparison(real_data, synth_data, 'flow_duration', 'Duration')

# Provide a Report of each feature and other stats from Ydata profiling
original_report = ProfileReport(real_data, title='Original Data', minimal=True)
generated_report = ProfileReport(synth_data, title='Generated Data', minimal=True)
comparison_report = original_report.compare(generated_report)
comparison_report.to_file(f'./profile_reports/{model}_TEST_original_vs_synth.html')

