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

real_test_data_sample_size = 2000  # input

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
    synth_train_data = synth.sample(samples_per_synth_class)  # for non cgans
else:
    synth_train_data = synth.sample(cond_array)  # # This uses the condition array


# End the training timer
generation_time = time.time() - start_time_gen
print("Finished Generating...\n")

# print data to review
print(synth_train_data.head(), "\n")

# Apply mapping to decode
# synth_train_data['label'] = synth_train_data['label'].map(synth_label_mapping)
# print(synth_train_data)
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
real_test_data = pd.DataFrame()
for data_set in training_data_sets:
    print(f"data set {data_set} out of {len(training_data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, data_set)
    df = pd.read_csv(data_path)
    real_test_data = pd.concat([real_test_data, df])

# Relabel the 'label' column using dict_7classes
if labelClass == "7+1":
    print("General Classes")
    real_test_data['label'] = real_test_data['label'].map(dict_7classes)

if labelClass == "1+1":
    print("Binary Classes")
    # Relabel the 'label' column using dict_2classes
    real_test_data['label'] = real_test_data['label'].map(dict_2classes)

# Shuffle data
real_test_data = shuffle(real_test_data, random_state=1)

#########################################################
#    Analyzing the Synthetic Data                       #
#########################################################

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels_synth = synth_train_data['label'].unique()

# Print the number of unique labels
# print(f"There are {unique_labels_synth} unique labels in the Synthetic dataset.")
num_unique_labels_synth = len(unique_labels_synth)
print(f"There are {num_unique_labels_synth} unique labels in the Synthetic dataset.")

# print the amount of instances for each label
class_counts_synth = synth_train_data['label'].value_counts()
print(class_counts_synth)

# Display the first few entries to verify the changes
print("Synthetic dataset:")
print(synth_train_data.head())

# prints an instance of each class
print("Each Instance in the Synthetic Training Dataset:")
for label in unique_labels_synth:
    instances = synth_train_data[synth_train_data['label'] == label]
    if not instances.empty:
        print(f"First instance of {label}:")
        print(instances.iloc[0])
    else:
        print(f"No instances found for label {label}")

#########################################################
#    Analyzing the Real Data                            #
#########################################################

# prove if the data is loaded properly
print("Real data:")
print(real_test_data[:2])
print(real_test_data.shape)

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels_real = real_test_data['label'].unique()

# Print the number of unique labels
# print(f"There are {unique_labels_real} unique labels in the Real dataset.")
num_unique_labels_real = len(unique_labels_real)
print(f"There are {num_unique_labels_real} unique labels in the Real dataset.")

class_counts_real = real_test_data['label'].value_counts()
print(class_counts_real)

# Display the first few entries to verify the changes
print("Real dataset:")
print(real_test_data.head())

# prints an instance of each class
print("Each Instance Before Encoding and Scaling for Real Test Dataset:")
for label in unique_labels_real:
    print(f"First instance of {label}:")
    print(real_test_data[real_test_data['label'] == label].iloc[0])

#########################################################
#    Preprocess Real Data                               #
#########################################################

# Shuffle data
real_test_data = shuffle(real_test_data, random_state=1)

# prints an instance of each class
print("Before Encoding and Scaling:")
unique_labels_synth = real_test_data['label'].unique()
for label in unique_labels_synth:
    print(f"First instance of {label}:")
    print(real_test_data[real_test_data['label'] == label].iloc[0])

# ---                   Scaling                     --- #

# Load up Scaler from GAN Training for Features
#scaler = joblib.load('RobustScaler_.pkl')

# scaler = joblib.load('PowerTransformer_.pkl')

# train the scalar on train data features
scaler.fit(real_test_data[num_cols])

# Scale the features in the real train dataframe
real_test_data[num_cols] = scaler.transform(real_test_data[num_cols])

# prove if the data is loaded properly
print("Real data After Scaling:")
print(real_test_data.head())
# print(real_train_data[:2])
print(real_test_data.shape)

# ---                   Labeling                     --- #

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels_synth = real_test_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels_synth} unique labels in the dataset.")

# print the amount of instances for each label
class_counts = real_test_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(real_test_data.head())

# Encodes the test label
label_encoder = LabelEncoder()
real_test_data['label'] = label_encoder.fit_transform(real_test_data['label'])

# Store label mappings
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label mappings:", label_mapping)

# Retrieve the numeric codes for classes
class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

# Print specific instances after label encoding
print("Real data After Encoding:")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(real_test_data[real_test_data['label'] == code].iloc[0])
print(real_test_data.head(), "\n")

#########################################################
#    Filtering Real Data to Match Synthetic Data Labels  #
#########################################################
# Extract unique labels from the synthetic data
synthetic_labels = set(synth_train_data['label'].unique())
print(f"Unique labels in the synthetic dataset: {synthetic_labels}")

# Filter the real data to only include labels that are present in the synthetic dataset
filtered_real_data = real_test_data[real_test_data['label'].isin(synthetic_labels)]
filtered_real_data_labels = set(filtered_real_data['label'].unique())
print(f"Filtered labels in the real dataset: {filtered_real_data_labels}")

# Optionally, balance or sample the real data to ensure the model is tested evenly across classes
# Here, we sample a fixed number (e.g., 10000 instances), but you can also use other sampling strategies.
sampled_real_test_data = filtered_real_data.sample(min(real_test_data_sample_size, len(filtered_real_data)), random_state=42)

print("Filtered and sampled real data statistics:")
print(sampled_real_test_data['label'].value_counts())

#########################################################
#    Splitting Synthetic Data as Features and Labels  #
#########################################################
X_train_synthetic = synth_train_data.drop('label', axis=1)  # Synthetic Features
y_train_synthetic = synth_train_data['label']               # Synthetic Labels

if model == "WGANGP" or model == "CTGAN":
    # re-encoding labels to go back to 0
    combined_label_encoder = LabelEncoder()

    y_train_synthetic = combined_label_encoder.fit_transform(y_train_synthetic)

#########################################################
#    Splitting Real Data as Features and Labels  #
#########################################################
X_test_real = sampled_real_test_data.drop('label', axis=1)  # Real Features
y_test_real = sampled_real_test_data['label']               # Real Labels

if model == "WGANGP" or model == "CTGAN":
    # re-encoding labels to go back to 0
    y_test_real = combined_label_encoder.fit_transform(y_test_real)

#########################################################
#    Setting up IDS Classifier Model  #
#########################################################

# Function to get evaluator based on type
def get_evaluator(evaluator_type, unique_labels_synth):
    if evaluator_type == 'XGBoost':
        print("Using XGBoost Classifier...\n")
        from xgboost import XGBClassifier
        if unique_labels_synth == 2:
            return XGBClassifier(objective='binary:logistic')
        return XGBClassifier()
    elif evaluator_type == 'LogisticRegression':
        print("Using LogisticRegression Classifier...\n")
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=42, n_jobs=-1)
    elif evaluator_type == 'Perceptron':
        print("Using Perceptron Classifier...\n")
        from sklearn.linear_model import Perceptron
        return Perceptron(random_state=42, n_jobs=-1)
    elif evaluator_type == 'AdaBoost':
        print("Using AdaBoost Classifier...\n")
        from sklearn.ensemble import AdaBoostClassifier
        return AdaBoostClassifier(random_state=42, algorithm='SAMME')
    elif evaluator_type == 'RandomForest':
        print("Using RandomForest Classifier...\n")
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42, n_jobs=-1)
    elif evaluator_type == 'DeepNeuralNetwork':
        print("Using DeepNeuralNetwork Classifier...\n")
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(random_state=42)
    elif evaluator_type == 'KNearestNeighbor':
        print("Using KNearestNeighbor Classifier...\n")
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(n_jobs=-1)
    else:
        print(f'Invalid evaluator model: {evaluator_type}')
        return None

#########################################################
#    Training and Testing Each Evaluator  #
#########################################################

# Directory to save classification report text files
report_dir = "./synth_data_reports"
os.makedirs(report_dir, exist_ok=True)

try:
    df_metrics = pd.read_json(path_or_buf=report_dir + '/synth_evaluator_metrics.json', orient='index')

except FileNotFoundError:
    df_metrics = pd.DataFrame(
        columns=['Synth', 'Label Classes', 'Evaluator', 'Test Duration', 'Accuracy', 'Precision', 'Recall', 'F1'])

for evaluator_type in evalautor_types:
    evaluator = get_evaluator(evaluator_type, unique_labels_synth)

    if evaluator is None:
        continue

    #########################################################
    #    Training  Evaluator Model  #
    #########################################################

    # Start the training timer
    start_time_train = time.time()
    print("Start Training...\n")

    # Train the model using the synthetic features and labels
    evaluator.fit(X_train_synthetic, y_train_synthetic)

    # End the training timer
    training_time = time.time() - start_time_train
    print("Finished Training...\n")

    #########################################################
    #    Testing IDS Model  #
    #########################################################

    # Start the training timer
    start_time_test = time.time()
    print("Start Testing...\n")

    # Use the real testing data set labels
    y_eval_pred = evaluator.predict(X_test_real)

    # End the training timer
    testing_time = time.time() - start_time_test
    print("Finished Testing...\n")

    #########################################################
    # Analyze Test Results  #
    #########################################################
    from sklearn.metrics import classification_report, accuracy_score

    # Calculating accuracy
    accuracy = accuracy_score(y_test_real, y_eval_pred)
    print(f"Accuracy of the model: {accuracy:.2%}")

    # Detailed classification report
    class_report = classification_report(y_test_real, y_eval_pred)
    print("Classification Report:")
    print(class_report)

    # classification report
    precision = precision_score(y_test_real, y_eval_pred, average='macro', zero_division=0.0)
    recall = recall_score(y_test_real, y_eval_pred, average='macro')
    f1 = f1_score(y_test_real, y_eval_pred, average='macro')

    evaluator_metrics = [model, labelClass, evaluator_type, testing_time, accuracy,
                         precision, recall, f1]

    #########################################################
    #         Saving Metrics and Results                     #
    #########################################################

    def save_results(evaluator_metrics):
        # Get the current timestamp
        timestamp = time.strftime("%Y%m%d%H%M%S")

        # Debugging: Print the column names and evaluator_metrics list
        print("DataFrame columns:", df_metrics.columns)
        print("Evaluator metrics:", evaluator_metrics)

        # Add metrics to dataframe and display
        update_row = df_metrics.loc[(df_metrics['Synth'] == model) &
                                    (df_metrics['Label Classes'] == labelClass) &
                                    (df_metrics['Evaluator'] == evaluator_type)]

        if update_row.empty:
            print("No existing record found. Adding new entry.")
            # No previous record
            df_metrics.loc[len(df_metrics.index)] = evaluator_metrics

            print(f'{evaluator_type} / {model} / {labelClass} Metrics')
            # display(df_metrics.loc[len(df_metrics.index) - 1])

        else:
            # Previous record exists, update it
            print("Existing record found. Updating entry.")
            df_metrics.loc[update_row.index, :] = evaluator_metrics

            print(f'{evaluator_type} / {model} / {labelClass} Metrics')
            # display(df_metrics.loc[(df_metrics['Synth'] == model) &
            #                        (df_metrics['Label Classes'] == labelClass) &
            #                        (df_metrics['Evaluator'] == evaluator_type)])

        # Save combined report
        df_metrics['Synth'] = pd.Categorical(df_metrics['Synth'], categories=synth_categories)
        df_metrics['Label Classes'] = pd.Categorical(df_metrics['Label Classes'], categories=label_classes)
        df_metrics['Evaluator'] = pd.Categorical(df_metrics['Evaluator'], categories=evalautor_types)

        df_metrics.sort_values(['Synth', 'Label Classes', 'Evaluator'], inplace=True)

        df_metrics.to_json(path_or_buf=report_dir + '/synth_evaluator_metrics.json', orient='index')

        synth_train_data.to_csv(f'../results/synthetic_EVALUATION_{model}_{evaluator_type}_{labelClass}_{timestamp}.csv', index=False)
        print("GAN reports saved successfully.")

    # Call save_results with the evaluator_metrics list
    save_results(evaluator_metrics)

    print(f"Generation time for Balanced Synthetic Dataset: {generation_time:.20f} seconds")
    print(f"Training time for Model: {training_time:.20f} seconds")
    print(f"Evaluation time for Real Dataset: {testing_time:.20f} seconds")

print(f"evaluation done for {model} {labelClass}")
