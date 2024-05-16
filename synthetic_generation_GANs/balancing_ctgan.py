#########################################################
#    Imports    #
#########################################################
import json
import subprocess

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
from sklearn.metrics import accuracy_score, f1_score

# Print versions and device configurations after ensuring GPU settings
print("TensorFlow version:", tf.__version__)
# print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
# print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
# print(tf.config.list_physical_devices(), "\n", tf.config.list_logical_devices(), "\n")
# print(tf.config.list_physical_devices('GPU'), "\n")

timestamp_experiment = time.strftime("%Y%m%d%H%M%S")
#########################################################
#    Loading the Real Data    #
#########################################################
DATASET_DIRECTORY = '../archive/'

# List the files in the dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
print(csv_filepaths)

# If there are more than X CSV files, randomly select X files from the list
sample_size = 1
if len(csv_filepaths) > sample_size:
    csv_filepaths = random.sample(csv_filepaths, sample_size)
    print(csv_filepaths)
csv_filepaths.sort()

# list of csv files used for training data sets
training_data_sets = csv_filepaths

# Mapping Features
num_cols = [
    'flow_duration', 'Header_Length', 'Duration',
    'Rate', 'Srate', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
]

cat_cols = [
    'Protocol Type', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC',
]

# Mapping Labels
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
                 'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'}

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
                 'DictionaryBruteForce': 'Attack'}

# Extracting data from csv to input into data frame
real_train_data = pd.DataFrame()
for data_set in training_data_sets:
    print(f"data set {data_set} out of {len(training_data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, data_set)
    df = pd.read_csv(data_path)
    real_train_data = pd.concat([real_train_data, df])

# Relabel the 'label' column using dict_7classes
# real_train_data['label'] = real_train_data['label'].map(dict_7classes)

# Relabel the 'label' column using dict_2classes
# real_train_data['label'] = real_train_data['label'].map(dict_2classes)

#########################################################
#    Preprocessing Data                                 #
#########################################################

# Shuffle data
real_train_data = shuffle(real_train_data, random_state=1)

# prints an instance of each class
print("Before Encoding and Scaling:")
unique_labels = real_train_data['label'].unique()
for label in unique_labels:
    print(f"First instance of {label}:")
    print(real_train_data[real_train_data['label'] == label].iloc[0])

# ---                   Scaling                     --- #

# Setting up Scaler for Features
# scaler = RobustScaler()
scaler = MinMaxScaler(feature_range=(0, 1))
# transformer = PowerTransformer(method='yeo-johnson')

# train the scalar on train data features
scaler.fit(real_train_data[num_cols])

# Save the Scaler for use in other files
# joblib.dump(scaler, 'RobustScaler_.pkl')
joblib.dump(scaler, './scalar_models/MinMaxScaler_.pkl')
# joblib.dump(scaler, 'PowerTransformer_.pkl')

# Scale the features in the real train dataframe
real_train_data[num_cols] = scaler.transform(real_train_data[num_cols])

# prove if the data is loaded properly
print("Real data After Scaling:")
print(real_train_data.head())
# print(real_train_data[:2])
print(real_train_data.shape)

# ---                   Labeling                     --- #

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = real_train_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

# print the amount of instances for each label
class_counts = real_train_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(real_train_data.head())

# Encodes the training label
label_encoder = LabelEncoder()
real_train_data['label'] = label_encoder.fit_transform(real_train_data['label'])

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
    print(real_train_data[real_train_data['label'] == code].iloc[0])
print(real_train_data.head(), "\n")

#########################################################
#    Defining Training Parameters and Training Model    #
#########################################################

# Extracting numeric codes for the labels from the previously created class_codes dictionary
labels_tuple = tuple(class_codes.values())

#  --- Define the Conditional GAN and training parameters ---

# values for model settings
noise_dim = 46
dim = 46
batch_size = 500
beta_1 = 0.5
beta_2 = 0.9

# neurons and layers for each sub model
generator_layers = [32, 16, 8]
critic_layers = [32]

# values for training settings
log_step = 10
label_amount = 34
epochs = 100 + 1
learning_rate = 5e-4
models_dir = 'GAN_analysis/cache'

ctgan_args = ModelParameters(batch_size=batch_size,
                             lr=learning_rate,
                             betas=(beta_1, beta_2),
                             layers_dim=dim,
                             noise_dim=noise_dim,
                             n_cols=dim,
                             n_features=dim,
                             generator_dims=generator_layers,
                             critic_dims=critic_layers,
                             latent_dim=dim,
                             )

# Init the CTGAN
synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)

train_args = TrainParameters(cache_prefix='cgan_cyberAttack',
                             label_dim=label_amount,
                             epochs=epochs,
                             sample_interval=log_step,
                             log_frequency=True,
                             labels=labels_tuple,
                             )

prep_real_train_data, prep_labels = synth._prep_fit(data=real_train_data, label_cols=['label'], num_cols=num_cols
                                                    , cat_cols=cat_cols)
# Start the training timer
print("Start Training...\n")
# start the hard logging
proc = subprocess.Popen(['python', './GAN_analysis/hardwareAnalyzer.py'])

# start the training timer
start_time_train = time.time()

# Training the Conditional GAN
synth.fit(data=real_train_data, label_cols=['label'], train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

# End the training timer
training_time = time.time() - start_time_train

# Ensure we kill the subprocess when done
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()

print("Training Over...\n")

# Saving the synthesizer
synth.save('./GAN_models/cyberattack_cwgangp_model.pkl')

#########################################################
#    Loading GAN and Generating Samples                 #
#########################################################

# Load the GAN Model
synth = RegularSynthesizer.load(f'./scalar_models/MinMaxScaler_{timestamp_experiment}.pkl')
synth.output_dimensions()

samples_per_class = 1000  # Adjust this as needed

# Create an array that contains the class code repeated for the number of samples per class
conditions = []
for code in class_codes.values():
    conditions.extend([code] * samples_per_class)

# Optionally shuffle the conditions to randomize the order
# np.random.shuffle(conditions)

# Create a DataFrame for these conditions
cond_array = pd.DataFrame(conditions, columns=['label'])

# start the hardware logging
proc = subprocess.Popen(['python', './GAN_analysis/hardwareAnalyzer.py'])

# Start the training timer
start_time_gen = time.time()
print("Start Generating...\n")

# Generating synthetic samples
synth_data = synth.sample(cond_array)  # # This uses the condition array

# End the training timer
generation_time = time.time() - start_time_gen

# Ensure we kill the subprocess when done
proc.terminate()
try:
    proc.wait(timeout=10)
except subprocess.TimeoutExpired:
    proc.kill()
print("Finished Generating...\n")

#########################################################
#               Postprocessing and Analysis             #
#########################################################

scaler = joblib.load(f'./scalar_models/MinMaxScaler_{timestamp_experiment}.pkl')

# find the amount of labels in the synth data
unique_labels = synth_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

# print the amount of instances for each label
class_counts = synth_data['label'].value_counts()
print(class_counts, "\n")

# prove that the scaled data is proper by printing each instance
print("Synthetic Data (SCALED):")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(synth_data[synth_data['label'] == code].iloc[0])
print(synth_data.head(), "\n")

print("real train data Data (SCALED):")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(real_train_data[real_train_data['label'] == code].iloc[0])
print(real_train_data.head(), "\n")

# inverse the scale on synthetic data
synth_data[num_cols] = scaler.inverse_transform(synth_data[num_cols])

# inverse the scale on synthetic data
real_train_data[num_cols] = scaler.inverse_transform(real_train_data[num_cols])

# prove that the unscaled data is proper by printing each instance
print("Synthetic Data (UNSCALED):")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(synth_data[synth_data['label'] == code].iloc[0])
print(synth_data.head(), "\n")

print("real train data Data (UNSCALED):")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(real_train_data[real_train_data['label'] == code].iloc[0])
print(real_train_data.head(), "\n")

# Decode the synthetic data labels
code_to_class = {v: k for k, v in class_codes.items()}
synth_data['label'] = synth_data['label'].map(code_to_class)

# Decode labels in the real dataset using the LabelEncoder
real_train_data['label'] = label_encoder.inverse_transform(real_train_data['label'])

# Print some of the decoded data
print(synth_data.head(), "\n")


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


print(f"Training time for Model: {training_time:.20f} seconds")
print(f"Generation time for Balanced Synthetic Dataset: {generation_time:.20f} seconds")

# Plot class distribution for both real and synthetic data
plot_class_distribution(real_train_data, 'Real Data Class Distribution')
plot_class_distribution(synth_data, 'Synthetic Data Class Distribution')

# Plot feature comparisons (adjust 'feature1' and 'feature2' to your dataset's features)
plot_feature_comparison(real_train_data, synth_data, 'flow_duration', 'Duration')

# Provide a Report of each feature and other stats from Ydata profiling
original_report = ProfileReport(real_train_data, title='Original Data', minimal=True)
resampled_report = ProfileReport(synth_data, title='Resampled Data', minimal=True)
comparison_report = original_report.compare(resampled_report)
comparison_report.to_file(f'./GAN_analysis/profile_reports/ctgan_original_vs_synth_{timestamp_experiment}.html')


#########################################################
#         Saving Metrics and Results                    #
#########################################################


def save_results(model_name, training_time_, generation_time_):
    # Directory to save classification report text files
    report_dir = "synth_data_reports"
    os.makedirs(report_dir, exist_ok=True)

    # Format the filenames to include the model name and type of dataset
    filename = f"{model_name}_train_report_{timestamp_experiment}.txt"

    # Combine reports with accuracy, confusion matrix, training and evaluation times for imbalanced dataset
    imbalanced_report = {
        "training_time_seconds": training_time_,
        "generation_time_seconds": generation_time_
    }

    # Save combined report for the imbalanced dataset
    report_filename = os.path.join(report_dir, filename)
    with open(report_filename, "w") as report_file:
        json.dump(imbalanced_report, report_file, indent=4)

    print("GAN reports saved successfully.")


save_results('ctgan', training_time, generation_time)

# Save the synthetic data to a CSV file
synth_data.to_csv(f'./GAN_analysis/results/synthetic_data_ctgan_{timestamp_experiment}.csv', index=False)
