#########################################################
#    Imports    #
#########################################################
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

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

import sklearn.cluster as cluster
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

#########################################################
#    Loading the CSV    #
#########################################################
DATASET_DIRECTORY = './archive/'          # If your dataset is within your python project directory, change this to the relative path to your dataset
csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]

print(csv_filepaths)

# If there are more than X CSV files, randomly select X files from the list
sample_size = 1

if len(csv_filepaths) > sample_size:
    csv_filepaths = random.sample(csv_filepaths, sample_size)
    print(csv_filepaths)

csv_filepaths.sort()

# list of csv files used
data_sets = csv_filepaths

num_cols = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration',
       'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
       'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
       'ece_flag_number', 'cwr_flag_number', 'ack_count',
       'syn_count', 'fin_count', 'urg_count', 'rst_count',
    'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP',
       'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min',
       'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Magnitue',
       'Radius', 'Covariance', 'Variance', 'Weight',
]
cat_cols = []

# feature scaling
scaler = StandardScaler()

for data_set in tqdm(data_sets):
    scaler.fit(pd.read_csv(DATASET_DIRECTORY + data_set)[num_cols])

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

# extracting data from csv to input into data frame
full_data = pd.DataFrame()
for data_set in data_sets:
    print(f"data set {data_set} out of {len(data_sets)} \n")
    data_path = os.path.join(DATASET_DIRECTORY, data_set)
    df = pd.read_csv(data_path)
    full_data = pd.concat([full_data, df])

# Shuffle data
full_data = shuffle(full_data, random_state=1)

# Scale the features in the dataframe
full_data[num_cols] = scaler.transform(full_data[num_cols])

# prove if the data is loaded properly
print("Real data:")
print(full_data[:2])
print(full_data.shape)

# Relabel the 'label' column using dict_7classes
# full_data['label'] = full_data['label'].map(dict_7classes)

# # Relabel the 'label' column using dict_2classes
# full_data['label'] = full_data['label'].map(dict_2classes)

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = full_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

class_counts = full_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(full_data.head())

# prep the data to be inputted into model
data = full_data

#########################################################
#    Preprocessing / Clustering of Class Data    #
#########################################################
# Assuming 'Attack' is the minority class; adjust as per your dataset analysis
minority_class_data = full_data.loc[full_data['label'] == 'Benign'].copy()

print(full_data[full_data['label'] == 'Attack'].iloc[0])
print(full_data[full_data['label'] == 'Benign'].iloc[0])

label_encoder = LabelEncoder()
full_data['label'] = label_encoder.fit_transform(full_data['label'])

# Store label mappings
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label mappings:", label_mapping)

# Retrieve the numeric codes for 'Attack' and 'Benign'
attack_code = label_encoder.transform(['Attack'])[0]
benign_code = label_encoder.transform(['Benign'])[0]

# Print specific instances after label encoding
print("After encoding:")
print(full_data[full_data['label'] == attack_code].iloc[0])
print(full_data[full_data['label'] == benign_code].iloc[0])


# Ensure all data for clustering is numeric
clustering_features = [col for col in num_cols if col in minority_class_data.columns]  # Ensure these are only numeric
#
# # --- KMeans Clustering ---
# algorithm = cluster.KMeans
# args, kwds = (), {'n_clusters': 2, 'random_state': 0}
# labels = algorithm(*args, **kwds).fit_predict(minority_class_data[clustering_features])
#
# # Creating a new DataFrame to see how many items are in each cluster
# cluster_counts = pd.DataFrame([[np.sum(labels == i)] for i in np.unique(labels)], columns=['count'], index=np.unique(labels))
# print("Cluster counts in the minority class:")
# print(cluster_counts)
#
# # Merging this back to the full dataset if needed
# full_data.loc[full_data['label'] == 'Benign', 'Cluster'] = labels
#
# # If 'Cluster' column has been created and needs encoding
# if 'Cluster' in full_data.columns:
#     full_data['Cluster'] = label_encoder.fit_transform(full_data['Cluster'])
#
#
# # Impute NaN values in 'label' and 'Cluster' with the mode (most frequent value)
# for column in ['label', 'Cluster']:
#     mode_value = full_data[column].mode()[0]
#     full_data[column].fillna(mode_value, inplace=True)
#
# print(full_data[['label', 'Cluster']].isna().sum())
#
# # Display some entries to verify the changes
# print(full_data[['label', 'Cluster']].head())

print(full_data.head())

#########################################################
#    Defining Training Parameters and Training Model    #
#########################################################

#Define the Conditional GAN and training parameters
noise_dim = 46
dim = 46
batch_size = 500
beta_1 = 0.5
beta_2 = 0.9

log_step = 100
epochs = 1 + 1
learning_rate = 5e-4
models_dir = '../cache'

#Test here the new inputs
gan_args = ModelParameters(batch_size=batch_size,
                           lr=learning_rate,
                           betas=(beta_1, beta_2),
                           noise_dim=noise_dim,
                           layers_dim=dim)

train_args = TrainParameters(epochs=epochs,
                             cache_prefix='cgan_cyberAttack',
                             sample_interval=log_step,
                             label_dim=-1,
                             labels=(0, 1))

#create a bining (WHY)
# minority_class_data[''] = pd.cut(minority_class_data[''], 5).cat.codes

#Init the Conditional GAN providing the index of the label column as one of the arguments
synth = RegularSynthesizer(modelname='cgan', model_parameters=gan_args)

#Training the Conditional GAN
synth.fit(data=full_data, label_cols=['label'], train_arguments=train_args, num_cols=num_cols, cat_cols=cat_cols)

#Saving the synthesizer
synth.save('cyberattack_cgan_model_full.pkl')

#########################################################
#    Loading and sampling from a trained synthesizer    #
#########################################################

synth = RegularSynthesizer.load('cyberattack_cgan_model_full.pkl')

# Optional Condition array
cond_array = pd.DataFrame(2000*[0, 1], columns=['label'])  # for cgans

# Generating synthetic samples
synth_data = synth.sample(cond_array)  # for cgans

# synth_data = synth.sample(100000)  # for non cgans

print(synth_data)

# find the amount of labels in the synth data
unique_labels = synth_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

class_counts = synth_data['label'].value_counts()
print(class_counts)

# Save the synthetic data to a CSV file
synth_data.to_csv('synthetic_data.csv', index=False)
