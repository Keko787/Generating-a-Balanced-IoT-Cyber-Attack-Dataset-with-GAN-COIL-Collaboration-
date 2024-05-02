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

import matplotlib.pyplot as plt

from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters


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
from sklearn.metrics import confusion_matrix

# Print versions and device configurations after ensuring GPU settings
print("TensorFlow version:", tf.__version__)
# print("CUDA version:", tf.sysconfig.get_build_info()['cuda_version'])
# print("cuDNN version:", tf.sysconfig.get_build_info()['cudnn_version'])
# print(tf.config.list_physical_devices(), "\n", tf.config.list_logical_devices(), "\n")
# print(tf.config.list_physical_devices('GPU'), "\n")

#########################################################
#    Loading GAN Model and Generating Data   #
#########################################################
synth = RegularSynthesizer.load('cyberattack_cwgangp_model_full.pkl')

samples_per_class = 1000  # Adjust this as needed

# Create an array that contains the class code repeated for the number of samples per class
conditions = []
for code in class_codes.values():
    conditions.extend([code] * samples_per_class)

# Optionally shuffle the conditions to randomize the order
# np.random.shuffle(conditions)

# Create a DataFrame for these conditions
cond_array = pd.DataFrame(conditions, columns=['label'])

# Generating synthetic samples
synth_data = synth.sample(cond_array)  # # This uses the condition array

# synth_data = synth.sample(100000)  # for non cgans

print(synth_data)

label_mapping = {0: 'Attack', 1: 'Benign'}

# Apply mapping to decode
synth_data['label'] = synth_data['label'].map(label_mapping)
print(synth_data)
#########################################################
#    Loading Real Data   #
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
cat_cols = ['label']

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
full_data['label'] = full_data['label'].map(dict_2classes)

# prep the data to be inputted into model
data = full_data
#########################################################
#    Analyzing the Synthetic Data   #
#########################################################

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = synth_data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the Synthetic dataset.")

class_counts = synth_data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(synth_data.head())

#########################################################
#    Analyzing the Real Data   #
#########################################################
# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels_real = data['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels_real} unique labels in the Real dataset.")

class_counts = data['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print("Synthetic dataset:")
print(synth_data.head())
#########################################################
#    Loading Synthetic Data as Training data  #
#########################################################
# Assuming your synthetic data is correctly labeled and structured similar to real data
X_synthetic = synth_data.drop('label', axis=1)  # Synthetic Features
y_synthetic = synth_data['label']               # Synthetic Labels

#########################################################
#    Filtering Real Data to Match Synthetic Data Labels  #
#########################################################
# Extract unique labels from the synthetic data
synthetic_labels = set(synth_data['label'].unique())
print(f"Unique labels in the synthetic dataset: {synthetic_labels}")

# Filter the real data to only include labels that are present in the synthetic dataset
filtered_real_data = full_data[full_data['label'].isin(synthetic_labels)]
filtered_real_data_labels = set(filtered_real_data['label'].unique())
print(f"Filtered labels in the real dataset: {filtered_real_data_labels}")

# Optionally, balance or sample the real data to ensure the model is tested evenly across classes
# Here, we sample a fixed number (e.g., 10000 instances), but you can also use other sampling strategies.
sampled_real_data = filtered_real_data.sample(min(100000, len(filtered_real_data)), random_state=42)

print("Filtered and sampled real data statistics:")
print(sampled_real_data['label'].value_counts())

#########################################################
#    Loading Real Data as Testing data  #
#########################################################
X_real = sampled_real_data.drop('label', axis=1)  # Features
y_real = sampled_real_data['label']               # Labels

#########################################################
#    Setting up Random Forest Classifier Model  #
#########################################################
from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

#########################################################
#    Training  Random Forest Classifier Model  #
#########################################################
# Train the model using the synthetic data
rf_classifier.fit(X_synthetic, y_synthetic)

#########################################################
#    Testing  Random Forest Classifier Model  #
#########################################################
# Use the real testing data set
y_pred = rf_classifier.predict(X_real)
#########################################################
# Analyze Test Results  Random Forest Classifier Model  #
#########################################################
from sklearn.metrics import classification_report, accuracy_score

# Calculating accuracy
accuracy = accuracy_score(y_real, y_pred)
print(f"Accuracy of the model: {accuracy:.2%}")

# Detailed classification report
print("Classification Report:")
print(classification_report(y_real, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_real, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Get unique labels
labels = sorted(y_real.unique())

fig, ax = plt.subplots(figsize=(10, 7))  # Adjust the figure size as needed
cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)  # Apply a color map
fig.colorbar(cax)  # Add a color bar

# Set axis labels with rotation for x-axis labels
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="left")  # Rotate x-axis labels for better visibility
ax.set_yticks(np.arange(len(labels)))
ax.set_yticklabels(labels)

# Adjust the margins and layout
plt.gcf().subplots_adjust(bottom=0.15)  # Increase bottom margin

# Ensure every label is displayed and add grid lines for better readability
ax.set_xticks(np.arange(conf_matrix.shape[1]+1)-.5, minor=True)
ax.set_yticks(np.arange(conf_matrix.shape[0]+1)-.5, minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
ax.tick_params(which="minor", size=0)  # Remove minor tick marks

# Setting labels and title
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')

# Add text annotations inside the heatmap squares
for (i, j), val in np.ndenumerate(conf_matrix):
    ax.text(j, i, f'{val}', ha='center', va='center', color='red')

plt.show()
