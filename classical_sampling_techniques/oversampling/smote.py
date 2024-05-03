#########################################################
#    Imports    #
#########################################################


# List all physical devices and configure them before any other operations
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE  # Import SMOTE
import random
from tqdm import tqdm
# from IPython.display import clear_output
import os
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.utils import shuffle

#########################################################
#    Loading the CSV    #
#########################################################
DATASET_DIRECTORY = './dataset/'  # If your dataset is within your python project directory, change this to the relative path to your dataset
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
    'flow_duration', 'Header_Length', 'Duration',
    'Rate', 'Srate', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight']
cat_cols = [
    'Protocol Type', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
    'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
    'ICMP', 'IPv', 'LLC']

# feature scaling
scaler = RobustScaler()

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

# Apply random over-sampling
min_class_size = full_data['label'].value_counts().min()

if min_class_size > 1:
    ros = SMOTE(k_neighbors=4, random_state=42)
else:
    print("Too few samples in the smallest class to apply SMOTE.")

X = full_data.drop('label', axis=1)
y = full_data['label']
X_res, y_res = ros.fit_resample(X, y)

# Combine the resampled features and labels back into a single DataFrame
full_data_resampled = pd.DataFrame(X_res, columns=X.columns)
full_data_resampled['label'] = y_res

# prints an instance of each class
print("Before encoding:")
unique_labels = full_data_resampled['label'].unique()
for label in unique_labels:
    print(f"First instance of {label}:")
    print(full_data_resampled[full_data_resampled['label'] == label].iloc[0])

# Shuffle data
full_data_resampled = shuffle(full_data_resampled, random_state=1)

# Scale the features in the dataframe
full_data_resampled[num_cols] = scaler.fit_transform(full_data_resampled[num_cols])

# prove if the data is loaded properly
print("Real data:")
print(full_data_resampled[:2])
print(full_data_resampled.shape)

# Relabel the 'label' column using dict_7classes
# full_data_resampled['label'] = full_data_resampled['label'].map(dict_7classes)

# # Relabel the 'label' column using dict_2classes
# full_data_resampled['label'] = full_data_resampled['label'].map(dict_2classes)

# Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
unique_labels = full_data_resampled['label'].nunique()

# Print the number of unique labels
print(f"There are {unique_labels} unique labels in the dataset.")

class_counts = full_data_resampled['label'].value_counts()
print(class_counts)

# Display the first few entries to verify the changes
print(full_data_resampled)

# prep the data to be inputted into model
data = full_data_resampled

#########################################################
#    Preprocessing / Clustering of Class Data    #
#########################################################

# encodes the label
label_encoder = LabelEncoder()
full_data_resampled['label'] = label_encoder.fit_transform(full_data_resampled['label'])

# Store label mappings
label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
print("Label mappings:", label_mapping)

# Retrieve the numeric codes for classes
class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

# Print specific instances after label encoding
print("After encoding:")
for label, code in class_codes.items():
    # Print the first instance of each class
    print(f"First instance of {label} (code {code}):")
    print(full_data_resampled[full_data_resampled['label'] == code].iloc[0])

# # Ensure all data for clustering is numeric
# clustering_features = [col for col in num_cols if col in minority_class_data.columns]  # Ensure these are only numeric
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
# full_data_resampled.loc[full_data_resampled['label'] == 'Benign', 'Cluster'] = labels
#
# # If 'Cluster' column has been created and needs encoding
# if 'Cluster' in full_data_resampled.columns:
#     full_data_resampled['Cluster'] = label_encoder.fit_transform(full_data_resampled['Cluster'])
#
# # Impute NaN values in 'label' and 'Cluster' with the mode (most frequent value)
# for column in ['label', 'Cluster']:
#     mode_value = full_data_resampled[column].mode()[0]
#     full_data_resampled[column].fillna(mode_value, inplace=True)
#
# print(full_data_resampled[['label', 'Cluster']].isna().sum())
#
# # Display some entries to verify the changes
# print(full_data_resampled[['label', 'Cluster']].head())

print(full_data_resampled.head())
