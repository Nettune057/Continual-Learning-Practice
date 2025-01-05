import pandas as pd
from configs.config import RAWS_DATA_PATH
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import StandardScaler

df1 = pd.read_csv(f'{RAWS_DATA_PATH}/CICIDS2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')
df2 = pd.read_csv(f'{RAWS_DATA_PATH}/CICIDS2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
df3 = pd.read_csv(f'{RAWS_DATA_PATH}/CICIDS2017/Friday-WorkingHours-Morning.pcap_ISCX.csv')
df4 = pd.read_csv(f'{RAWS_DATA_PATH}/CICIDS2017/Wednesday-workingHours.pcap_ISCX.csv')
df5 = pd.concat ([df1 , df2])
df6 = pd.concat ([df3 , df4])
df = pd.concat ([df5 , df6])

sample_data = df.sample(frac=0.1, random_state=42)

sample_data[' Source IP'] = sample_data[' Source IP'].apply(str)
sample_data[' Source Port'] = sample_data[' Source Port'].apply(str)
sample_data[' Destination IP'] = sample_data[' Destination IP'].apply(str)
sample_data[' Destination Port'] = sample_data[' Destination Port'].apply(str)
sample_data[' Source IP'] = sample_data[' Source IP'] + ':' + sample_data[' Source Port']
sample_data[' Destination IP'] = sample_data[' Destination IP'] + ':' + sample_data[' Destination Port']
sample_data.drop(columns=['Flow ID',' Source Port',' Destination Port', ' Timestamp'], inplace=True)

# Create new TCP_Flags column by summing all flag counts
sample_data['TCP_Flags'] = (sample_data['Fwd PSH Flags'] + 
                           sample_data[' Bwd PSH Flags'] + 
                           sample_data[' Fwd URG Flags'] + 
                           sample_data[' Bwd URG Flags'] +
                           sample_data['FIN Flag Count'] + 
                           sample_data[' SYN Flag Count'] + 
                           sample_data[' RST Flag Count'] + 
                           sample_data[' ACK Flag Count'] +
                           sample_data[' PSH Flag Count'] +
                           sample_data[' URG Flag Count'] +
                           sample_data[' CWE Flag Count'] +
                           sample_data[' ECE Flag Count'])

# Drop the individual flag columns
sample_data.drop(columns=['Fwd PSH Flags', 
                         ' Bwd PSH Flags',
                         ' Fwd URG Flags',
                         ' Bwd URG Flags',
                         ' PSH Flag Count',
                         ' URG Flag Count',
                         'FIN Flag Count',
                         ' CWE Flag Count',
                         ' ECE Flag Count',
                         ' SYN Flag Count',
                         ' RST Flag Count',
                         ' ACK Flag Count'], inplace=True)

sample_data.rename(columns={' Label': 'label'},inplace = True)
sample_data['label'] = sample_data['label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

sample_data['Fwd_Bytes'] = sample_data['Total Length of Fwd Packets']
sample_data['Bwd_Bytes'] = sample_data[' Total Length of Bwd Packets']

# Drop the original columns
sample_data.drop(columns=['Total Length of Fwd Packets', 
                         ' Total Length of Bwd Packets'], inplace=True)

# Keep only the specified columns
columns_to_keep = [' Source IP', 
                   ' Destination IP', 
                   ' Protocol',
                   ' Down/Up Ratio',
                   'Fwd_Bytes',
                   'Bwd_Bytes',
                   ' Total Fwd Packets',
                   ' Total Backward Packets',
                   'TCP_Flags',
                   ' Flow Duration',
                   'label']

# Keep only selected columns
sample_data = sample_data[columns_to_keep]

sample_data = sample_data.reset_index()
sample_data.replace([np.inf, -np.inf], np.nan,inplace = True)
sample_data.fillna(0,inplace = True)

sample_data.drop(columns=['index'],inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
     sample_data, sample_data['label'], test_size=0.3, random_state=123,stratify= sample_data['label'])

encoder = ce.TargetEncoder(cols=['TCP_Flags',' Protocol',' Down/Up Ratio'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)
X_test = encoder.transform(X_test)

scaler = StandardScaler()
cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns ))  - set(list(['label'])) )
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train['h'] = X_train[ cols_to_norm ].values.tolist()
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[ cols_to_norm ].values.tolist()