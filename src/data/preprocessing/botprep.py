import random
import socket
import struct

import category_encoders as ce
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('../../../data/raw/NF-BoT-IoT.csv')
data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(lambda x: socket.inet_ntoa(struct.pack('>I', random.randint(0xac100001, 0xac1f0001))))

data['IPV4_SRC_ADDR'] = data.IPV4_SRC_ADDR.apply(str)
data['L4_SRC_PORT'] = data.L4_SRC_PORT.apply(str)
data['IPV4_DST_ADDR'] = data.IPV4_DST_ADDR.apply(str)
data['L4_DST_PORT'] = data.L4_DST_PORT.apply(str)

data['IPV4_SRC_ADDR'] = data['IPV4_SRC_ADDR'] + ':' + data['L4_SRC_PORT']
data['IPV4_DST_ADDR'] = data['IPV4_DST_ADDR'] + ':' + data['L4_DST_PORT']

data.drop(columns=['L4_SRC_PORT','L4_DST_PORT'],inplace=True)

data.drop(columns=['Attack'],inplace = True)

data.rename(columns={"Label": "label"},inplace = True)

label = data.label

data.drop(columns=['label'],inplace = True)

scaler = StandardScaler()

data =  pd.concat([data, label], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
     data, label, test_size=0.3, random_state=123,stratify= label)

encoder = ce.TargetEncoder(cols=['TCP_FLAGS','L7_PROTO','PROTOCOL'])
encoder.fit(X_train, y_train)
X_train = encoder.transform(X_train)

cols_to_norm = list(set(list(X_train.iloc[:, 2:].columns ))  - set(list(['label'])) )
X_train[cols_to_norm] = scaler.fit_transform(X_train[cols_to_norm])
X_train['h'] = X_train[ cols_to_norm ].values.tolist()

X_test = encoder.transform(X_test)


X_test = encoder.transform(X_test)
X_test[cols_to_norm] = scaler.transform(X_test[cols_to_norm])
X_test['h'] = X_test[ cols_to_norm ].values.tolist()