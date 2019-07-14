# Resample from train.csv to create our train set and test set
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

seed      = 1126
num       = 25*10000
DATA_PATH = "../data/raw"

# Generate Data set
df_train      = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))  #read the training data
rng           = np.random.RandomState(seed)
sample_index  = rng.choice(df_train.index.tolist(), num, replace=False) #Select 250000 samples from the training data randomly(Because our computer can not calcualte such big dataset )
resample_data = df_train.iloc[sample_index, :].reset_index(drop=True)
print("Positive sample rate: %s" % (resample_data.target.sum()/len(resample_data)))

# Split to train set & test set
X = resample_data.iloc[:, :-1].values
y = resample_data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed) #selcet 20 percent of sample randomly as test set
print("Number of Train set: %s" % (len(X_train)))
print("Number of Test set: %s" % (len(X_test)))
train_set  = pd.DataFrame(np.hstack([X_train, y_train.reshape(len(y_train),1)]), columns=resample_data.columns)
test_set   = pd.DataFrame(X_test, columns=resample_data.columns[:-1])
test_set_y = pd.DataFrame(y_test.reshape(len(y_test),1), columns=["target"])

# Save to "../data/interim"
train_set.to_csv("../data/interim/Train.csv", index=False)
test_set.to_csv("../data/interim/Test.csv", index=False)
test_set_y.to_csv("../data/processed/Test_target.csv", index=False)