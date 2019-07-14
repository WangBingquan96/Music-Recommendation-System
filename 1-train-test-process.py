# Preprocessing on Train.csv & Test.csv
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings(action='ignore',category=DeprecationWarning)

DATA_PATH = "../data/interim"

df_train    = pd.read_csv(os.path.join(DATA_PATH, "Train.csv"))
df_test     = pd.read_csv(os.path.join(DATA_PATH, "Test.csv"))


# 1. combine Train.csv & Test.csv
df = pd.concat([df_train, df_test], axis=0, sort=True)
df.reset_index(drop=True, inplace=True)


# 2. [MI]DT on single missing: source_system_tab, source_screen_name, source_type
cols = ["source_screen_name","source_system_tab","source_type"]

A = df.source_screen_name.notnull()
B = df.source_system_tab.notnull()
C = df.source_type.notnull()

leA = LabelEncoder()
leB = LabelEncoder()
leC = LabelEncoder()

leA.fit(df[A].source_screen_name)
leB.fit(df[B].source_system_tab)
leC.fit(df[C].source_type)

No_nan_data = df[cols][A & B & C]
No_nan_data.source_screen_name = leA.transform(No_nan_data.source_screen_name)
No_nan_data.source_system_tab = leB.transform(No_nan_data.source_system_tab)
No_nan_data.source_type = leC.transform(No_nan_data.source_type)
No_nan_data = No_nan_data.values

A_tree_clf,B_tree_clf,C_tree_clf = DecisionTreeClassifier(max_depth=10, random_state=7),DecisionTreeClassifier(max_depth=10, random_state=7),DecisionTreeClassifier(max_depth=10, random_state=7)
trees = [A_tree_clf,B_tree_clf,C_tree_clf]
for i,j in enumerate(trees):
    index = [0,1,2]
    index.remove(i)
    temp_X_train, temp_X_test, temp_y_train, temp_y_test = train_test_split(No_nan_data[:,index], No_nan_data[:,i], test_size=0.3, random_state=7)
    j.fit(temp_X_train, temp_y_train)
    print("Accuracy: %s" % j.score(temp_X_test,temp_y_test))

print("Imputation: only-A-missing")
temp = df.loc[(-A) & (B) & (C), ['source_system_tab','source_type']]
temp.source_system_tab = leB.transform(temp.source_system_tab)
temp.source_type = leC.transform(temp.source_type)
result = A_tree_clf.predict(temp.values)
result = leA.inverse_transform(result)
df.loc[(-A) & (B) & (C),'source_screen_name'] = result
print("Imputation: only-B-missing")
temp = df.loc[(A) & (-B) & (C), ['source_screen_name','source_type']]
temp.source_screen_name = leA.transform(temp.source_screen_name)
temp.source_type = leC.transform(temp.source_type)
result = B_tree_clf.predict(temp.values)
result = leB.inverse_transform(result)
df.loc[(A) & (-B) & (C),'source_system_tab'] = result
print("Imputation: only-C-missing")
temp = df.loc[(A) & (B) & (-C), ['source_screen_name','source_system_tab']]
temp.source_screen_name = leA.transform(temp.source_screen_name)
temp.source_system_tab = leB.transform(temp.source_system_tab)
result = C_tree_clf.predict(temp.values)
result = leC.inverse_transform(result)
df.loc[(A) & (B) & (-C),'source_type'] = result


# 3. [MI]source_system_tab, source_screen_name, source_type, missing value = " "
df[cols] = df[cols].fillna(" ")


# 4. [FG]combine_tab_name_type
df["combine_tab_name_type"] = df["source_screen_name"]+"+"+df["source_system_tab"]+"+"+df["source_type"]


# 5. [FG]msno_song & song_msno
msno_song = df.groupby("msno").apply(lambda x:len(x["song_id"]))
song_msno = df.groupby("song_id").apply(lambda x:len(x["msno"]))
df["msno_per_song"] = df["song_id"].map(song_msno)
df["song_per_msno"] = df["msno"].map(msno_song)


# 6. Save
df.to_csv("../data/interim/Train_Test.csv", index=False)