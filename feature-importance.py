import numpy as np
import pandas as pd
import os
import time
import pickle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier


#read the processed train data
DATA_PATH       = r".\data\interim"
DATA_PATH_FINAL = r".\data\processed"

df         = pd.read_csv(os.path.join(DATA_PATH, "Train_Test.csv"))
df_member  = pd.read_csv(os.path.join(DATA_PATH, "Member.csv"))
df_song    = pd.read_csv(os.path.join(DATA_PATH, "Song.csv"))

temp = pd.merge(df, df_song, on="song_id", how="left")
df   = pd.merge(temp, df_member, on="msno", how="left")


# Because there are fews songs in train&test sets, which are not included in song.csv
# 1. [MI]Mode
def imput_by_mode(col_list):
	for i in col_list:
		mode = stats.mode(df[i])[0][0]
		df[i].fillna(mode, inplace=True)
cols = ['language','genre_id_1',"genre_id_2","genre_id_3",'state_is_US','state_is_GB',
        'state_is_JP','state_is_TW','state_is_HK','state_is_KR',"state_is_QM","state_is_FR","state_is_NL","state_is_DE",
        "state_is_AU","state_is_CA","state_is_TC","state_is_IT",
        "state_is_ES"]
imput_by_mode(cols)

# 2. [MI]Mean
df["song_length_log"].fillna(df["song_length_log"].mean(), inplace=True)
df_song["song_length_short_than_1min"] = df_song.song_length < 1*60*1000
df_song["song_length_long_than_10min"] = df_song.song_length > 10*60*1000

# 3. Last Label Encoding
def factorize_save_encoding(col_list):
	dict_ = dict()
	for i in col_list:
		temp = pd.factorize(df[i])
		df[i] = temp[0]+1
		dict_[i] = temp[1]
	return dict_
decoder = factorize_save_encoding(["msno","song_id","source_screen_name",
	"source_system_tab","source_type","combine_tab_name_type",
	"artist_name","composer","lyricist","language","genre_id_1","genre_id_2","genre_id_3"])

# 4. Save decoder
with open(os.path.join(DATA_PATH_FINAL, "decoder.pkl"), "wb") as f:
	pickle.dump(decoder, f)


# 5. Save id-info
id_cols = ["msno","song_id","artist_name","target"]
df[id_cols].to_csv(os.path.join(DATA_PATH_FINAL, "Train-Test-id.csv"), index=False)


# 6. [FE]Conditional Probability
print("Start")
s1=time.time()
df["p_source_type_msno"]           = df.groupby(["msno","source_type"]).source_type.transform(len)/df.groupby(["msno"]).source_type.transform(len)
df["p_source_screen_name_msno"]    = df.groupby(["msno","source_screen_name"]).source_screen_name.transform(len)/df.groupby(["msno"]).source_screen_name.transform(len)
df["p_source_system_tab_msno"]     = df.groupby(["msno","source_system_tab"]).source_system_tab.transform(len)/df.groupby(["msno"]).source_system_tab.transform(len)
df["p_source_type_song_id"]        = df.groupby(["song_id","source_type"]).source_type.transform(len)/df.groupby(["song_id"]).source_type.transform(len)
df["p_source_screen_name_song_id"] = df.groupby(["song_id","source_screen_name"]).source_screen_name.transform(len)/df.groupby(["song_id"]).source_screen_name.transform(len)
df["p_source_system_tab_song_id"]  = df.groupby(["song_id","source_system_tab"]).source_system_tab.transform(len)/df.groupby(["song_id"]).source_system_tab.transform(len)
df["p_artist_name_msno"]           = df.groupby(["msno","artist_name"]).artist_name.transform(len)/df.groupby(["msno"]).artist_name.transform(len)
df["p_language_msno"]              = df.groupby(["msno","language"]).language.transform(len)/df.groupby(["msno"]).language.transform(len)
df["p_genre_id_1_msno"]            = df.groupby(["msno","genre_id_1"]).genre_id_1.transform(len)/df.groupby(["msno"]).genre_id_1.transform(len)
s2=time.time()
print("End, Time: %s" % str(s2-s1))


# 7. [DE]ID info
df.drop(["msno","song_id","artist_name","composer","lyricist","name"], axis=1, inplace=True)

# 8. feature importance calculated by Random Forest
df_train = df.loc[df.target.notnull(), :]
df_test = df.loc[df.target.isnull(), :]

#separate the x and y in the train dateset
X_train = df_train.drop(["target"], axis=1)
y_train = df_train.target
X_train = X_train.values
y_train = y_train.values

tree_clf = RandomForestClassifier(n_estimators=10000, max_depth=10, oob_score=True, n_jobs=-1)
tree_clf.fit(X_train,y_train)
importance = pd.Series(tree_clf.feature_importances_, index=cols).sort_values(ascending=False)
importance.to_csv(os.path.join(DATA_PATH_FINAL, "feature_importance.csv"))
# We can read the csv file and draw the cummulative feature importance to see how many features are effective