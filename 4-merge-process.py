# Preprocessing on three charts
import numpy as np
import pandas as pd
import os
import time
import pickle
from scipy import stats

DATA_PATH       = r".\data\interim"
DATA_PATH_FINAL = r".\data\processed"

#print(os.getcwd())

df         = pd.read_csv(os.path.join(DATA_PATH, "Train_Test.csv"))
df_member  = pd.read_csv(os.path.join(DATA_PATH, "Member.csv"))
df_song    = pd.read_csv(os.path.join(DATA_PATH, "Song.csv"))

temp = pd.merge(df, df_song, on="song_id", how="left")
df   = pd.merge(temp, df_member, on="msno", how="left")

# Less Importance by RF
df.drop(["state_is_QM","state_is_FR","state_is_NL","state_is_DE",
"state_is_AU","state_is_CA","state_is_TC","state_is_IT",
"state_is_ES","genre_id_2","genre_id_3",
"song_length_short_than_1min","song_length_long_than_10min","name"], axis=1, inplace=True)

# Because there are fews songs in train&test sets, which are not included in song.csv
# 1. [MI]Mode
def imput_by_mode(col_list):
	for i in col_list:
		mode = stats.mode(df[i])[0][0]
		df[i].fillna(mode, inplace=True)
cols = ['language','genre_id_1','state_is_US','state_is_GB',
        'state_is_JP','state_is_TW','state_is_HK','state_is_KR']
imput_by_mode(cols)


# 2. [MI]Mean
df["song_length_log"].fillna(df["song_length_log"].mean(), inplace=True)


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
	"artist_name","composer","lyricist","language","genre_id_1"])


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
df.drop(["msno","song_id","artist_name","composer","lyricist"], axis=1, inplace=True)


# 8. Save final
df.loc[df.target.notnull(), :].to_csv(os.path.join(DATA_PATH_FINAL, "Train_pre.csv"), index=False)
df.loc[df.target.isnull(), :].to_csv(os.path.join(DATA_PATH_FINAL, "Test_pre.csv"), index=False)


