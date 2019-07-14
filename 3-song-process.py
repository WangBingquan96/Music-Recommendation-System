# Preprocessing on song.csv & song_extra_info.csv
import numpy as np
import pandas as pd
import os
import time
from scipy import stats

DATA_PATH = "../data/interim"

df_song     = pd.read_csv(os.path.join(DATA_PATH, "songs.csv"))
df_song_ex  = pd.read_csv(os.path.join(DATA_PATH, "song_extra_info.csv"))


# 1. [FG]song_length_log, song_length_short_than_1min, song_length_long_than_10min
df_song["song_length_log"] = np.log(df_song.song_length)
df_song["song_length_short_than_1min"] = df_song.song_length < 1*60*1000
df_song["song_length_long_than_10min"] = df_song.song_length > 10*60*1000
df_song.drop(["song_length"], axis=1, inplace=True)


# 2. [FG]genre_id_1, genre_id_2, genre_id_3
def get_genre(genre_ids, num):
    if pd.isnull(genre_ids):
        return np.nan
    genres = genre_ids.split("|")
    if num > len(genres):
        return np.nan
    return int(genre_ids.split("|")[num-1])

df_song["genre_id_1"] = df_song.genre_ids.apply(get_genre, args=(1,))
df_song["genre_id_2"] = df_song.genre_ids.apply(get_genre, args=(2,))
df_song["genre_id_3"] = df_song.genre_ids.apply(get_genre, args=(3,))
df_song.drop(["genre_ids"], axis=1, inplace=True)


# 3. [MI]genre_id_1, filled with mode of artist_name,lyricist,composer grouping
s1=time.time()
print("Before imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
mode = df_song[df_song.artist_name.notnull()].groupby("artist_name").genre_id_1.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_song.genre_id_1.fillna(mode, inplace=True)
print("After imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
s2=time.time()
print("Time: %s" % str(s2-s1))

s1=time.time()
print("Before imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
mode = df_song[df_song.lyricist.notnull()].groupby("lyricist").genre_id_1.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_song.genre_id_1.fillna(mode, inplace=True)
print("After imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
s2=time.time()
print("Time: %s" % str(s2-s1))

s1=time.time()
print("Before imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
mode = df_song[df_song.composer.notnull()].groupby("composer").genre_id_1.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_song.genre_id_1.fillna(mode, inplace=True)
print("After imputation, genre_id_1-missing: %s" % df_song.genre_id_1.isnull().sum())
s2=time.time()
print("Time: %s" % str(s2-s1))


# 4. [MI]language=-1 replaced by missing values, then filled with mode of lyricist grouping
s1=time.time()
df_song.loc[df_song.language==-1,"language"] = np.nan
print("Before imputation, language-missing: %s" % df_song.language.isnull().sum())
mode = df_song[df_song.lyricist.notnull()].groupby("lyricist").language.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_song.language.fillna(mode, inplace=True)
print("After imputation, language-missing: %s" % df_song.language.isnull().sum())
s2=time.time()
print("Time: %s" % str(s2-s1))


# 5. combine songs.csv & song_extra_info.csv
df_song_full = pd.merge(df_song, df_song_ex, on="song_id", how="left")
df_song_full.reset_index(drop=True, inplace=True)


# 6. [FG]state_code, extract the first two uppercase letters from ISRC,because it may represents the region where the music comes from
df_song_full["state_code"] = df_song_full["isrc"].apply(lambda x:x[:2] if pd.notnull(x) else np.nan)


# 7. [FG]state_is_XX
for i in df_song_full["state_code"].value_counts().index[:15]:
    name = "state_is_%s" % i
    df_song_full[name] = (df_song_full["state_code"] == i)


# 8. [MI]language. filled with mode of state_code grouping
s1=time.time()
print("Before imputation, language-missing: %s" % df_song_full.language.isnull().sum())
mode = df_song_full[df_song_full.state_code.notnull()].groupby("state_code").language.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_song_full.language.fillna(mode, inplace=True)
print("After imputation, language-missing: %s" % df_song_full.language.isnull().sum())
s2=time.time()
print("Time: %s" % str(s2-s1))


# 9. [DE]state_code, isrc
df_song_full.drop(["state_code","isrc"], axis=1, inplace=True)


# 10. Save
df_song_full.to_csv("../data/interim/Song.csv", index=False)