# Preprocessing on members.csv
import numpy as np
import pandas as pd
import os
from scipy import stats
from datetime import datetime

DATA_PATH = "../data/interim"

df_member   = pd.read_csv(os.path.join(DATA_PATH, "members.csv"))

# 1. Label encoding city, no 2
df_member["city"] = pd.factorize(df_member.city)[0]+1

# 2. [MI]bd(Age), >=100<=5 replaced by missing values, then filled with mode of city grouping
df_member.loc[df_member.bd <= 5, 'bd']   = np.nan
df_member.loc[df_member.bd >= 100, 'bd'] = np.nan
mode = df_member.groupby("city").bd.transform(lambda x: np.nan if x.isnull().all() else stats.mode(x)[0][0])
df_member.bd.fillna(mode, inplace=True)

# 3. [FG]age_state, <18, >=18&<60, >=60
df_member["age_state"] = df_member.bd.apply(lambda x:1 if x<18 else (3 if x >=60 else 2))

# 4. [MI] Label encoding gender
df_member["gender"] = pd.factorize(df_member.gender)[0]+1

# 5. Label encoding registered_via
df_member["registered_via"] = pd.factorize(df_member.registered_via)[0]+1

# 6. [FG]registration_year
df_member["registration_year"] = df_member["registration_init_time"]//10000

# 7. [FG]active_time_log
df_member["active_time_log"] = np.log((datetime.now() - pd.to_datetime(df_member.registration_init_time, format = "%Y%m%d")).apply(lambda x:x.days))

# 8. [De]expiration_date, registration_init_time
df_member.drop(["registration_init_time","expiration_date"], axis=1, inplace=True)

# 9. Save
df_member.to_csv("../data/interim/Member.csv", index=False)