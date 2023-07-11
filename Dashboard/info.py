import pandas as pd


#Load the data and model
df = pd.read_csv('data/X_sample.csv', index_col="SK_ID_CURR", encoding="utf-8")

def info(customer_id):
    gender = ""
    Age = ""
    if df.loc[customer_id, "CODE_GENDER_M"] == 1:
        gender = "Male"
    else:
        gender = "Female"
    return gender, Age

print(info(280998))
