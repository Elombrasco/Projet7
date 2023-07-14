import pandas as pd


#Load the data and model
df = pd.read_csv('data/X_sample.csv', index_col="SK_ID_CURR", encoding="utf-8")

def info(customer_id):
    gender = ""
    Age = ""
    if df.loc[int(customer_id), "CODE_GENDER_M"] == 1:
        gender = "Male"
    else:
        gender = "Female"
    return gender, Age

print(info("280998"))

1. `sorted_idx = np.argsort(lgbm.feature_importances_)[::-1]` : Cette ligne calcule les indices des fonctionnalités (features) dans l'ordre décroissant de leur importance. Il utilise `np.argsort` pour obtenir les indices triés en fonction des valeurs d'importance renvoyées par `lgbm.feature_importances_`. Ensuite, `[::-1]` inverse l'ordre pour avoir une liste en ordre décroissant.

2. `for index in sorted_idx:` : Cette ligne itère sur les indices triés des fonctionnalités.

3. `print([X_data.columns[index], lgbm.feature_importances_[index]])` : Cette ligne imprime le nom de la fonctionnalité (colonne) correspondante, ainsi que son importance renvoyée par `lgbm.feature_importances_[index]`. `X_data.columns[index]` renvoie le nom de la colonne associée à l'indice `index`, et `lgbm.feature_importances_[index]` renvoie l'importance de cette fonctionnalité selon le modèle LightGBM.

