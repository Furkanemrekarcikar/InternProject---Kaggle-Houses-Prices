from EDA import *

print("Veri seti yüklendi\n")
print("Eksik veriler dolduruluyor\n")

full_df.loc[(full_df['MasVnrArea'] == 0) & (full_df['MasVnrType'].isnull()), 'MasVnrType'] = 'None'
full_df.loc[(full_df['MasVnrType'] == 'None') & (full_df['MasVnrArea'].isnull()), 'MasVnrArea'] = 0
full_df.loc[(full_df['MasVnrType'].isnull()) & (full_df['MasVnrArea'] > 0), 'MasVnrType'] = 'BrkFace'
full_df.loc[full_df['MasVnrType'].isnull(), 'MasVnrType'] = 'None'
full_df.loc[full_df['MasVnrArea'].isnull(), 'MasVnrArea'] = 0

full_df['BsmtFinSF1'] = full_df['BsmtFinSF1'].fillna(0)
full_df['BsmtFinSF2'] = full_df['BsmtFinSF2'].fillna(0)
full_df['BsmtUnfSF'] = full_df['BsmtUnfSF'].fillna(0)
full_df['TotalBsmtSF'] = full_df['TotalBsmtSF'].fillna(0)

full_df['BsmtFullBath'] = full_df['BsmtFullBath'].fillna(0)
full_df['BsmtHalfBath'] = full_df['BsmtHalfBath'].fillna(0)

full_df['GarageCars'] = full_df['GarageCars'].fillna(0)
full_df['GarageArea'] = full_df['GarageArea'].fillna(0)

full_df['Exterior1st'] = full_df['Exterior1st'].fillna('None')
full_df['Exterior2nd'] = full_df['Exterior2nd'].fillna('None')

for col in ['MSZoning', 'Utilities', 'KitchenQual', 'Functional', 'SaleType']:
    mode_value = full_df[col].mode()[0]
    full_df[col] = full_df[col].fillna(mode_value)


cols = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in cols:
    full_df[col] = full_df[col].fillna("NA")

full_df['Alley'] = full_df['Alley'].fillna('None')


full_df['LotFrontage'] = full_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

full_df["Electrical"] = full_df["Electrical"].fillna("NA")
full_df["FireplaceQu"] = full_df["FireplaceQu"].fillna("NA")

cols2= ["GarageType","GarageYrBlt","GarageFinish", "GarageQual", "GarageCond","PoolQC", "Fence", "MiscFeature" ]
for col in cols2:
    full_df[col] = full_df[col].fillna("NA")


full_df["MSSubClass"] = full_df["MSSubClass"].astype("str")
dummies = pd.get_dummies(full_df['MSSubClass'], prefix='MSSubClass')
df = pd.concat([full_df.drop('MSSubClass', axis=1), dummies], axis=1)

print("\nEksik veriler dolduruldu.\n")

qual_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
full_df['KitchenQualEncoded'] = full_df['KitchenQual'].map(qual_map)

full_df['TotalBath'] = full_df.apply(lambda row: row['FullBath'] + 0.5 * row['HalfBath'] + row['BsmtFullBath'] + 0.5 * row['BsmtHalfBath'], axis=1)
full_df['HasLuxuryKitchen'] = full_df['KitchenQual'].apply(lambda x: 1 if x in ['Ex', 'Gd'] else 0)
full_df['IsMultiKitchen'] = full_df['KitchenAbvGr'].apply(lambda x: 1 if x > 1 else 0)
full_df['BathPerRoom'] = full_df.apply(lambda row: (row['FullBath'] + 0.5 * row['HalfBath'] + row['BsmtFullBath'] + 0.5 * row['BsmtHalfBath']) / row['TotRmsAbvGrd'] if row['TotRmsAbvGrd'] != 0 else 0, axis=1)

full_df["TotalSF"] = full_df["TotalBsmtSF"] + full_df["1stFlrSF"] + full_df["2ndFlrSF"]
full_df["PorchSF"] = full_df["OpenPorchSF"] + full_df["EnclosedPorch"] + full_df["3SsnPorch"] + full_df["ScreenPorch"]
full_df["HasPorch"] = full_df["PorchSF"].apply(lambda x: 1 if x > 0 else 0)

full_df["HouseAge"] = full_df["YrSold"] - full_df["YearBuilt"]
full_df["RemodAge"] = full_df["YrSold"] - full_df["YearRemodAdd"]
full_df["IsRemodeled"] = full_df.apply(lambda row: 1 if row["YearBuilt"] != row["YearRemodAdd"] else 0, axis=1)
full_df["OverallGrade"] = full_df["OverallQual"] * full_df["OverallCond"]
full_df["IsNew"] = full_df.apply(lambda row: 1 if row["YrSold"] == row["YearBuilt"] else 0, axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(full_df, cat_th=10, car_th=14)
full_df_encoded = pd.get_dummies(full_df, columns=cat_cols, drop_first=False)


full_df.head(5)

for col in cat_but_car:
    freq = full_df[col].value_counts() / len(full_df)
    full_df_encoded[col + '_FreqEnc'] = full_df[col].map(freq)

full_df_encoded = full_df_encoded.drop(columns=['Neighborhood', 'Exterior1st', 'Exterior2nd'])


train_processed_df = full_df_encoded[full_df_encoded["is_train_1"] == 1].drop("is_train_1", axis=1)
test_processed_df = full_df_encoded[full_df_encoded["is_train_0"] == 0].drop("is_train_0", axis=1)

train_processed_df.head(5)

train_processed_df["SalePrice"] = train_labels
train_processed_df.astype({col: float for col in train_processed_df.select_dtypes(include='bool').columns})

train_processed_df.head(5)