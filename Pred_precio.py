import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

airbnb_df = pd.read_excel('AirbnbRD.xlsx', sheet_name='AirbnbRD')

cols_limpiar = ['TIPO_ALOJAMIENTO', 'ESPACIO_ALOJAMIENTO', 'UBICACION', 'PRECIO_USD',
                'MAX_HUESPEDES', 'CANTIDAD_HABITACIONES', 'CANTIDAD_CAMAS', 'CANTIDAD_BAÑOS']

cols_categoricas = ['TIPO_ALOJAMIENTO', 'ESPACIO_ALOJAMIENTO', 'UBICACION']
cols_numericas = ['MAX_HUESPEDES', 'CANTIDAD_HABITACIONES', 'CANTIDAD_CAMAS', 'CANTIDAD_BAÑOS']

#1
cols_mask = ~airbnb_df[cols_limpiar].isin(['No encontrado']).any(axis=1)
airbnb_df = airbnb_df[cols_mask]

#2
for col in cols_numericas:
    airbnb_df[col] = pd.to_numeric(airbnb_df[col], errors='coerce')

#3
airbnb_df = airbnb_df.dropna(subset=cols_numericas + ['PRECIO_USD'])

#4
airbnb_encoded = pd.get_dummies(airbnb_df, columns=cols_categoricas, drop_first=True)

#5
cols_categoricas_dummy = [col for col in airbnb_df
                          if col.startswith(('TIPO_ALOJAMIENTO_', 'ESPACIO_ALOJAMIENTO_', 'UBICACION_'))]

cols_X = cols_numericas + cols_categoricas_dummy

print(airbnb_df.head())

y = airbnb_encoded['PRECIO_USD']
X = airbnb_encoded[cols_X]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,random_state=42)

modelo_airbnb = RandomForestRegressor(
      n_estimators=10,
    max_depth=5,
    min_samples_leaf=5,
    random_state=42,
)

modelo_airbnb.fit(X_train,y_train)

"""
print("Prediccion de los 5 primeros alojamientos:")
print(X_test.head())
print("El precio por noche real es:")
print(y_test.head())
print("El precio predicho es:")
print(modelo_airbnb.predict(X_test.head()))
"""

#Implementando el modelo xgboost

modelo_xgb = XGBRegressor(
  n_estimators=6,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)

modelo_xgb.fit(X_train, y_train)

y_pred = modelo_xgb.predict(X_test)
y_pred_Rfr = modelo_airbnb.predict(X_test)

print("Prediccion de los 5 primeros alojamientos:")
print(X_test.head())
print("El precio por noche real es:")
print(y_test.head())
print("El precio predicho es:")
print(y_pred[:5])

#Validacion
xgb_mae = mean_absolute_error(y_test, y_pred)
rfr_mae = mean_absolute_error(y_test,y_pred_Rfr)

precio_promedio = y.mean()

porcentaje_error_xgb = (xgb_mae / precio_promedio) * 100
porcentaje_error_rfr = (rfr_mae / precio_promedio) * 100


print("MAE de XGBoost:")
print(xgb_mae)
print("Procentaje de MAE:")
print(porcentaje_error_xgb)
print(porcentaje_error_rfr)
