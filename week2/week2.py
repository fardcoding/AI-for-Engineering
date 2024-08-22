import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('data.csv')


data['AT_AP_ratio'] = data['AT'] / data['AP']
data['AT_RH_interaction'] = data['AT'] * data['RH']
data['V_AP_interaction'] = data['V'] * data['AP']
data['Combined_effect'] = data['AT'] * data['V'] * data['AP'] * data['RH']
data['Composite_feature'] = data['AT'] * data['V'] + data['AP'] * data['RH']


target = data['PE']
feature_sets = {
    'Original Features': ['AT', 'V', 'AP', 'RH'],
    'Original + AT_AP_ratio': ['AT', 'V', 'AP', 'RH', 'AT_AP_ratio'],
    'Original + AT_RH_interaction': ['AT', 'V', 'AP', 'RH', 'AT_RH_interaction'],
    'Original + V_AP_interaction': ['AT', 'V', 'AP', 'RH', 'V_AP_interaction'],
    'All Features': ['AT', 'V', 'AP', 'RH', 'AT_AP_ratio', 'AT_RH_interaction', 'V_AP_interaction', 'Combined_effect', 'Composite_feature']
}


results = []


X_train, X_test, y_train, y_test = train_test_split(data.drop(columns='PE'), target, test_size=0.2, random_state=42)


for key, features in feature_sets.items():
    
    model = DecisionTreeRegressor(random_state=42)
    
    model.fit(X_train[features], y_train)
    
    
    y_pred = model.predict(X_test[features])
    
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    
    results.append([key, mae, rmse, r2])


results_df = pd.DataFrame(results, columns=['Feature Set', 'MAE', 'RMSE', 'R^2 Score'])
results_df = results_df.sort_values(by='R^2 Score', ascending=False)


print(results_df)
