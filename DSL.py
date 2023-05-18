import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

url = 'https://api.gbif.org/v1/occurrence/search'

params = {
    'scientificName': 'Lantana camara L.',
    'hasCoordinate': 'true',
    'country': 'IN',
    'limit': 100
}
response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()['results']
else:
    print(f"Error fetching data: {response.status_code} - {response.text}")

df = pd.DataFrame(data)

df = df[['year', 'decimalLatitude', 'decimalLongitude']]
df = df.dropna() 

df['range'] = ((df['decimalLatitude'].max() - df['decimalLatitude'].min()) + 
(df['decimalLongitude'].max() - df['decimalLongitude'].min())) / 2

df = df[['year', 'range']]

df['category'] = [0] * len(df)

X = df[['year', 'range']]
y = df['category'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean squared error: {mse:.2f}")
print(f"R-squared value: {r2:.2f}")

plt.scatter(X_test['year'], y_test, color='pink') 
plt.plot(X_test['year'], y_pred, color='purple', linewidth=3) 
plt.xlabel('Year')
plt.ylabel('Conservation Status')
plt.show()

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")