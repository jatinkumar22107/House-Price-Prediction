import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Sample house data
data = {
    "area": [800, 1000, 1200, 1500, 1800, 2000],
    "bedrooms": [1, 2, 2, 3, 3, 4],
    "bathrooms": [1, 1, 2, 2, 3, 3],
    "price": [2500000, 3500000, 4500000, 6000000, 7500000, 9000000]
}

df = pd.DataFrame(data)

X = df[["area", "bedrooms", "bathrooms"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# Save model
with open("house_price_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ house_price_model.pkl created successfully")
