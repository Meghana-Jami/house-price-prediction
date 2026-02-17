import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("C:/Users/nalab/OneDrive/Documents/house_price_dataset.csv")

# Remove unwanted column if exists
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Features and target
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms']]
y = data['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Take user input
print("\nEnter House Details:")

sqft = float(input("Square Feet: "))
bed = int(input("Number of Bedrooms: "))
bath = int(input("Number of Bathrooms: "))

# Create DataFrame (Fixes Warning)
new_house = pd.DataFrame({
    'SquareFeet': [sqft],
    'Bedrooms': [bed],
    'Bathrooms': [bath]
})

# Prediction
price = model.predict(new_house)[0]

# Display result
print("\nPredicted House Price: ₹", round(price, 2))
