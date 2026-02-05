import numpy as np
import pandas as pd

df = pd.read_csv("Classifying waste_dataset.csv")
df
X = df[["Weight", "Color", "Texture", "Odor"]].copy()
y = df["Type"].copy()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_test, y_test)
