import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import joblib

data=pd.read_csv("data2.csv")
x=data[["wind_speed","blade_angle","rotor_speed"]]
y=data["power_output"]

model=LinearRegression()
model.fit(x,y)
print("coefficient:",model.coef_)
print("intercept:",model.intercept_)

joblib.dump(model,"trained.pkl")
print("training completed and model has been saved as trained.pkl")
