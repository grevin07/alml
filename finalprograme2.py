import joblib
model=joblib.load("trained.pkl")
windspeed=int(input("enter windspeed"))
bladeangle=float(input("enter blade angle"))
rotorspeed=float(input("enter rotor speed"))
predicted_power=model.predict([[windspeed,bladeangle,rotorspeed]])
print("predicted power output:",predicted_power[0])

