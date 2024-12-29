import math
from vpython import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

'''
Lines 17 through 83 were copied from https://trinket.io/glowscript/7069d44ff5 from Dot Physics on Youtube. 
'''

data = []

# Graph setup
g1 = graph(title="Double Pendulum", xtitle="t [s]", ytitle="E [J]")
fE = gcurve(color=color.blue, label="Total E")
fU = gcurve(color=color.green, label="U")

# Constants
g = 9.8
M1 = 0.1
M2 = 0.2
R1 = 0.1
R2 = 0.1

# Initial conditions
theta1 = 180 * math.pi / 180
theta1dot = 0
theta2 = 175 * math.pi / 180
theta2dot = 0

# VPython objects
pivot = sphere(pos=vector(0, R1, 0), radius=R1 / 20)
m1 = sphere(pos=pivot.pos + vector(R1 * math.sin(theta1), -R1 * math.cos(theta1), 0),
            radius=R1 / 10, color=color.yellow)
m2 = sphere(pos=m1.pos + vector(R2 * math.sin(theta2), -R2 * math.cos(theta2), 0),
            radius=R1 / 10, color=color.yellow, make_trail=True)
s1 = cylinder(pos=pivot.pos, axis=m1.pos - pivot.pos, radius=R1 / 30)
s2 = cylinder(pos=m1.pos, axis=m2.pos - m1.pos, radius=R1 / 30)

# Time setup
t = 0
dt = 0.0001

# Simulation loop
while t < 4:
    rate(50000)

    # Physics calculations
    a = -(M1 + M2) * g * R1 * math.sin(theta1) - M2 * R1 * R2 * theta2dot ** 2 * math.sin(theta1 - theta2)
    b = (M1 + M2) * R1 ** 2
    c = M2 * R1 * R2 * math.cos(theta1 - theta2)
    f = -M2 * g * R2 * math.sin(theta2) + M2 * R1 * R2 * theta1dot ** 2 * math.sin(theta1 - theta2)
    k = M2 * R2 ** 2
    w = M2 * R1 * R2 * math.cos(theta1 - theta2)

    # Corrected equations for accelerations
    theta2ddot = (f - a * w / b) / (k - c * w / b)
    theta1ddot = (a - c * theta2ddot) / b

    # Update velocities and angles
    theta2dot += theta2ddot * dt
    theta1dot += theta1ddot * dt
    theta1 += theta1dot * dt
    theta2 += theta2dot * dt
    t += dt

    # Update positions
    m1.pos = pivot.pos + vector(R1 * math.sin(theta1), -R1 * math.cos(theta1), 0)
    m2.pos = m1.pos + vector(R2 * math.sin(theta2), -R2 * math.cos(theta2), 0)
    s1.axis = m1.pos - pivot.pos
    s2.pos = m1.pos
    s2.axis = m2.pos - m1.pos

    # Optional: Plot energy (Total energy or other metrics can be added)
    T = 0.5 * M1 * R1 ** 2 * theta1dot ** 2 + 0.5 * M2 * (R2 ** 2 * theta2dot ** 2 +
                                                          2 * R1 * R2 * theta1dot * theta2dot * math.cos(
                theta1 - theta2) + R1 ** 2 * theta1dot ** 2)
    U = -(M1 + M2) * g * R1 * math.cos(theta1) - M2 * g * R2 * math.cos(theta2)
    fE.plot(t, T + U)
    fU.plot(t, U)

    data.append({
        "t": t,
        "theta1": theta1,
        "theta2": theta2,
        "theta1dot": theta1dot,
        "theta2dot": theta2dot,
        "theta1_init": 180 * math.pi / 180,  # Example: Initial theta1
        "theta2_init": 175 * math.pi / 180  # Example: Initial theta2

    })

df = pd.DataFrame(data)
df.to_csv("double_pendulum.csv", index=False)

df = pd.read_csv("double_pendulum.csv")

X = df[["theta1_init", "theta2_init", "t"]]
y = df[["theta1", "theta2", "theta1dot", "theta2dot"]]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Predict for new initial conditions
new_conditions = [[40 * math.pi / 180, 70 * math.pi / 180, 1.5]]  # Example: theta1_init, theta2_init, t
predicted_state = model.predict(new_conditions)
print(f"Predicted State: {predicted_state}")


# Plot simulation vs. ML predictions
plt.plot(df["t"], df["theta1"], label="Simulation", color="blue")
plt.plot(df["t"], model.predict(X)[:, 0], label="ML Prediction", color="red", linestyle="--")
plt.xlabel("Time (s)")
plt.ylabel("Theta1 (radians)")
plt.legend()
plt.show()

