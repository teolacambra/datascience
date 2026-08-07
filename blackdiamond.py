import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the data from a CSV file
data = pd.read_csv('carabiner.csv')
# display all columns in the DataFrame
print(data.columns.tolist())
# print the first few rows of the DataFrame
print(data.head())
# filter the data to include only the relevant columns
axial = data[['Carabiner type', 'Gate type', 'Axial(kn)']]
print(axial.head())
# filter the data to include only the relevant rows
axial = axial.sort_values(by='Axial(kn)', ascending=True)
axial_wiregate = axial.loc[(axial['Gate type'] == 'Wire')]
axial_screwgate = axial.loc[(axial['Gate type'] == 'Solid')]
axial_wiregate.describe()
axial_screwgate.describe()
print(axial_wiregate.head())
print(axial_screwgate.head())
# create a scatter plot to compare the axial strength of wire gate and solid gate carabiners, with wire and solid having different colors
plt.figure(figsize=(10, 6))
plt.scatter(range(len(axial_wiregate)), axial_wiregate['Axial(kn)'], label='Wire Gate')
plt.scatter(range(len(axial_screwgate)), axial_screwgate['Axial(kn)'], label='Solid Gate')
plt.title('Axial Strength of Wire Gate vs Solid Gate Carabiners')
plt.xlabel('Carabiner Type')
plt.ylabel('Axial Strength (kN)')
plt.legend()
plt.show()
# make regression lines for wire gate and solid gate carabiners
from sklearn.linear_model import LinearRegression
# create a linear regression model for wire gate carabiners
X_wire = np.array(range(len(axial_wiregate))).reshape(-1, 1)
y_wire = axial_wiregate['Axial(kn)'].values
model_wire = LinearRegression()
model_wire.fit(X_wire, y_wire)