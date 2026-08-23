import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Create a sample DataFrame
data = pd.read_csv('materials_qc_raw.csv')
data.head()
data.columns = data.columns.str.strip()
print(data.columns)

stainless = data['Material'] == 'Stainless 304'
stainless_data = data[stainless]


def simple_plot(ax, data:pd.DataFrame,x:str,y:str,title:str):
    ax.plot(data[x], data[y], 'o')
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)



def multiseries_plot(ax, data:pd.DataFrame,x:str,y:str,series_col:str,title:str):
    series = data[series_col].unique()
    for s in series:
        series_data = data[data[series_col] == s]
        ax.plot(series_data[x], series_data[y], '.', label=s)
    ax.set_xlabel(f'{x}')
    ax.set_ylabel(f'{y}')
    ax.set_title(title)
    ax.legend(series)
    ax.grid(True)
    


fig, ax = plt.subplots(1,2, figsize=(20,6))
# put the function calls in the subplots

simple_plot(ax[0], stainless_data,"Yield Strength (MPa)","Elongation (%)","Stainless 304 Yield Strength vs Elongation")
multiseries_plot(ax[1], data,"Yield Strength (MPa)","Elongation (%)","Material","Yield Strength vs Elongation for Different Materials")
plt.show()