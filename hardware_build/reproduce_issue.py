
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Mocking data to avoid dependency on the CSV file existence or path changes, 
# but using the structure we saw in hls_metrics.csv
data = {
    'Run_ID': [f'run_{i}' for i in range(5)],
    'Dataflow': [False, False, True, True, True],
    'P': [1, 2, 1, 4, 8],
    'P_Suffix': [1, 1, 1, 2, 8],
    'Latency_Cycles': [100, 200, 50, 40, 30],
    'BRAM': [3000, 3000, 4000, 4000, 5000]
}
df = pd.DataFrame(data)

plt.figure(figsize=(12, 6))

colors = []
shapes = []
for i in range(len(df)):
    if df['Dataflow'][i]:
        colors.append('blue')
    else:
        colors.append('red')

    if df['P'][i] == df['P_Suffix'][i]:
        shapes.append('o')
    else:
        shapes.append('x')

print("Attempting to scatter with list of colors and unused shapes...")
try:
    plt.scatter(
        df['Latency_Cycles'], 
        df['BRAM'], 
        color=["blue" if d else "red" for d in df['Dataflow']]
    )
    print("Scatter call successful (colors used, shapes ignored).")
except Exception as e:
    print(f"Scatter call failed: {e}")

print("Attempting to add colorbar with label 'Parallelism (P)'...")
try:
    plt.colorbar(label='Parallelism (P)')
    print("Colorbar call successful (but might be empty/wrong).")
except Exception as e:
    print(f"Colorbar call failed: {e}")

# Check if shapes were used? No, they were ignored in the notebook code.
print("Verifying if shapes were used: The code in the notebook does NOT pass 'shapes' to scatter.")

plt.close()
