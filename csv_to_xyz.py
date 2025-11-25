import pandas as pd
df = pd.read_csv('positions.csv')

with open('trajectory.xyz', 'w') as f:
    for step in df['step'].unique():
        frame_data = df[df['step'] == step]
        f.write(f"{len(frame_data)}\n")
        f.write(f"Step: {step}\n")
        for _, row in frame_data.iterrows():
            f.write(f"{int(row['body'])} {row['x']} {row['y']} {row['z']}\n")