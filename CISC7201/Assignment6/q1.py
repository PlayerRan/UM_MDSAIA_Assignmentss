import pandas as pd

def input_data():
    data = []
    while True:
        index1 = input().strip()
        if index1 == 'done':
            break
        index2 = input().strip()
        value = float(input().strip())
        data.append((index1, index2, value))
    
    index = pd.MultiIndex.from_tuples([(d[0], d[1]) for d in data], names=['index1', 'index2'])
    df = pd.DataFrame([d[2] for d in data], index=index, columns=['value'])
    return df

def output_data(df):
    for (index1, index2), row in df.iterrows():
        print(f"{index1}, {index2}: {row['value']}")

df = input_data()
output_data(df)