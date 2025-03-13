import pandas as pd

def process_scores(df: pd.DataFrame):
    """
    Process the student scores DataFrame, calculate the average score, and return the result as a DataFrame.
    :param df: Input DataFrame containing columns Name, Subject, Score
    :return: DataFrame sorted by average scores in descending order
    """
    # Group by 'Name' and calculate the mean of 'Score'
    avg_scores = df.groupby('Name')['Score'].mean().reset_index()
    # Rename the 'Score' column to 'AverageScore'
    avg_scores.rename(columns={'Score': 'AverageScore'}, inplace=True)
    # Sort by 'AverageScore' in descending order
    sorted_avg_scores = avg_scores.sort_values(by='AverageScore', ascending=False)
    
    return sorted_avg_scores

def input_table():
    """
    Read tabular data from standard input. Each row contains Name, Subject, and Score, 
    separated by spaces. Input ends with an empty line.
    Data format:
    Alice Math 85
    Bob Math 90
    Alice English 95
    ...
    """
    data = []
    while True:
        line = input().strip()
        if line == "":
            break
        parts = line.split()
        if len(parts) != 3:
            raise ValueError("Each row must contain exactly three fields: Name Subject Score")
        name, subject, score = parts[0], parts[1], int(parts[2])
        data.append([name, subject, score])
    
    # Convert the list of data into a Pandas DataFrame
    df = pd.DataFrame(data, columns=['Name', 'Subject', 'Score'])
    return df

def output_table(df: pd.DataFrame):
    """
    Print the contents of the DataFrame line by line to standard output.
    """
    for _, row in df.iterrows():
        print(f"{row['Name']} {row['AverageScore']:.2f}")

input_data = input_table()
result = process_scores(input_data)
output_table(result)