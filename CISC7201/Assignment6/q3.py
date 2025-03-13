import pandas as pd

def input_data():
    students_data = []
    scores_data = []
    
    # Input students data
    while True:
        student_id = input().strip()
        if student_id == 'done':
            break
        student_name = input().strip()
        students_data.append({'id': int(student_id.strip()), 'name': student_name.strip()})
    
    # Input scores data
    while True:
        score_id = input().strip()
        if score_id == 'done':
            break
        student_score = input().strip()
        scores_data.append({'id': int(score_id.strip()), 'score': float(student_score.strip())})
    
    students = pd.DataFrame(students_data)
    scores = pd.DataFrame(scores_data)
    
    return students, scores

def merge_tables(students, scores):
    merged = pd.merge(students, scores, on='id', how='left')
    merged['score'].fillna(0, inplace=True)
    return merged

students, scores = input_data()
for index, row in merge_tables(students, scores).iterrows():
    print(f"{row['name']}: {row['score']}")