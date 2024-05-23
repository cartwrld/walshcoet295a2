from pathlib import Path
import random as randy

path = Path('eq_output.geojson')


# opens the file --> writes the text --> closes the file

def get_start_data():
    headers="age,gender,occupation,education_level,marital_status,income,credit_score,loan_status"
    return headers


def get_randy_person():


    return text_to_write


def get_finisher():


    return finisher


output = get_start_data()

for i in range(0, 1000):
    output += get_randy_feature()
output += get_finisher()

path.write_text(output)
