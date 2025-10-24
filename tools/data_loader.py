import json
import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from datasets import load_dataset


def remove_calc_annotations(text):
    return re.sub(r"<<.*?>>", "", text)


def read_gsm8k(train = False):
    dataset = load_dataset("gsm8k", 'main')
    train_input = dataset.data['train']['question'].to_pylist()
    train_output = dataset.data['train']['answer'].to_pylist()
    train_output = [remove_calc_annotations(i) for i in train_output]
    train_output = [i.replace("#### ", "") for i in train_output]

    test_input = dataset.data['test']['question'].to_pylist()
    test_output = dataset.data['test']['answer'].to_pylist()
    test_output = [remove_calc_annotations(i) for i in test_output]
    test_output = [i.replace("#### ", "") for i in test_output]
    

    if train == True:
        return train_input, train_output
    else:
        return test_input, test_output

def read_r2ata():
    df = pd.read_csv("datasets/r2ata.csv")
    df = df[df['dataset'] == 'gsm8k']
    df['Edited'] = df['Edited'].apply(lambda x: x.replace("Question: ", "").replace("\nAnswer: Let's think step by step.", ""))
    # df['Target'] = "The answer is " + df['Target']
    df['Target'] = df['Target'].apply(lambda x: x.strip())

    r2ata_input = df['Edited'].tolist()
    r2ata_output = df['Target'].tolist()

    return r2ata_input, r2ata_output

def read_svamp():
    df = pd.read_json('datasets/SVAMP.json')
    df['problem'] = df['Body'] + " " + df['Question']
    # df['cot_output'] = "The answer is " + df['Answer'].apply(str)
    df['Answer'] = df['Answer'].apply(str)

    svamp_input = df['problem'].tolist()
    svamp_output = df['Answer'].tolist()

    return svamp_input, svamp_output


def read_asdiv():
    # Parse the XML file
    tree = ET.parse('datasets/ASDiv.xml')  # Replace 'file.xml' with the path to your XML file
    root = tree.getroot()

    # Extract data
    data = []
    for problem in root.find('ProblemSet'):
        problem_id = problem.attrib.get('ID')
        grade = problem.attrib.get('Grade')
        source = problem.attrib.get('Source')
        body = problem.find('Body').text
        question = problem.find('Question').text
        solution_type = problem.find('Solution-Type').text
        answer = problem.find('Answer').text
        formula = problem.find('Formula').text

        data.append(
            {
                'Problem ID': problem_id,
                'Grade': grade,
                'Source': source,
                'Body': body,
                'Question': question,
                'Solution Type': solution_type,
                'Answer': answer,
                'Formula': formula,
            }
        )

    # Convert to pandas DataFrame
    df = pd.DataFrame(data)
    df = df[df['Answer'].apply(lambda x: str(x).find(';') == -1)]  # remove problems asking multiple answers

    df['problem'] = df['Body'] + " " + df['Question']
    # df['Answer'] = "The answer is " + df['Answer'].apply(lambda x: str(x).split()[0])

    asdiv_input = df['problem'].tolist()

    asdiv_output = df['Answer'].tolist()

    return asdiv_input, asdiv_output



def read_math_500(train = False):
    dataset = load_dataset("HuggingFaceH4/MATH-500")
    train_input = dataset.data['train']['question'].to_pylist()
    train_output = dataset.data['train']['answer'].to_pylist()
    train_output = [remove_calc_annotations(i) for i in train_output]
    train_output = [i.replace("#### ", "The answer is ") for i in train_output]

    test_input = dataset.data['test']['question'].to_pylist()
    test_output = dataset.data['test']['answer'].to_pylist()
    test_output = [remove_calc_annotations(i) for i in test_output]
    test_output = [i.replace("#### ", "The answer is ") for i in test_output]
    

    if train == True:
        return train_input, train_output
    else:
        return test_input, test_output


def read_data(task, train=False):
    if task == 'gsm8k':
        return read_gsm8k(train)
    elif task == 'r2ata':
        return read_r2ata()
    elif task == 'svamp':
        return read_svamp()
    elif task == 'asdiv':
        return read_asdiv()
    