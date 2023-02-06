import requests
import pandas as pd
import json

def load_job():
    response = requests.get("http://www.recommendu.kro.kr:30001/services/job_total")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result
    
def load_answer():
    response = requests.get("http://www.recommendu.kro.kr:30001/services/answer_total")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result

def load_document():
    response = requests.get("http://www.recommendu.kro.kr:30001/services/document_total")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result

def load_user():
    response = requests.get("http://www.recommendu.kro.kr:30001/services/accounts/total")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result

def load_answerlog():
    response = requests.get("http://www.recommendu.kro.kr:30001/logs/total/answerlog")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result
    
def load_evallog():
    response = requests.get("http://www.recommendu.kro.kr:30001/logs/total/evallog")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result

def load_recommendlog():
    response = requests.get("http://www.recommendu.kro.kr:30001/logs/total/recommendlog")
    content = json.loads(response.text)
    result = pd.DataFrame(content)
    return result