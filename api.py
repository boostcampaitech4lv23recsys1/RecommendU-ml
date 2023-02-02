import pymysql
import ast
import pandas as pd


def get_api():
    conn = pymysql.connect(
        user='root', 
        passwd='recsys6', 
        host='34.64.237.2', 
        db='recommendu', 
        charset='utf8'
    )    
    
    return conn

#### service data ####
def get_joblargs_api(conn):
    query = 'SELECT * FROM services_joblarge'
    joblarge = pd.read_sql_query(query, conn)

    return joblarge

def get_jobsmall_api(conn):
    query = 'SELECT * FROM services_jobsmall'
    jobsmall = pd.read_sql_query(query, conn)

    return jobsmall

def get_majorsmall_api(conn):
    query = 'SELECT * FROM services_majorsmall'
    majorsmall = pd.read_sql_query(query, conn)

    return majorsmall

def get_majorsmall_api(conn):
    query = 'SELECT * FROM services_company'
    company = pd.read_sql_query(query, conn)

    return company

def get_majorsmall_api(conn):
    query = 'SELECT * FROM services_questiontype'
    question = pd.read_sql_query(query, conn)

    return question

def get_majorsmall_api(conn):
    query = 'SELECT * FROM services_recommendtype'
    rectype = pd.read_sql_query(query, conn)

    return rectype

def get_document_api(conn):
    query = 'SELECT * FROM services_document'
    documents = pd.read_sql_query(query, conn)

    return documents

def get_answer_question_types_api(conn):
    query = 'SELECT * FROM services_answer_question_types'
    answer_question_types = pd.read_sql_query(query, conn)

    return answer_question_types

def get_services_answer_api(conn):
    query = 'SELECT * FROM services_answer'
    answers = pd.read_sql_query(query, conn)

    return answers

#### user data ####

def get_user_data_api(conn):
    query = 'SELECT * FROM accounts_user'
    user_data = pd.read_sql_query(query, conn)

    return user_data

### log data ###
def get_answerlogs_api(conn):
    query = 'SELECT * FROM logs_answerlog'
    answerlogs = pd.read_sql_query(query, conn)

    return answerlogs

def get_reclogs_api(conn):
    query = 'SELECT * FROM logs_recommendlog'
    reclogs = pd.read_sql_query(query, conn) 

    return reclogs   
