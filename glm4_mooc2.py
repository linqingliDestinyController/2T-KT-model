import os
import argparse
from mylogger1 import Logger
import dataclasses
from typing import List, Optional, Literal,Tuple, Union
from utils import aggregate_data
import random
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
import time
import pandas as pd
from GLM2 import glm_chat
from Advice import advice
moderate_exercise_info_type = {'exercise_id': 'str', 'exercise_desc': 'str', 'skill_ids': 'object', 'skill_desc': 'object'}
knowledge_graph_info = ''
def load_user_data(data_path: str, is_shuffle: bool, train_split: float, create_train_test = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # load user_logs from recordings.csv, each line is a user's exercise record with the format above
        
        recordings_path = os.path.join(data_path, "recordings.jsonl")
        recordings_df = pd.read_json(recordings_path, lines=True)

        # check if the data is already splitted
        if os.path.exists(os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")) and os.path.exists(os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")):
            logger.write(f"Train and test data already exist, loading from file")
            create_train_test = False

        if not create_train_test:
            # read from existing train and test data
            train_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")
            test_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")
            train_data = pd.read_json(train_data_path, lines=True)
            test_data = pd.read_json(test_data_path, lines=True)
            train_data = train_data.astype({"student_id": str})
            test_data = test_data.astype({"student_id": str})

            return train_data, test_data

        else:  
            # create train and test data from recordings.jsonl
            logger.write(f"Creating train and test data from recordings.jsonl")
            # initialize the train and test data with recodings_df.columns
            train_data = pd.DataFrame(columns=recordings_df.columns)
            test_data = pd.DataFrame(columns=recordings_df.columns)
            for i, row in tqdm.tqdm(recordings_df.iterrows()):
                student_id = str(row["student_id"])
                exercises_logs = np.array(row["exercises_logs"])
                is_corrects = np.array(row["is_corrects"])
                # check if the student has at least two exercise logs to do train/test split
                if len(exercises_logs) < 2:
                    continue
                # randomly split the exercise logs into train and test set
                # is_shuffle means whether to shuffle the data before split
                if True :
                    # create the idx of logs to randomly sample from the student's exercise logs
                    idx = list(range(len(exercises_logs)))
                    train_sample_size = int(len(idx)*train_split) if len(idx)*train_split > 1 else 1
                    train_idx = random.sample(idx, train_sample_size)
                    test_idx = [i for i in idx if i not in train_idx]
                else:
                    train_sample_size = int(len(exercises_logs)*train_split) if len(exercises_logs)*train_split > 1 else 1
                    train_idx = list(range(train_sample_size))
                    test_idx = list(range(train_sample_size, len(exercises_logs)))
                # create the train and test data for the student
                train_data_row = {"student_id": student_id, "exercises_logs": exercises_logs[train_idx], "is_corrects": is_corrects[train_idx]}
                test_data_row = {"student_id": student_id, "exercises_logs": exercises_logs[test_idx], "is_corrects": is_corrects[test_idx]}
                train_data = train_data._append(train_data_row, ignore_index=True)
                test_data = test_data._append(test_data_row, ignore_index=True)
                # save the train and test data to file
                train_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_train_data.jsonl")
                test_data_path = os.path.join(data_path, f"{is_shuffle}_{train_split}_test_data.jsonl")
                train_data.to_json(train_data_path, lines=True, orient='records')
                test_data.to_json(test_data_path, lines=True, orient='records')

        logger.write(f"Split Total {len(recordings_df)} recordings, ratio {train_split}, shuffle {is_shuffle}, {len(train_data)} train data, {len(test_data)} test data")
        return train_data, test_data
    
def load_moderate_data(data_path: str) -> dict:
        # moderate: question contents and skills
        mo_e_info_path = os.path.join(data_path, "moderate_exercise_info.jsonl")
        mo_e_info_df = pd.read_json(mo_e_info_path, lines=True, dtype=moderate_exercise_info_type)
        # format: exercise_id, exercise_desc, list of skill_ids in exercise_id, list of skill_desc of skill_ids
        logger.write("Loaded moderate_exercise_info data")
        return {"exercise_info": mo_e_info_df}

def get_args():
    parser = argparse.ArgumentParser()
    # LLM model name
    parser.add_argument('--model_type', type=str, default='llm', help='model type llm or ktm')
    parser.add_argument('--model_name', type=str, default='glm-4', help='model name')
    parser.add_argument('--data_path', type=str, default='./datasets', help='data path')
    # data_mode: sparse, moderate, rich
    parser.add_argument('--data_mode', type=str, default='moderate', help='data mode: onehot, sparse, moderate, rich')
    # dataset name
    parser.add_argument('--dataset_name', type=str, default='MOOCRadar', help='dataset name')
    parser.add_argument('--log_path', type=str, default='./logs', help='log path')
    # train_split
    parser.add_argument('--train_split', type=float, default=0.8, help='train split')
    # parser.add_argument('--is_shuffle', type=bool, default=True, help='shuffle data when splitting')
    parser.add_argument('--is_shuffle', action='store_true', help='shuffle data when splitting')
    # test number
    parser.add_argument('--test_num', type=int, default=20, help='test number')
    # random seed
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')

    # llm fewshot settings
    parser.add_argument('--fewshot_num', type=int, default=8, help='fewshot num, 0 means zero-shot')
    parser.add_argument('--fewshot_strategy', type=str, default='random', help='fewshot strategy, random/first/last/RAG')
    parser.add_argument('--eval_strategy', type=str, default='simple', help='eval strategy, simple/analysis/self_correct')
    args = parser.parse_known_args()[0]
    return args

args = get_args()
my_logger = Logger(args)
logger = my_logger

data_path = os.path.join(args.data_path, args.data_mode, args.dataset_name)
if not os.path.exists(data_path):
    print(data_path)
    raise ValueError("data path not exist, check path, mode, or dataset name")
    
if args.model_type == 'llm':
    train_data, test_data = load_user_data(data_path=data_path, train_split=args.train_split, is_shuffle=args.is_shuffle)
    # extra_datas is a list of extra data, each one is pd.DataFrame
    if args.data_mode == 'onehot':
        extra_datas = load_onehot_data(data_path=data_path)
    elif args.data_mode =='sparse':
        extra_datas = load_sparse_data(data_path=data_path)
    elif args.data_mode =='moderate':
        extra_datas = load_moderate_data(data_path=data_path)
    elif args.data_mode == 'rich':
        extra_datas = load_rich_data()
    else:
        raise ValueError("data mode not in ['onehot','sparse','moderate', 'rich']")
elif args.model_type == 'ktm':
    # TODO: add ktm dataloader
    pass
else:
    raise ValueError("model type not in ['llm', 'ktm']")

model_name="glm-4" 
train_data=train_data
test_data=test_data
extra_datas=extra_datas
logger=my_logger
data_mode=args.data_mode
fewshots_num=args.fewshot_num
fewshots_strategy=args.fewshot_strategy
eval_strategy=args.eval_strategy
test_num=args.test_num
random_seed=args.random_seed

MessageRole = Literal["system", "user", "assistant"]
@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str

def create_user_data(student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info) -> str:
        return generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info)

def generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info) -> str:
    # create prediction data for a test exercise to send to llm
    # data_mode: "onehot", "sparse", "moderate", "rich"
    user_data = ''
    
    # add the related knowledge graph
    # user_data += '<Knowledge graph information>\n'
    # user_data += knowledge_graph_info
    # user_data += '<END knowledge graph information>\n'
    
     # # add the advice
    # user_data += '<Advice>\n'
    # user_data += advice
    # user_data += '<END Advice>\n'
    
    if data_mode == "onehot":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise_id: {test_exercise_info['exercise_id']}, knowledge concepts: {test_exercise_info['skill_ids']}\n"
    elif data_mode == "sparse":
        user_data += '<Exercise to Predict>\n'
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "moderate":
        user_data += '<Exercise to Predict>\n'
        user_data += f"exercise content: {test_exercise_info['exercise_desc']}\n"
        user_data += f"knowledge concepts: {test_exercise_info['skill_desc']}\n"
    elif data_mode == "rich":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}, when creating user data")
   
    user_data += '<Output Predicted is_correct>\n'
    return user_data

def generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot=True) -> List[Message]:
    if len(fewshots) > 0: # few-shot
        messages = [
            Message(
                role="system",
                content=prompts['sys_instr'],
            )
        ]
        if use_selected_fewshot:
            fs_content = '\n'.join(fewshots)
            messages.append(
                Message(
                    role="user",
                    content=fs_content + '\n' + user_data + '\n' + prompts['predict_instr'],
                )
            )
        else:
            messages.append(
                Message(
                    role="user",
                    content=prompts['fewshot_u_ex'] + "\n" + prompts['predict_instr'] + '\n',
                )
            )
            messages.append(
                Message(
                    role="assistant",
                    content=prompts['fewshot_s_ex']
                )
            )
            messages.append(
                Message(
                    role="user",
                    content=user_data + "\n" + prompts['predict_instr'],
                )
            )
        # print_messages(messages[0].content, messages[-1].content)
    else: # zero-shot
        messages = [
            Message(
                role="system",
                content=prompts['sys_instr'],
            ),
            Message(
                role="user",
                content=user_data + "\n" + prompts['predict_instr'],
            ),
        ]
        # print_messages(messages[0].content, messages[-1].content)
    return messages

def create_prediction_messages(fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> List[Message]:
        return generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)

def generate_prediction(fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
        prediction_messages = create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
        # print(f"The prediction message is {prediction_messages}")
        return glm_chat(model_name, prediction_messages, num_comps=1), prediction_messages
    
def check_response_format(response: str) -> str:
    # response if the response is '1' or '0'
    if response == '1' or response == '0':
        return response
    else:
        raise ValueError(f"Invalid response format: {response}, should be '1' or '0'")

def evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode, knowledge_graph_info):
    eval_result = {'student_id': student_info['student_id'], 
            'pre_exe_id': test_exercise_info['exercise_id']
            }
    if eval_strategy =='simple':
        # generate fewshots and prompts to predict
        #就是加了一个exercise to predict
        user_data = create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode, knowledge_graph_info = knowledge_graph_info )
        
        prediction_response, prediction_message = generate_prediction(fewshots, user_data, prompts)
        
        try:
            eval_result['prediction'] = check_response_format(prediction_response)
        except ValueError as e:
            # if the repsonse format is not correct, generate again, if again, then randomly choose 0 or 1
            response_tmp,prediction_message = generate_prediction(fewshots, user_data, prompts)
            try:
                eval_result['prediction'] = check_response_format(response_tmp)
            except ValueError as e:
                logger.write(f"Error in prediction response format: at student {student_info['student_id']}, exercise {test_exercise_info['exercise_id']}")
                logger.write(f"response: {prediction_response}")
                eval_result['prediction'] = '0' if random.random() < 0.5 else '1'
        # generate explaination
        # delete previous "<Exercise to Predict>"
        # add new prediction to fewshots
        user_data = user_data.replace("<Exercise to Predict>", "")
        user_data += ("is_correct: " + eval_result['prediction'] + "\n")
        fewshots.append(user_data)
       
        eval_result['explaination'] = generate_explaination(fewshots, eval_result['prediction'], prompts)
        

    elif eval_strategy == 'analysis':
        raise NotImplementedError
    elif eval_strategy =='self_correct':
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid eval_strategy: {eval_strategy}")
    
    # self.logger.write(f"Evaluation result: {eval_result}")
    return eval_result, fewshots

import dataclasses

# GLM sparse prompts
# GLM_SPARSE_SYSTEM_INSTRUCTION = '''
# You are a knowledge tracking model that predicts whether a student will be able to get a new question right when he encounters it based on his history of doing the question, and the knowledge concepts in the question.
# Give the format of the history of exercising as follows: each exercise information begins with <Exercise exercise_id> and ends with <END Exercise exercise_id>, the knowledge graph information begin with <Knowledge graph information> and ends with <END knowledge graph information>, and the advice begin with <Advice> and ends with <END Advice>

# <Exercise exercise_id>
# knowledge concepts: [knowledge concepts descriptions in the exercise],
# is_correct: 0 or 1
# <END Exercise exercise_id>

# <Knowledge graph information>
# knowledge graph info: [the student subgraph information]
# <END Knowledge graph information>

# <Advice>
# The advice is here
# <END Advice>


# Then, a question and corresponding knowledge points will be given, and you need to predict whether the student will be able to answer these questions correctly or not.
# The output should only be 0 or 1, 1 for correct and 0 for incorrect. no other explanation is needed.
# '''
# GLM_SPARSE_FEWSHOT_USER_EXAMPLE = ''''''# only used when not generated by llms
# GLM_SPARSE_PREDICT_INSTRUCTION = '''
# Base on the system instructions, few shots, advice, knowledge graph information, and other information above. You can choose to accept the advice or ignore it. Only respond with only one 0 or 1 to predict the student can answer <Exercise to Predict> correctly or not.
# '''
GLM_SPARSE_ANALYSIS_INSTRUCTION = '''
Analyze the student knowledge state by the exercise logs and knowledge graph information above.
'''
GLM_SPARSE_EXPLAINATION_INSTRUCTION = '''
Now, you are a teacher analysing the student's performance in the previous question.
The student's ability to get this question right depends on many factors, so please analyze why the student performed as shown above in the previous question in the following ways, taking into account the given record of doing the question as well as the historical analysis.
1. Find out the knowledge points involved in the new question, it will follow the format of "knowledge concepts: ['kc1', 'kc2'...]"

The new exercise contains <knowledge points 1>, <knowledge points 2, ...>

2. analyze the link between the question and the topic in the student's record of work: is the question new, and does the knowledge point in this question exist in previous questions? with the following format:
Similar to question <q1,q2... > or It's a new question, there is <some kind of> connection between the previous knowledge points and questions.

3. For Student's mastery of knowledge, update the knowledge state based on the current question and the previous exercises with the following format:

Student's Knowledge state:
<previous knowledge points 1, good/fair/bad>, <previous knowledge points 2, good/fair/bad>, <previous knowledge points 3, good/fair/bad> ...
<knowledge points 1 in this exercise, good/fair/bad>, <knowledge points 2 in this exercise, good/fair/bad> ...

4. whether the student mastered the knowledge points involved in the question, whether there is carelessness and other reasons to get the question wrong? with the following format:
The student gets it <right, wrong>, <almost impossible, possible, likely> because of <guessing, mastery> / <carelessness, incorrect mastery>.

Explain the result, no additional warnings or PREDICTION needed.
'''
# GLM_SPARSE_SELF_REFLECTION_INSTRUCTION = ''''''
# GPT_SPARSE_SYSTEM_INSTRUCTION = f'{GLM_SPARSE_SYSTEM_INSTRUCTION}'
# GPT_SPARSE_FEWSHOT_USER_EXAMPLE = f'{GLM_SPARSE_FEWSHOT_USER_EXAMPLE}'
# GPT_SPARSE_PREDICT_INSTRUCTION = '''
# Base on the system instructions, fewshots and information above, predict the student can answer the last <Exercise to Predict> correctly or not. Only respond with one 0 or 1.
# '''
# GPT_SPARSE_ANALYSIS_INSTRUCTION = f'{GLM_SPARSE_ANALYSIS_INSTRUCTION}'
# GPT_SPARSE_EXPLAINATION_INSTRUCTION = f'{GLM_SPARSE_EXPLAINATION_INSTRUCTION}'
# GPT_SPARSE_SELF_REFLECTION_INSTRUCTION = f'{GLM_SPARSE_SELF_REFLECTION_INSTRUCTION}'
# FEW_SHOT_RESPONSE_EXMAPLE = '0'

# model_sp_sys_instr = {
#     'glm': GLM_SPARSE_SYSTEM_INSTRUCTION,
#     'gpt': GPT_SPARSE_SYSTEM_INSTRUCTION
# }
# model_sp_fewshot_u_ex = {
#     'glm': GLM_SPARSE_FEWSHOT_USER_EXAMPLE,
#     'gpt': GPT_SPARSE_FEWSHOT_USER_EXAMPLE
# }
# model_sp_fewshot_s_ex = {
#     'glm': FEW_SHOT_RESPONSE_EXMAPLE,
#     'gpt': FEW_SHOT_RESPONSE_EXMAPLE
# }
# model_sp_predict_instr = {
#     'glm': GLM_SPARSE_PREDICT_INSTRUCTION,
#     'gpt': GPT_SPARSE_PREDICT_INSTRUCTION
# }
# model_sp_analysis_instr = {
#     'glm': GLM_SPARSE_ANALYSIS_INSTRUCTION,
#     'gpt': GPT_SPARSE_ANALYSIS_INSTRUCTION
# }
# model_sp_explain_instr = {
#     'glm': GLM_SPARSE_EXPLAINATION_INSTRUCTION,
#     'gpt': GPT_SPARSE_EXPLAINATION_INSTRUCTION
# }
# model_sp_self_refl_instr = {
#     'glm': GLM_SPARSE_SELF_REFLECTION_INSTRUCTION,
#     'gpt': GPT_SPARSE_SELF_REFLECTION_INSTRUCTION
# }

GLM_MODERATE_SYSTEM_INSTRUCTION = '''
You are a knowledge tracking model that predicts whether a student will be able to get a new question right when he encounters it based on his history of doing the question, and the knowledge concepts in the question.
Give the format of the history of exercising as follows, with each line representing an exercise and a part of knowledge graph information beginning with<Knowledge graph information> and ending with <END knowledge graph information> :
\"exercise_id: exercise_id, 
exercise_content: exercise content, 
knowledge concepts: [knowledge concepts in the exercise], is_correct: 0 or 1\".
<Knowledge graph information>
knowledge graph
<END knowledge graph information>
a question, corresponding knowledge points, and knowledge graph information will be given, and you need to predict whether the student will be able to answer these questions correctly or not.
The output should only be 0 or 1, 1 for correct and 0 for incorrect. no other explanation is needed.
'''
GLM_MODERATE_FEWSHOT_USER_EXAMPLE = '''''' # only used when not generated by llms
GLM_MODERATE_PREDICT_INSTRUCTION = '''
Based on the system instructions, a few shots, and the information above. Note that the relationships between knowledge points in the knowledge graph are for reference only. Please analyze the specific connections between knowledge points in detail. Only respond with only one 0 or 1 to predict whether the student can answer <Exercise to Predict> correctly or not.
'''
GLM_MODERATE_ANALYSIS_INSTRUCTION = f'{GLM_SPARSE_ANALYSIS_INSTRUCTION}'
GLM_MODERATE_EXPLAINATION_INSTRUCTION = f'{GLM_SPARSE_EXPLAINATION_INSTRUCTION}'
GLM_MODERATE_SELF_REFLECTION_INSTRUCTION = ''''''
FEW_SHOT_RESPONSE_EXMAPLE = '0'


GPT_MODERATE_SYSTEM_INSTRUCTION = '''
You are a teacher who predicts and analyzes whether a student will be able to get a new question right when he encounters it based on his history of doing the question, and the knowledge concepts in the question based on the knowledge graph. Note that the relationships between knowledge points in the knowledge graph are for reference only. Please analyze the specific connections between knowledge points in detail.
Give the format of the history of exercising as follows, with each line representing an exercise:
\"exercise_id: exercise_id, 
exercise_content: exercise content, 
knowledge concepts: [knowledge concepts in the exercise], is_correct: 0 or 1\".
Then, a question and corresponding knowledge points will be given, and you need to predict whether the student will be able to answer these questions correctly or not.
The output should only be 0 or 1, 1 for correct and 0 for incorrect. no other explanation is needed.
'''
GPT_MODERATE_FEWSHOT_USER_EXAMPLE = ''''''
GPT_MODERATE_PREDICT_INSTRUCTION = '''
Based on the system instructions, few shots and information above. Only respond with only one 0 or 1 to predict whether the student can answer the last <Exercise to Predict> correctly or not.
'''
GPT_MODERATE_ANALYSIS_INSTRUCTION = f'{GLM_MODERATE_ANALYSIS_INSTRUCTION}'
GPT_MODERATE_EXPLAINATION_INSTRUCTION = f'{GLM_MODERATE_EXPLAINATION_INSTRUCTION}'
GPT_MODERATE_SELF_REFLECTION_INSTRUCTION = f'{GLM_MODERATE_SELF_REFLECTION_INSTRUCTION}'


model_mo_sys_instr = {
    'glm': GLM_MODERATE_SYSTEM_INSTRUCTION,
    'gpt': GPT_MODERATE_SYSTEM_INSTRUCTION
}
model_mo_fewshot_u_ex = {
    'glm': GLM_MODERATE_FEWSHOT_USER_EXAMPLE,
    'gpt': GPT_MODERATE_FEWSHOT_USER_EXAMPLE
}
model_mo_fewshot_s_ex = {
    'glm': FEW_SHOT_RESPONSE_EXMAPLE,
    'gpt': FEW_SHOT_RESPONSE_EXMAPLE
}
model_mo_predict_instr = {
    'glm': GLM_MODERATE_PREDICT_INSTRUCTION,
    'gpt': GPT_MODERATE_PREDICT_INSTRUCTION
}
model_mo_analysis_instr = {
    'glm': GLM_MODERATE_ANALYSIS_INSTRUCTION,
    'gpt': GPT_MODERATE_ANALYSIS_INSTRUCTION
}
model_mo_explain_instr = {
    'glm': GLM_MODERATE_EXPLAINATION_INSTRUCTION,
    'gpt': GPT_MODERATE_EXPLAINATION_INSTRUCTION
}
model_mo_self_refl_instr = {
    'glm': GLM_MODERATE_SELF_REFLECTION_INSTRUCTION,
    'gpt': GPT_MODERATE_SELF_REFLECTION_INSTRUCTION
}
def sample_fs_id(sample_list, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy) -> list:
    # return a list of tuples of (idex in sample_list, exercise_id)
    fs_ex_ids = []
    if fewshots_num == 0:
        return fs_ex_ids
    # fewshots_strategy: "random", "first", "last", "BLEU", "RAG"
    if fewshots_strategy == "random":
        # randomly sample exercise_id from student_info to create fewshots
        # sample_list is a numpy array of str
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in random.sample(range(len(sample_list)), fewshots_num)]
    elif fewshots_strategy == "first":
        # use the first fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(fewshots_num)]
    elif fewshots_strategy == "last":
        # use the last fewshot_num exercise_id from student_info to create fewshots
        if fewshots_num > len(sample_list):
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list))]
        else:
            fs_ex_ids = [(i, sample_list[i]) for i in range(len(sample_list)-fewshots_num, len(sample_list))]
    elif fewshots_strategy == "RAG":
        # TODO: implement Strategy
        raise NotImplementedError
    elif fewshots_strategy == "BLEU":
        # TODO: implement Strategy
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid fewshots_strategy: {fewshots_strategy}")
    return fs_ex_ids

def generic_create_sparse_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                   prompts,
                                   generate_analysis = True, generate_explanation = True) -> List[str]:
    # new create_fewshots function for llmsforkt
    ret_fewshots = []
    # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
    # 根据sample strategy来sample
    fs_ex_ids = sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
    if fewshots_num == 0:
        return ret_fewshots
    first = True

    for i, ex_id in fs_ex_ids:
        # get all the info of exercise and convert it to string
        fewshot = ''
        # add onehot info
        other_ex_info = extra_datas['exercise_info']
        ex_id_skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        is_correct = 'right' if int(student_info['is_corrects'][i]) else 'wrong'
        # generate explaination of the fewshot
        if first:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"
            print(f'initializing fewshot with exercise {ex_id}')
            first = False
        else:
            fewshot += f"exercise_id: {ex_id}, knowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"
            if generate_explanation:
                print(f'generating explaination for exercise {ex_id} in fewshot')
                # print(f"few shot is {fewshot}")
                # print("XXXXXXXXXXXXXXXX")
                # print(f"ret fewshots previous shot is {ret_fewshots}")
                # print("XXXXXXXXXXXXXXXX")
              
                explanation = generate_explaination(ret_fewshots, test_exercise_info['is_correct'], prompts)
                
                # print(f"ret fewshots after shot is {ret_fewshots}")
                # print("XXXXXXXXXXXXXXXX")
                # print(f"ret exlaination shot is {explanation}")
                # print("XXXXXXXXXXXXXXXX")
                fewshot += f"Explaination: {explanation}\n"
                # print(explanation)
                # exit()

        ret_fewshots.append(fewshot)
        print()
    # print('\n'.join(ret_fewshots))
    return ret_fewshots
 # generic_create_moderate_fewshots(student_info, test_exercise_info, 
#  extra_datas, fewshots_num, fewshots_strategy, prompts,generate_analysis, generate_explanation)   
def generic_create_moderate_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                   prompts, 
                                   generate_analysis = True, generate_explanation = True) -> List[str]:
    
    # new create_fewshots function for llmsforkt
    ret_fewshots = []
    # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
    fs_ex_ids = sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
    if fewshots_num == 0:
        return ret_fewshots
    first = True

    for i, ex_id in fs_ex_ids:
        # get all the info of exercise and convert it to string
        fewshot = ''
        # add onehot info
        other_ex_info = extra_datas['exercise_info']
        ex_id_skill_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['skill_desc'].values[0]
        is_correct = 'right' if student_info['is_corrects'][i] else 'wrong'
        ex_desc = other_ex_info[other_ex_info['exercise_id'] == ex_id]['exercise_desc'].values[0]
        fewshot += f"exercise_id: {ex_id}\nexercise content: {ex_desc}\nknowledge concepts: {ex_id_skill_desc}, is_correct: {is_correct}\n"    
        # generate explaination of the fewshot
        if first:
            print(f'initializing fewshot with exercise {ex_id}')
            # can change to not generate explanation for the first fewshot
            # if generate_explanation:
            #     print(f'generating explaination for exercise {ex_id} in fewshot')
            #     explanation = generate_explanation_func(fewshot, test_exercise_info['is_correct'], prompts)
            #     fewshot += f"Explaination: {explanation}\n"
            first = False
        else:
            if generate_explanation:
                print(f'generating explaination for exercise {ex_id} in fewshot')
                # print(ret_fewshots)
                # print(test_exercise_info['is_correct'])
                explanation = generate_explaination(ret_fewshots, test_exercise_info['is_correct'], prompts)
                fewshot += f"Explaination: {explanation}\n"
                # print(explanation)
                # exit()
        ret_fewshots.append(fewshot)


    # print('\n'.join(ret_fewshots))
    return ret_fewshots
    
def generic_create_fewshots(student_info, 
                            test_exercise_info, 
                            extra_datas, 
                            fewshots_num, 
                            fewshots_strategy, 
                            data_mode, 
                            prompts,
                            generate_analysis = True, 
                            generate_explanation = True) -> List[str]:
    if data_mode == "onehot":
        return generic_create_onehot_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis, generate_explanation)
    elif data_mode == "sparse":
        return generic_create_sparse_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis, generate_explanation)
    elif data_mode == "moderate":
        return generic_create_moderate_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts,generate_analysis, generate_explanation)
    elif data_mode == "rich":
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid data_mode: {data_mode}, when creating fewshots")

def generic_create_analysis_messages(fewshots: List[str], prompts: dict):
    # create analysis messages for the prompt
    messages = []
    if len(fewshots) == 0:
        raise ValueError("fewshots should not be empty, when creating analysis messages")
    messages.append(
        Message(
            role='system',
            content=prompts['sys_instr']
        )
    )
    messages.append(
        Message(
            role='user',
            content='\n'.join(fewshots) + '\n' + prompts['analysis_instr']
        )
    )
    return messages

def generic_create_explaination_messages(fewshots: List[str], prediction: str, prompts: dict):
    # create explanation messages for the prediction
    # ["exercise_id: 13, knowledge concepts: 
    #['Separate a whole number from a fraction', 'Find a common denominator', 'Borrow from whole number part', 'Subtract numerators'], 
    #is_correct: right\n"]
    messages = []
    if len(fewshots) == 0:
        raise ValueError("fewshots should not be empty, when creating explanation messages")

    messages.append(
        Message(
            role='system',
            content=prompts['sys_instr']
        )
    )
    if len(fewshots) > 1:
        messages.append(
            Message(
                role='user',
                content="Previous exercise information:" + '\n'.join(fewshots[:-1]).replace("\\","")
            )
        )
    messages.append(
        Message(
            role='user',
            content="Current exercise information:" + '\n'.join(fewshots[-1].split(",")) + f"The student answer is: {'right' if prediction == '1' else 'wrong'}.".replace("\\","")
        )
    )
    messages.append(
        Message(
            role='user',
            content=prompts['explain_instr'].replace("\\","")
        )
    )
    return  messages

def generate_analysis(fewshots: List[str], prompts: dict) -> Union[List[str], str]:
    analysis_messages = generic_create_analysis_messages(fewshots, prompts)
    return glm_chat(model_name, analysis_messages, num_comps=1)

def generate_explaination(fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
    explaination_messages = generic_create_explaination_messages(fewshots, prediction, prompts)      
    return glm_chat(model_name, explaination_messages, num_comps=1)

def create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts) -> List[str]:

        return generic_create_fewshots(student_info, 
                                       test_exercise_info, 
                                       extra_datas, 
                                       fewshots_num, 
                                       fewshots_strategy, 
                                       data_mode, 
                                       prompts
                                       )

def get_prompts(data_mode: str):
        return generic_get_prompts('glm', data_mode)
def generic_get_prompts(model_name:str, data_mode:str):
    return {
            'sys_instr': model_mo_sys_instr[model_name],
            'fewshot_u_ex': model_mo_fewshot_u_ex[model_name],
            'fewshot_s_ex': model_mo_fewshot_s_ex[model_name],
            'predict_instr': model_mo_predict_instr[model_name],
            'analysis_instr': model_mo_analysis_instr[model_name],
            'explain_instr': model_mo_explain_instr[model_name],
            'self_refl_instr': model_mo_self_refl_instr[model_name]
        }
def aggregate_data(id, extra_data, data_type):
    # data_type is exercise or student
    agg_data = {}
    if data_type == "exercise":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['exercise_id'] == id][col].values[0]
    elif data_type == "student":
        for col in extra_data.columns:
            agg_data[col] = extra_data[extra_data['student_id'] == id][col].values[0]
    else:
        raise ValueError(f"Invalid data_type: {data_type}, when aggregating data")
    return agg_data

    
def item_selection(train_data):
    data1 = train_data.sample(frac=1).reset_index(drop=True)
    return data1[0:50], data1[50:100]
    

data1, data2 = item_selection(train_data)
list_data = []
list_data.append(data1)

for data3 in list_data: 
    train_data = data3
    eval_results = {}
    
    total_y_pre = []
    total_y_true = []
    # start evaluation, each iteration returns a dict of eval results for one student
    for i, student_info in train_data.iterrows():
        logger.write(f"----------------Evaluating student {student_info['student_id']}")
        test_exercises = test_data[test_data['student_id'] == student_info['student_id']]['exercises_logs'].values[0]
        test_corrects = test_data[test_data['student_id'] == student_info['student_id']]['is_corrects'].values[0]
        # key: exercise_id, value: dict of eval results for one exercise
        stu_eval_results = {}
        # extra_exercicse_info = extra_datas['exercise_info']
        flag = True
        # test_exercises is a nump array of exercise_ids
        for i, test_exercise_id in enumerate(test_exercises):
            test_exercise_id = str(test_exercise_id)
            logger.write(f"*****Evaluating student {student_info['student_id']} on exercise {test_exercise_id}")
            is_correct = test_corrects[i]
            # get exercise_info
            test_exercise_info = {'exercise_id': test_exercise_id, 'is_correct': str(is_correct)}
            test_exercise_info.update(aggregate_data(test_exercise_id, extra_datas['exercise_info'], 'exercise'))
            
            prompts = get_prompts(data_mode=data_mode)
            fewshots = create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts)
    
            # 这里为testing
            logger.write(f"Fewshots:\n{fewshots}")
            exe_eval_results,prompt_final = evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode, knowledge_graph_info)
            exe_eval_results.update({'is_correct': is_correct})
            stu_eval_results[test_exercise_id] = exe_eval_results
            if i>10:
                break
        y_pre = []
        y_true = []
        for k, v in stu_eval_results.items(): # k is exercise_id, v is dict of eval results for one exercise
            y_pre.append(int(v['prediction']))
            y_true.append(int(v['is_correct']))
        stu_acc = accuracy_score(y_true, y_pre)
        stu_precision = precision_score(y_true, y_pre)
        stu_recall = recall_score(y_true, y_pre)
        stu_f1 = f1_score(y_true, y_pre)
        stu_test_count = len(y_true)
        logger.write(f"y_true: {y_true}, y_pre: {y_pre}")
        logger.write(f"Student {student_info['student_id']}, len: {stu_test_count}, acc: {stu_acc}, precision: {stu_precision}, recall: {stu_recall}, f1: {stu_f1}")
        stu_result = {'student_id': student_info['student_id'], 'accuracy': stu_acc, 'eval_results': stu_eval_results}
        eval_results[student_info['student_id']] = stu_result
        total_y_pre.extend(y_pre)
        total_y_true.extend(y_true)
        
    final_acc = accuracy_score(total_y_true, total_y_pre)
    fianl_precision = precision_score(total_y_true, total_y_pre)
    fianl_recall = recall_score(total_y_true, total_y_pre)
    fianl_f1 = f1_score(total_y_true, total_y_pre)
    final_auc = roc_auc_score(total_y_true, total_y_pre)
    logger.write(f"Total test student: {len(train_data)}, total test count: {len(total_y_true)}")
    logger.write(f"Final accuracy: {final_acc}, precision: {fianl_precision}, recall: {fianl_recall}, f1: {fianl_f1}, AUC: {final_auc}")
# return eval_results








