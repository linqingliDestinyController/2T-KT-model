import os
import argparse
from mylogger1 import Logger
import dataclasses
from typing import List, Optional, Literal,Tuple, Union
import random
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
import time
import pandas as pd
from GLM import glm_chat
import prompt
from model import TT_KT

sparse_exercise_info_type = {'exercise_id': 'str', 'skill_ids': 'object', 'skill_desc': 'object'}
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
    
def load_sparse_data(data_path: str) -> dict:
    # spase: skills in each questions
    sp_e_info_path = os.path.join(data_path, "sparse_exercise_info.jsonl")
    sp_e_info_df = pd.read_json(sp_e_info_path, lines=True, dtype=sparse_exercise_info_type)
    # format: exercise_id, list of skill_ids in exercise_id, list of skill_desc of skill_ids
    logger.write("Loaded sparse_exercise_info data")
    return {"exercise_info": sp_e_info_df}

def get_args():
    parser = argparse.ArgumentParser()
    # LLM model name
    parser.add_argument('--model_type', type=str, default='llm', help='model type llm or ktm')
    parser.add_argument('--model_name', type=str, default='glm-4', help='model name')
    parser.add_argument('--data_path', type=str, default='./datasets', help='data path')
    # data_mode: sparse, moderate, rich
    parser.add_argument('--data_mode', type=str, default='sparse', help='data mode: onehot, sparse, moderate, rich')
    # dataset name
    parser.add_argument('--dataset_name', type=str, default='FrcSub', help='dataset name')
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
    parser.add_argument('--fewshot_num', type=int, default=4, help='fewshot num, 0 means zero-shot')
    parser.add_argument('--fewshot_strategy', type=str, default='random', help='fewshot strategy, random/first/last/RAG')
    parser.add_argument('--eval_strategy', type=str, default='simple', help='eval strategy, simple/analysis/self_correct')
    args = parser.parse_known_args()[0]
    return args


  
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
        user_data = model.create_user_data(student_info, test_exercise_info, extra_datas, data_mode=data_mode, knowledge_graph_info = knowledge_graph_info )
        
        prediction_response, prediction_message = model.generate_prediction(fewshots, user_data, prompts)
        
        try:
            eval_result['prediction'] = check_response_format(prediction_response)
        except ValueError as e:
            # if the repsonse format is not correct, generate again, if again, then randomly choose 0 or 1
            response_tmp,prediction_message = model.generate_prediction(fewshots, user_data, prompts)
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
       
        eval_result['explaination'] = model.generate_explaination(fewshots, eval_result['prediction'], prompts)
        

    elif eval_strategy == 'analysis':
        raise NotImplementedError
    elif eval_strategy =='self_correct':
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid eval_strategy: {eval_strategy}")
    
    # self.logger.write(f"Evaluation result: {eval_result}")
    return eval_result, fewshots


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



args = get_args()
my_logger = Logger(args)
logger = my_logger

data_path = os.path.join(args.data_path, args.data_mode, args.dataset_name)
if not os.path.exists(data_path):
    print(data_path)
    raise ValueError("data path not exist, check path, mode, or dataset name")
    
if args.model_type == 'llm':
    train_data, test_data = load_user_data(data_path=data_path, train_split=args.train_split, is_shuffle=args.is_shuffle)
   
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

eval_results = {}
total_y_pre = []
total_y_true = []
model = TT_KT(args)
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
        logger.write(f"*****Evaluating student {student_info['student_id']} on exercise {test_exercise_id}")
        is_correct = test_corrects[i]
        # get exercise_info
        test_exercise_info = {'exercise_id': test_exercise_id, 'is_correct': is_correct}
        test_exercise_info.update(aggregate_data(test_exercise_id, extra_datas['exercise_info'], 'exercise'))
        
        prompts = model.get_prompts(data_mode=data_mode)
        fewshots = model.create_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts)

        # 这里为testing
        logger.write(f"Fewshots:\n{fewshots}")
        exe_eval_results,prompt_final = evaluate(student_info, test_exercise_info, extra_datas, eval_strategy, fewshots, prompts, data_mode, prompt.knowledge_graph_info)
        exe_eval_results.update({'is_correct': is_correct})
        stu_eval_results[test_exercise_id] = exe_eval_results
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