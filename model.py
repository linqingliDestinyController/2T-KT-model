import os
import argparse
from mylogger1 import Logger
import dataclasses
from typing import List, Optional, Literal,Tuple, Union
# from utils import aggregate_data
import random
import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import datetime
import time
import pandas as pd
from GLM import glm_chat
# from Advice import advice
import prompt
sparse_exercise_info_type = {'exercise_id': 'str', 'skill_ids': 'object', 'skill_desc': 'object'}
MessageRole = Literal["system", "user", "assistant"]
# @dataclasses.dataclass()
model_sp_sys_instr = {
    'glm': prompt.GLM_SPARSE_SYSTEM_INSTRUCTION,
    'gpt': prompt.GPT_SPARSE_SYSTEM_INSTRUCTION
}
model_sp_fewshot_u_ex = {
    'glm': prompt.GLM_SPARSE_FEWSHOT_USER_EXAMPLE,
    'gpt': prompt.GPT_SPARSE_FEWSHOT_USER_EXAMPLE
}
model_sp_fewshot_s_ex = {
    'glm': prompt.FEW_SHOT_RESPONSE_EXMAPLE,
    'gpt': prompt.FEW_SHOT_RESPONSE_EXMAPLE
}
model_sp_predict_instr = {
    'glm': prompt.GLM_SPARSE_PREDICT_INSTRUCTION,
    'gpt': prompt.GPT_SPARSE_PREDICT_INSTRUCTION
}
model_sp_analysis_instr = {
    'glm': prompt.GLM_SPARSE_ANALYSIS_INSTRUCTION,
    'gpt': prompt.GPT_SPARSE_ANALYSIS_INSTRUCTION
}
model_sp_explain_instr = {
    'glm': prompt.GLM_SPARSE_EXPLAINATION_INSTRUCTION,
    'gpt': prompt.GPT_SPARSE_EXPLAINATION_INSTRUCTION
}
model_sp_self_refl_instr = {
    'glm': prompt.GLM_SPARSE_SELF_REFLECTION_INSTRUCTION,
    'gpt': prompt.GPT_SPARSE_SELF_REFLECTION_INSTRUCTION
}

model_name="glm-4"
MessageRole = Literal["system", "user", "assistant"]
@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


class TT_KT:
    def __init__(self, params):
        self.params = params
   

    def create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info) -> str:
            return self.generic_create_user_data(student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info)
    
    def generic_create_user_data(self, student_info, test_exercise_info, extra_datas, data_mode, knowledge_graph_info) -> str:
      
        user_data = ''
        
        # add the related knowledge graph
        user_data += '<Knowledge graph information>\n'
        user_data += knowledge_graph_info
        user_data += '<END knowledge graph information>\n'
        
         # # add the advice
        user_data += '<Advice>\n'
        user_data += prompt.advice
        user_data += '<END Advice>\n'
        
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

    def generic_create_prediction_messages(self, fewshots, user_data, prompts, use_selected_fewshot=True) -> List[Message]:
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
    
    def create_prediction_messages(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> List[Message]:
            return self.generic_create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
    
    def generate_prediction(self, fewshots: List[str], user_data: str, prompts: dict, use_selected_fewshot: bool = True) -> Union[List[str], str]:
            prediction_messages = self.create_prediction_messages(fewshots, user_data, prompts, use_selected_fewshot)
            # print(f"The prediction message is {prediction_messages}")
            return glm_chat(model_name, prediction_messages, num_comps=1), prediction_messages
    def sample_fs_id(self, sample_list, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy) -> list:
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
    
    def generic_create_sparse_fewshots(self, student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy,
                                       prompts,
                                       generate_analysis = True, generate_explanation = True) -> List[str]:
        # new create_fewshots function for llmsforkt
        ret_fewshots = []
        # fs_ex_ids = list of tuples (idex in student_info['exercises_logs'], exercise_id)
        # 根据sample strategy来sample
        fs_ex_ids = self.sample_fs_id(student_info['exercises_logs'], test_exercise_info, extra_datas, fewshots_num, fewshots_strategy)
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
                   
                  
                    explanation = self.generate_explaination(ret_fewshots, test_exercise_info['is_correct'], prompts)
                    
                  
                    fewshot += f"Explaination: {explanation}\n"
                   
    
            ret_fewshots.append(fewshot)
            print()

        return ret_fewshots
    
    def generic_create_fewshots(self, student_info, 
                                test_exercise_info, 
                                extra_datas, 
                                fewshots_num, 
                                fewshots_strategy, 
                                data_mode, 
                                prompts,
                                generate_analysis = True, 
                                generate_explanation = True) -> List[str]:
        if data_mode == "onehot":
            return self.generic_create_onehot_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis, generate_explanation)
        elif data_mode == "sparse":
            return self.generic_create_sparse_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts, generate_analysis, generate_explanation)
        elif data_mode == "moderate":
            return self.generic_create_moderate_fewshots(student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, prompts,generate_analysis, generate_explanation)
        elif data_mode == "rich":
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid data_mode: {data_mode}, when creating fewshots")
    
    def generic_create_analysis_messages(self, fewshots: List[str], prompts: dict):
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
    
    def generic_create_explaination_messages(self, fewshots: List[str], prediction: str, prompts: dict):
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
    
    def generate_analysis(self, fewshots: List[str], prompts: dict) -> Union[List[str], str]:
        analysis_messages = generic_create_analysis_messages(fewshots, prompts)
        return glm_chat(model_name, analysis_messages, num_comps=1)
    
    def generate_explaination(self, fewshots: List[str], prediction, prompts: dict) -> Union[List[str], str]:
        explaination_messages = self.generic_create_explaination_messages(fewshots, prediction, prompts)      
        return glm_chat(model_name, explaination_messages, num_comps=1)
    
    def create_fewshots(self, student_info, test_exercise_info, extra_datas, fewshots_num, fewshots_strategy, data_mode, prompts) -> List[str]:
    
            return self.generic_create_fewshots(student_info, 
                                           test_exercise_info, 
                                           extra_datas, 
                                           fewshots_num, 
                                           fewshots_strategy, 
                                           data_mode, 
                                           prompts
                                           )
    
    def get_prompts(self, data_mode: str):
            return self.generic_get_prompts('glm', data_mode)
    def generic_get_prompts(self, model_name:str, data_mode:str):
        return {
                'sys_instr': model_sp_sys_instr[model_name],
                'fewshot_u_ex': model_sp_fewshot_u_ex[model_name],
                'fewshot_s_ex': model_sp_fewshot_s_ex[model_name],
                'predict_instr': model_sp_predict_instr[model_name],
                'analysis_instr': model_sp_analysis_instr[model_name],
                'explain_instr': model_sp_explain_instr[model_name],
                'self_refl_instr': model_sp_self_refl_instr[model_name]
            }