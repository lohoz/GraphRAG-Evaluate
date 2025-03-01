import os
from utils.common_utils import common_utils
import argparse
from openai import OpenAI
from multiprocessing import Pool
from env import BASEURL, APIKEY

SYS_PROMPT = """
---Role---
You are an expert at adding some words to a paragraph without changing the meaning of the paragraph.
"""
PROMPT = """
You will be given a paragraph your task is to add {gap} words to the paragraph without changing the meaning of the paragraph, and return the content(just the content).

Here is the paragraph:
{ans}

add {gap} words to the paragraph without changing the meaning of the paragraph, don't make it too long.

Output: A paragraph that has been added {gap} words.
"""


def add_words(ans, add_len):
    client = OpenAI(
        base_url=BASEURL,
        api_key=APIKEY,
        timeout=120
    )
    response = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": PROMPT.format(ans=ans, gap=add_len)}
          ]
        )
    new_ans = response.choices[0].message.content
    new_ans = new_ans.replace("\n", " ").strip()
    new_ans_length = len(new_ans.split())

    return new_ans, new_ans_length


def align(answer_path):
    parent_dir = os.path.dirname(answer_path)
    rag_name_1 = answer_path.split('/')[-1].split('.')[0].split('_')[0]
    rag_name_2 = answer_path.split('/')[-1].split('.')[0].split('_')[2]
    align_path = os.path.join(parent_dir, f'{rag_name_1}_and_{rag_name_2}_align.jsonl')

    exist_align = []
    if os.path.exists(align_path):
        exist_dataset = common_utils.load_jsonl_dataset(align_path)
        for data in exist_dataset:
            exist_align.append(data['question_id'])

    answer_dataset = common_utils.load_jsonl_dataset(answer_path)

    for data in answer_dataset:
        if data['question_id'] in exist_align:
            continue
        
        if data['is_remove']:
            answer_1 = data['answer'][rag_name_1]
            answer_2 = data['answer'][rag_name_2]
            answer_1_length = len(answer_1.split())
            answer_2_length = len(answer_2.split())

            retry_times = 5
            if answer_1_length < answer_2_length:
                while(retry_times > 0):
                    print(f"Attempt {6 - retry_times} of 5: Aligning {rag_name_1} and {rag_name_2} for question {data['question_id']}...")
                    answer_1, answer_1_length = add_words(answer_1, answer_2_length - answer_1_length)
                    if abs(answer_1_length - answer_2_length) < 10:
                        data['is_remove'] = False
                        break
                    retry_times -= 1
                data['answer'][rag_name_1] = answer_1

            elif answer_1_length > answer_2_length:
                while(retry_times > 0):
                    print(f"Attempt {6 - retry_times} of 5: Aligning {rag_name_1} and {rag_name_2} for question {data['question_id']}...")
                    answer_2, answer_2_length = add_words(answer_2, answer_1_length - answer_2_length)
                    if abs(answer_1_length - answer_2_length) < 10:
                        data['is_remove'] = False
                        break
                    retry_times -= 1
                data['answer'][rag_name_2] = answer_2
            
        else:
            print(f"Skip {rag_name_1} and {rag_name_2} align for question {data['question_id']}")

        common_utils.append_jsonl(align_path, data)


def mutli(dataset_name):
    folder_path = f'./datasets/{dataset_name}/answer'
    answer_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jsonl')]

    with Pool() as p:
        data_infos = p.map(align, answer_files)
    del data_infos



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    args = parser.parse_args()

    mutli(args.dataset_name)


if __name__ == '__main__':
    main()
