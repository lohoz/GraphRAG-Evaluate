import json
from utils.common_utils import common_utils

from openai import OpenAI
import argparse
import os
from datetime import datetime
import concurrent.futures
from prompt import SYS_PROMPT, EVAL_PROMPT
from env import BASEURL, APIKEY


RESULT_LOG_FILE = 'evaluate_result_log.jsonl'


def parse_args():
    parser = argparse.ArgumentParser(
        description="evaluate"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rag_name_1",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rag_name_2",
        type=str,
        default=None
    )

    parser.add_argument(
        "--answer_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--repet_time",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--compare_result_path",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--evaluate_times",
        type=int,
        default=1,
    )    

    return parser.parse_args()



def evaluate(dataset_name, rag_name_1, rag_name_2, answer_path, repet_time, compare_result_path):
    ans1win = 0
    ans2win = 0
    answintie = 0

    answer_dataset = common_utils.load_jsonl_dataset(answer_path)
    
    # print(len(answer_dataset))

    exist_list = []
    if os.path.exists(compare_result_path):
        exist_data = common_utils.load_jsonl_dataset(compare_result_path)
        for item in exist_data:
            if item['gap'] > 0:
                ans1win += 1
            elif item['gap'] < 0:
                ans2win += 1
            else:
                answintie += 1
            exist_list.append(item['question_id'])
    
    is_remove_count = 0

    for answer_data in answer_dataset:
        question_id = answer_data['question_id']
        is_remove = answer_data['is_remove']

        if is_remove:
            is_remove_count += 1
            continue

        if question_id in exist_list:
            continue

        query = answer_data['question'].strip()
        ans1 = answer_data['answer'][rag_name_1].replace("\n", " ").strip()
        ans2 = answer_data['answer'][rag_name_2].replace("\n", " ").strip()

        

        ans1_aspect1_1 = 0
        ans2_aspect1_1 = 0
        ans1_aspect2_1 = 0
        ans2_aspect2_1 = 0
        ans1_aspect3_1 = 0
        ans2_aspect3_1 = 0
        ans1_aspect4_1 = 0
        ans2_aspect4_1 = 0

        ans2_aspect1_2 = 0
        ans1_aspect1_2 = 0
        ans2_aspect2_2 = 0
        ans1_aspect2_2 = 0
        ans2_aspect3_2 = 0
        ans1_aspect3_2 = 0
        ans2_aspect4_2 = 0
        ans1_aspect4_2 = 0

        client = OpenAI(
                base_url=BASEURL,
                api_key=APIKEY,
                timeout=120
                )
        
        prompt = EVAL_PROMPT.format(query=query, ans1=ans1, ans2=ans2)
        for j in range (0,repet_time):

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt}
                ]
            )
            judge = response.choices[0].message.content
            judge = common_utils.extract_json(judge).replace("\\", "\\\\")

            while True:
                try:
                    data = json.loads(judge)
                    print(judge)

                    int(data["Aspect 1"]["Answer 1"])
                    int(data["Aspect 2"]["Answer 1"])
                    int(data["Aspect 3"]["Answer 1"])
                    int(data["Aspect 4"]["Answer 1"])
                    int(data["Aspect 1"]["Answer 2"])
                    int(data["Aspect 2"]["Answer 2"])
                    int(data["Aspect 3"]["Answer 2"])
                    int(data["Aspect 4"]["Answer 2"])

                    break
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error: {e}, retrying... ")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                                {"role": "system", "content": SYS_PROMPT},
                                {"role": "user", "content": prompt}
                        ]
                    )
                    judge = response.choices[0].message.content
                    judge = common_utils.extract_json(judge).replace("\\", "\\\\")

            ans1_aspect1_1 += int(data["Aspect 1"]["Answer 1"])
            ans2_aspect1_1 += int(data["Aspect 1"]["Answer 2"])

            ans1_aspect2_1 += int(data["Aspect 2"]["Answer 1"])
            ans2_aspect2_1 += int(data["Aspect 2"]["Answer 2"])

            ans1_aspect3_1 += int(data["Aspect 3"]["Answer 1"])
            ans2_aspect3_1 += int(data["Aspect 3"]["Answer 2"])

            ans1_aspect4_1 += int(data["Aspect 4"]["Answer 1"])
            ans2_aspect4_1 += int(data["Aspect 4"]["Answer 2"])

        ans1_aspect1_1 = float(ans1_aspect1_1) / repet_time
        ans2_aspect1_1 = float(ans2_aspect1_1) / repet_time
        ans1_aspect2_1 = float(ans1_aspect2_1) / repet_time
        ans2_aspect2_1 = float(ans2_aspect2_1) / repet_time
        ans1_aspect3_1 = float(ans1_aspect3_1) / repet_time
        ans2_aspect3_1 = float(ans2_aspect3_1) / repet_time
        ans1_aspect4_1 = float(ans1_aspect4_1) / repet_time
        ans2_aspect4_1 = float(ans2_aspect4_1) / repet_time


        prompt = EVAL_PROMPT.format(query=query, ans1=ans2, ans2=ans1)
        for k in range (0 , repet_time):

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                        {"role": "system", "content": SYS_PROMPT},
                        {"role": "user", "content": prompt}
                ]
            )
            judge = response.choices[0].message.content
            judge = common_utils.extract_json(judge).replace("\\", "\\\\")

            while True:
                try:
                    data = json.loads(judge)
                    print(judge)

                    int(data["Aspect 1"]["Answer 1"])
                    int(data["Aspect 1"]["Answer 2"])
                    int(data["Aspect 2"]["Answer 1"])
                    int(data["Aspect 2"]["Answer 2"])
                    int(data["Aspect 3"]["Answer 1"])
                    int(data["Aspect 3"]["Answer 2"])
                    int(data["Aspect 4"]["Answer 1"])
                    int(data["Aspect 4"]["Answer 2"])

                    break
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    print(f"Error: {e}, retrying... ")
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                                {"role": "system", "content": SYS_PROMPT},
                                {"role": "user", "content": prompt}
                        ]
                    )
                    judge = response.choices[0].message.content
                    judge = common_utils.extract_json(judge).replace("\\", "\\\\")

            ans2_aspect1_2 += int(data["Aspect 1"]["Answer 1"])
            ans1_aspect1_2 += int(data["Aspect 1"]["Answer 2"])
            ans2_aspect2_2 += int(data["Aspect 2"]["Answer 1"])
            ans1_aspect2_2 += int(data["Aspect 2"]["Answer 2"])
            ans2_aspect3_2 += int(data["Aspect 3"]["Answer 1"])
            ans1_aspect3_2 += int(data["Aspect 3"]["Answer 2"])
            ans2_aspect4_2 += int(data["Aspect 4"]["Answer 1"])
            ans1_aspect4_2 += int(data["Aspect 4"]["Answer 2"])

        ans2_aspect1_2 = float(ans2_aspect1_2) / repet_time
        ans1_aspect1_2 = float(ans1_aspect1_2) / repet_time
        ans2_aspect2_2 = float(ans2_aspect2_2) / repet_time
        ans1_aspect2_2 = float(ans1_aspect2_2) / repet_time
        ans2_aspect3_2 = float(ans2_aspect3_2) / repet_time
        ans1_aspect3_2 = float(ans1_aspect3_2) / repet_time
        ans2_aspect4_2 = float(ans2_aspect4_2) / repet_time
        ans1_aspect4_2 = float(ans1_aspect4_2) / repet_time



        ans1_aspect1 = (ans1_aspect1_2 + ans1_aspect1_1) / 2
        ans2_aspect1 = (ans2_aspect1_2 + ans2_aspect1_1) / 2

        ans1_aspect2 = (ans1_aspect2_2 + ans1_aspect2_1) / 2
        ans2_aspect2 = (ans2_aspect2_2 + ans2_aspect2_1) / 2

        ans1_aspect3 = (ans1_aspect3_2 + ans1_aspect3_1) / 2
        ans2_aspect3 = (ans2_aspect3_2 + ans2_aspect3_1) / 2

        ans1_aspect4 = (ans1_aspect4_2 + ans1_aspect4_1) / 2
        ans2_aspect4 = (ans2_aspect4_2 + ans2_aspect4_1) / 2

        ans1sum = ans1_aspect1 + ans1_aspect2 + ans1_aspect3 + ans1_aspect4
        ans2sum = ans2_aspect1 + ans2_aspect2 + ans2_aspect3 + ans2_aspect4

        print("Answer 1 Aspect 1: " + str(ans1_aspect1))
        print("Answer 1 Aspect 2: " + str(ans1_aspect2))
        print("Answer 1 Aspect 3: " + str(ans1_aspect3))
        print("Answer 1 Aspect 4: " + str(ans1_aspect4))
        print("Answer 1 Total: " + str(ans1sum))

        print("Answer 2 Aspect 1: " + str(ans2_aspect1))
        print("Answer 2 Aspect 2: " + str(ans2_aspect2))
        print("Answer 2 Aspect 3: " + str(ans2_aspect3))
        print("Answer 2 Aspect 4: " + str(ans2_aspect4))
        print("Answer 2 Total: " + str(ans2sum))


        data = {}

        data['question_id'] = question_id
        data['question'] = query
        
        data_1 = {}
        data_1['answer'] = ans1
        data_1['aspect_1'] = ans1_aspect1
        data_1['aspect_2'] = ans1_aspect2
        data_1['aspect_3'] = ans1_aspect3
        data_1['aspect_4'] = ans1_aspect4
        data_1['sum'] = ans1sum
        data[rag_name_1] = data_1

        data_2 = {}
        data_2['answer'] = ans2
        data_2['aspect_1'] = ans2_aspect1
        data_2['aspect_2'] = ans2_aspect2
        data_2['aspect_3'] = ans2_aspect3
        data_2['aspect_4'] = ans2_aspect4
        data_2['sum'] = ans2sum
        data[rag_name_2] = data_2

        gap = ans1sum - ans2sum
        data['gap'] = gap

        common_utils.append_jsonl(compare_result_path, data)


        if(ans1sum > ans2sum):
            ans1win += 1
        elif (ans1sum < ans2sum):
            ans2win += 1
        else: answintie += 1

        print(f"------{rag_name_1} VS {rag_name_2}--------------")
        print(ans1win)
        print(ans2win)
        print(answintie)
    


    if ans1win + ans2win + answintie != len(answer_dataset) - is_remove_count:
        raise("Error: The number of answers evaluated does not match the total number of answers.")
    
    compare_info = {}
    compare_info['compare_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    compare_info['dataset_name'] = dataset_name
    compare_info['compare_rag_names'] = [rag_name_1, rag_name_2]
    compare_info['compare_result_path'] = compare_result_path
    compare_info[rag_name_1 + '_win'] = ans1win
    compare_info[rag_name_2 + '_win'] = ans2win
    compare_info['tie'] = answintie

    if os.path.exists(RESULT_LOG_FILE):
        result_dataset = common_utils.load_jsonl_dataset(RESULT_LOG_FILE)
        exist_result_file_list = []
        for item in result_dataset:
            exist_result_file_list.append(item['compare_result_path'])
        if compare_result_path not in exist_result_file_list:
            common_utils.append_jsonl(RESULT_LOG_FILE, compare_info)
        else:
            print("The result file already exists in the log file.")
    else:
        common_utils.append_jsonl(RESULT_LOG_FILE, compare_info)



def main():
    args = parse_args()
    print(args)

    dataset_name = args.dataset_name

    rag_name_1 = args.rag_name_1

    rag_name_2 = args.rag_name_2

    answer_path = args.answer_path

    repet_time = args.repet_time

    compare_result_path = args.compare_result_path

    evaluate_times = args.evaluate_times

    parent_dir = f'./datasets/{dataset_name}/evaluate_result'
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if evaluate_times > 1:
        compare_result_path_list = [f'{parent_dir}/{rag_name_1}_and_{rag_name_2}_{i+1}.jsonl' for i in range(evaluate_times)]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result_futures = [
                executor.submit(evaluate, dataset_name, rag_name_1, rag_name_2, answer_path, repet_time, compare_result_path) 
                for compare_result_path in compare_result_path_list
            ]
            for future in concurrent.futures.as_completed(result_futures):
                try:
                    future.result()
                except Exception as e:
                    print('Exception is', e, type(e))
        
        return

    if compare_result_path is None:
        compare_result_path = f'{parent_dir}/{rag_name_1}_and_{rag_name_2}.jsonl'
        evaluate(dataset_name, rag_name_1, rag_name_2, answer_path, repet_time, compare_result_path)



if __name__ == '__main__':
    main()