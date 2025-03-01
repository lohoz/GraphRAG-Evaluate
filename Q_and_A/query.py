import os
from utils.common_utils import common_utils
from rag_methods import get_query
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--question_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rag_name_1", 
        type=str, 
        default=None, 
        help="rag name 1"
    )

    parser.add_argument(
        "--graph_info_dir_1", 
        type=str, 
        default=None, 
        help="working directory"  
    )

    parser.add_argument(
        "--rag_name_2", 
        type=str, 
        default=None, 
        help="rag name 2"
    )

    parser.add_argument(
        "--graph_info_dir_2", 
        type=str, 
        default=None, 
        help="working directory"  
    )

    parser.add_argument(
        "--length_gap", 
        type=int, 
        default=None, 
        help="length gap between two answers"
    )

    parser.add_argument(
        "--length_ratio", 
        type=float, 
        default=None, 
        help="length ratio between two answers"
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None
    )


    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    rag_name_1 = args.rag_name_1
    rag_name_2 = args.rag_name_2
    graph_info_dir_1 = args.graph_info_dir_1
    graph_info_dir_2 = args.graph_info_dir_2
    question_path = args.question_path
    length_gap = args.length_gap
    length_ratio = args.length_ratio
    output_path = args.output_path

    if graph_info_dir_1 is None: 
        if rag_name_1 == 'NaiveRAG':
            graph_info_dir_1 = f'{os.path.dirname(question_path)}/graph_info/LightRAG'
        else:
            graph_info_dir_1 = f'{os.path.dirname(question_path)}/graph_info/{rag_name_1}'
    if graph_info_dir_2 is None: 
        if rag_name_2 == 'NaiveRAG':
            graph_info_dir_2 = f'{os.path.dirname(question_path)}/graph_info/LightRAG'
        else:
            graph_info_dir_2 = f'{os.path.dirname(question_path)}/graph_info/{rag_name_2}'

    if output_path is None: 
        parent_dir = f'{os.path.dirname(question_path)}/answer'
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        output_path = f'{parent_dir}/{rag_name_1}_and_{rag_name_2}.jsonl'

    if length_gap is None and length_ratio is None:
        raise ValueError("length_gap and length_ratio cannot be None at the same time")

    question_list = []
    question_dataset = common_utils.load_jsonl_dataset(question_path)
    for item in question_dataset:
        question_list.append((item["question_id"], item["question"].strip()))
    
    exist_list = []
    if os.path.exists(output_path):
        exist_data = common_utils.load_jsonl_dataset(output_path)
        for item in exist_data:
            exist_list.append(item['question_id'])
    
    for q_id, q in question_list:
        if q_id in exist_list:
            continue

        ans1 = get_query(rag_name_1, dataset_name, graph_info_dir_1, q)
        ans2 = get_query(rag_name_2, dataset_name, graph_info_dir_2, q)

        ans1 = ans1.replace("\n", " ").strip()
        ans2 = ans2.replace("\n", " ").strip()

        ans1_length = len(ans1.split())
        ans2_length = len(ans2.split())

        print(f"{rag_name_1}-original-length:"+str(ans1_length))
        print(f"{rag_name_2}-original-length:"+str(ans2_length))


        if length_gap is not None:
            if(ans1_length > ans2_length + length_gap):
                ans2, ans2_length = generate_answer(rag_name_2, dataset_name, graph_info_dir_2, q, ans1_length, length_gap=length_gap)

                print(f"{rag_name_1}-original-length:"+str(ans1_length))
                print(f"{rag_name_2}-duiqi-length:"+str(ans2_length))


            elif(ans2_length > ans1_length + length_gap):
                ans1, ans1_length = generate_answer(rag_name_1, dataset_name, graph_info_dir_1, q, ans2_length, length_gap=length_gap)

                print(f"{rag_name_1}-duiqi-length:"+str(ans1_length))
                print(f"{rag_name_2}-original-length:"+str(ans2_length))


        elif length_ratio is not None:
            if(ans1_length > ans2_length * length_ratio):
                ans2, ans2_length = generate_answer(rag_name_2, dataset_name, graph_info_dir_2, q, ans1_length, length_ratio=length_ratio)

                print(f"{rag_name_1}-original-length:"+str(ans1_length))
                print(f"{rag_name_2}-duiqi-length:"+str(ans2_length))

            elif(ans2_length > ans1_length * length_ratio):
                ans1, ans1_length = generate_answer(rag_name_1, dataset_name, graph_info_dir_1, q, ans2_length, length_ratio=length_ratio)

                print(f"{rag_name_1}-duiqi-length:"+str(ans1_length))
                print(f"{rag_name_2}-original-length:"+str(ans2_length)) 
        else:
            pass


        data = {}
        data['question_id'] = q_id
        data["question"] = q
        data['answer'] = {rag_name_1: ans1, rag_name_2: ans2}
        is_remove = False

        if length_gap is not None:
            if abs(ans1_length - ans2_length) > length_gap: 
                is_remove = True
        elif length_ratio is not None:
            if float(abs(ans1_length - ans2_length)) > (length_ratio - 1) * min(ans1_length, ans2_length):
                is_remove = True
        
        data['is_remove'] = is_remove

        common_utils.append_jsonl(output_path, data)



def generate_answer(rag_name, dataset_name, graph_info_dir, q, target_length, length_gap=None, length_ratio=None):

    attempts = 0
    max_attempts = 5  
    answer = ''
    answer_length = 0
    
    while attempts < max_attempts:
        answer = get_query(rag_name, dataset_name, graph_info_dir, q, target_length)
        if not answer:  
            print(f"Warning: The {attempts+1} attempt returned an empty answer, continue retrying ...")
            attempts += 1
            continue

        answer = answer.replace("\n", " ").strip()
        answer_length = len(answer.split())
        
        # 根据条件判断长度是否符合要求
        if length_gap is not None:
            if abs(answer_length - target_length) <= length_gap:
                return answer, answer_length
        elif length_ratio is not None:
            if float(abs(answer_length - target_length)) <= (length_ratio - 1) * min(answer_length, target_length):
                return answer, answer_length
        else:
            return answer, answer_length
        
        attempts += 1
        
    print(f"Reaches the maximum number of retries ({attempts}) and returns the last generated answer.")
    return answer, answer_length


if __name__ == "__main__":
    main()


