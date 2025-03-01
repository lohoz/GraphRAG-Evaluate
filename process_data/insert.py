import json
import time
import os
from rag_methods import get_rag
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="insert"
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--rag_name",
        type=str,
        default=None
    )

    parser.add_argument(
        "--graph_info_dir",
        type=str,
        default=None
    )

    return parser.parse_args()



def insert(rag, file_path):
    if file_path.endswith(".json"):
        with open(file_path, mode="r") as f:
            contexts = json.load(f)
    else:
        with open(file_path, mode="r", encoding="utf-8-sig") as f:
            contexts = f.read()

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")



def main():
    args = parse_args()
    dataset_path = args.dataset_path
    dataset_name = args.dataset_name
    rag_name = args.rag_name
    graph_info_dir = args.graph_info_dir

    if graph_info_dir is None:
        graph_info_dir = f'./datasets/{dataset_name}/graph_info/{rag_name}'

    if not os.path.exists(graph_info_dir):
        os.makedirs(graph_info_dir, exist_ok=True)

    rag = get_rag(rag_name, dataset_name, graph_info_dir)

    insert(rag, dataset_path)


if __name__ == "__main__":
    main()