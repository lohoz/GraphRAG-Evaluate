import datasets
import json
import re
import os


class common_utils:

    @staticmethod
    def load_text(file_path: str):
        with open(file_path, "r") as f:
            content= f.readlines()
        return "\n".join(content)


    # Load the dataset from the JSONL file
    @staticmethod
    def load_jsonl_dataset(file_path: str):
        dataset = datasets.load_dataset("json", data_files=file_path, split="train")
        return dataset
    
    @staticmethod
    def extract_json(text: str):
        pattern = r'```json(.*?)```'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1)
        else:
            return text

    

    # Write a piece of data to the file
    @staticmethod
    def append_jsonl(file_path: str, data: str):
        with open(file_path, 'a') as file:
            line = json.dumps(data)
            file.write(line + '\n')
        
    
    # Write multiple pieces of data to a file
    @staticmethod
    def write_jsonl(file_path: str, data: str):
        with open(file_path, 'w', encoding='utf-8') as output_file:
            for item in data:
                output_file.write(json.dumps(item) + '\n')


    # Retrieve the number of words in the text
    @staticmethod
    def get_word_count(text: str):
        words = text.split()
        return len(words)


    # Merge JSONL files with the same format and return the merged dataset
    @staticmethod
    def merge_jsonl(file_path_1: str, file_path_2: str, merged_file_path: str):
        dataset_1 = common_utils.load_jsonl_dataset(file_path_1)
        dataset_2 = common_utils.load_jsonl_dataset(file_path_2)
        merged_dataset = datasets.concatenate_datasets([dataset_1, dataset_2])
        merged_dataset.to_json(merged_file_path)
        print(f'The number of merged data items is: {len(merged_dataset)}')
        return merged_dataset
    

    
    
    @staticmethod
    def save_to_file(file_name, content):
        with open(file_name, 'w') as f:
            f.write(content)


    @staticmethod
    def delete_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    @staticmethod
    def rename_rag(rag_name):
        return 'MGRAG' if rag_name == 'NanoRAG' else ('FGRAG' if rag_name == 'FastRAG' else rag_name)

