import os
from utils.common_utils import common_utils
import argparse
import networkx as nx
import random
from openai import OpenAI
from prompt import PROMPTS
import re
from env import BASEURL, APIKEY


client = OpenAI(
    base_url=BASEURL,
    api_key=APIKEY,
    timeout=120
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Get question"
    )

    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=None, 
    )

    parser.add_argument(
        "--LightRAG_graph_info_dir", 
        type=str, 
        default=None, 
    )

    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None, 
        help="output path"
    )

    return parser.parse_args()

def get_entity_questions(info: str):
    sys_prompt_temp = PROMPTS["get_entity_questions"]
    sys_prompt = sys_prompt_temp.format(
        entity_data=info
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sys_prompt}],
    )
    entity_response = response.choices[0].message.content

    return entity_response

def get_relation_questions(info: str):
    sys_prompt_temp = PROMPTS["get_relation_questions"]
    sys_prompt = sys_prompt_temp.format(
        relation_data=info
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sys_prompt}],
    )
    relation_response = response.choices[0].message.content

    return relation_response

def get_subgraph_questions(nodes: str , edges: str):
    sys_prompt_temp = PROMPTS["get_subgraph_questions"]
    sys_prompt = sys_prompt_temp.format(
        entity_data=nodes,
        relation_data=edges
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sys_prompt}],
    )
    subgraph_response = response.choices[0].message.content

    return subgraph_response

def random_walk(graph, start_node, walk_length):
    visited_nodes = []  
    visited_edges = []  
    
    current_node = start_node
    visited_nodes.append(current_node)
    
    for _ in range(walk_length):

        neighbors = list(graph.neighbors(current_node))
        if not neighbors:
            break  

        next_node = random.choice(neighbors)

        visited_edges.append((current_node, next_node))

        current_node = next_node
        visited_nodes.append(current_node)
    
    return visited_nodes, visited_edges


def get_edge_data(edges_dict, src, dst):

    if (src, dst) in edges_dict:
        return edges_dict[(src, dst)]

    elif (dst, src) in edges_dict:
        return edges_dict[(dst, src)]

    else:
        return None
    

def deduplicate_edges(edges):

    unique_edges = set()
    for u, v in edges:
        if u < v:
            unique_edges.add((u, v))
        else:
            unique_edges.add((v, u))
    return list(unique_edges)


def main():
    args = parse_args()
    dataset_name = args.dataset_name
    lightrag_graph_info_dir = args.LightRAG_graph_info_dir
    output_path = args.output_path

    dir_path = f'./datasets/{dataset_name}'

    # os.makedirs(dir_path, exist_ok=True)

    if lightrag_graph_info_dir is None:
        lightrag_graph_info_dir = f"{dir_path}/graph_info/LightRAG/graph_chunk_entity_relation.graphml"

    if output_path is None:
        output_path = f"{dir_path}/questions.jsonl"

    print(f"Loading graph from {lightrag_graph_info_dir} ...")
    graph = nx.read_graphml(lightrag_graph_info_dir)
    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    adj_matrix = nx.to_numpy_array(graph)

    nodes_dict = {node_id: node_data for node_id, node_data in nodes}

    edges_dict = {(src, dst): edge_data for src, dst, edge_data in edges}

    selected_entities = random.sample(nodes, 50)

    nodes_info = ""
    for selected_entity in selected_entities:
        nodes_info += selected_entity[0] + ","
        nodes_info += selected_entity[1]['description'] + "\n"
    
    node_level_question = get_entity_questions(nodes_info)

    print("node_level_question done ...")
    
    # get edge level question
    selected_edges = random.sample(edges, 50)

    edges_info = ""
    for edge_info in selected_edges:
        edges_info += edge_info[0] + "," + edge_info[1] + "," + edge_info[2]['description'] + "\n"
    
    edge_level_question = get_relation_questions(edges_info)

    print("edge_level_question done ...")


    #get subgraph level question
    subgraph_entity_info = ""
    subgraph_relation_info = ""

    flag = True
    while flag:
        start_node = random.choice(nodes)

        walk_length = 100

        visited_nodes, visited_edges = random_walk(graph , start_node[0], walk_length)

        deduplicated_edges = deduplicate_edges(visited_edges)
        if len(deduplicated_edges) > 50:
            flag = False

    for node in visited_nodes:
        subgraph_entity_info += node +" , "+ nodes_dict[node]['description'] + "\n"
    for edge in deduplicated_edges:
        info = get_edge_data(edges_dict, edge[0], edge[1])
        subgraph_relation_info += edge[0] + " , " + edge[1] +" , " + info['description'] + "\n"

    subgraph_level_question = get_subgraph_questions(subgraph_entity_info , subgraph_relation_info)

    print("subgraph_level_question done ...")

    all_questions = node_level_question + '\n' + edge_level_question + '\n' + subgraph_level_question + '\n' 

    questions = re.findall(r'- Question \d+: (.+)', all_questions)


    for i, question in enumerate(questions):
        question_obj = {
            "question_id": i,
            "question": question.strip()
        }
        common_utils.append_jsonl(output_path, question_obj)

    print(f"The questions have been successfully written to the file: {output_path}") 
    


if __name__ == "__main__":
    main()