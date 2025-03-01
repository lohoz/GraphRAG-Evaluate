import asyncio
from openai import OpenAI
import json
import re
from typing import Union
from collections import Counter, defaultdict
import warnings
import networkx as nx
import numpy as np
import io
import csv
from .utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
    process_combine_contexts,
    locate_json_string_body_from_string,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS

base_url = ""
api_key = ""

def set_base_url(url):
    global base_url
    base_url = url

def set_api_key(key):
    global api_key
    api_key = key


client = OpenAI(
base_url=base_url,
api_key=api_key,
timeout=120
)

routecount = 0



async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = []
    seen = set()

    for this_edges in all_related_edges:
        for e in this_edges:
            sorted_edge = tuple(sorted(e))
            if sorted_edge not in seen:
                seen.add(sorted_edge)
                all_edges.append(sorted_edge)

    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )

    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )

    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=1000,
    )
    return all_edges_data


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )

    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # print(all_one_hop_nodes_data)
    # exit()

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    # print(all_one_hop_text_units_lookup)
    # exit()

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                    "relation_counts": 0,
                }

            if this_edges:
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        all_text_units_lookup[c_id]["relation_counts"] += 1

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    # print(all_text_units)
    # exit()

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units





async def _build_zqm_entity_query_context(
    query,
    workingdir,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    user_query,
):
    routeflag = True
    #对于entity keywords中的实体，分别检索其top2相似实体
    query = query.upper()
    # print(query)
    entity_keywords_list = [element.strip() for element in query.split(",")]
    filter_entities = []
    #按照对应的keyword进行区分
    entity_match_list = []
    entity_match =[]
    for entity in entity_keywords_list:
        results_top10 = await entities_vdb.query(entity, top_k=6)
        # print(results_top5)
        #调用大模型，留下与问题相关的entity。
        #首先提取出5个entity对应的entity name
        top10_name = []
        for r in results_top10:
            top10_name.append(r["entity_name"])
        entity_string = ', '.join(top10_name)
        # print("entitys:"+entity_string)

        #现在调用大模型，留下与问题相关的entity
        entity_filter_prompt_temp = PROMPTS["filter_entity"]
        subgraph_prompt = entity_filter_prompt_temp.format(
            entities = entity_string, user_query = user_query
        )
        # 创建聊天完成
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": subgraph_prompt}]
        )
        entity_response = response.choices[0].message.content
        #万一没有相关的
        if(entity_response == None):
            entity_match.extend(results_top10)
        else:
            response_entity = [element.strip() for element in entity_response.split(",")]
            filtered_response_entity = [r for r in response_entity if r in top10_name]
            if filtered_response_entity == None :
                entity_match.extend(results_top10)
            else:
                for r in filtered_response_entity:
                    filter_entities.append(r)
                # 我们可以使用列表推导式来创建一个新的列表，只包含需要保留的元素
                results_top10 = [r for r in results_top10 if r["entity_name"] in filter_entities]
                entity_match_list.append(filtered_response_entity)
                entity_match.extend(results_top10)
    #     # 打印结果，查看哪些元素被保留了
    #     print(filter_entities)
    # print(entity_match)

    #此时entity_match中存储了全部filter后的entities。



    if not len(entity_match):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_match]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_match]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(entity_match, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    #TODO：这里需要调整为entities间的路径信息
    routenum , entities_route_csv = await _find_route_between_entities(
        node_datas, query_param, knowledge_graph_inst, entity_match_list,workingdir , query_param.route_ranking, user_query
    )

    if(routenum < 4):
        use_relations = await _find_most_related_edges_from_entities(
            node_datas, query_param, knowledge_graph_inst
        )
    
        relations_section_list = [
            ["id", "source", "target", "description", "keywords", "weight", "rank"]
        ]
        for i, e in enumerate(use_relations):
            relations_section_list.append(
                [
                    i,
                    e["src_tgt"][0],
                    e["src_tgt"][1],
                    e["description"],
                    e["keywords"],
                    e["weight"],
                    e["rank"],
                ]
            )
        relations_context = list_of_list_to_csv(relations_section_list)

    else:
        use_relations=[]
        relations_context = ""

    logger.info(
        f"Local query uses {len(node_datas)} entites,{len(use_relations)} relationships, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entity_match_list, routenum, f"""
-----Entities-----
```csv
{entities_context}
```
-----Routes-----
```csv
{entities_route_csv}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```

"""


async def _build_zqm_entity_query_context_new(
    query,
    workingdir,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    user_query,
):
    routeflag = True
    #对于entity keywords中的实体，分别检索其top2相似实体
    query = query.upper()
    # print(query)
    entity_keywords_list = [element.strip() for element in query.split(",")]
    filter_entities = []
    #按照对应的keyword进行区分
    entity_match_list = []
    entity_match =[]
    for entity in entity_keywords_list:
        results_top10 = await entities_vdb.query(entity, top_k=6)
        # print(results_top5)
        #调用大模型，留下与问题相关的entity。
        #首先提取出5个entity对应的entity name
        top10_name = []
        for r in results_top10:
            top10_name.append(r["entity_name"])
        entity_string = ', '.join(top10_name)
        # print("entitys:"+entity_string)

        #现在调用大模型，留下与问题相关的entity
        entity_filter_prompt_temp = PROMPTS["filter_entity"]
        subgraph_prompt = entity_filter_prompt_temp.format(
            entities = entity_string, user_query = user_query
        )
        # 创建聊天完成
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": subgraph_prompt}]
        )
        entity_response = response.choices[0].message.content
        #万一没有相关的
        if(entity_response == None):
            entity_match.extend(results_top10)
        else:
            response_entity = [element.strip() for element in entity_response.split(",")]
            filtered_response_entity = [r for r in response_entity if r in top10_name]
            if filtered_response_entity == None :
                entity_match.extend(results_top10)
            else:
                for r in filtered_response_entity:
                    filter_entities.append(r)
                # 我们可以使用列表推导式来创建一个新的列表，只包含需要保留的元素
                results_top10 = [r for r in results_top10 if r["entity_name"] in filter_entities]
                entity_match_list.append(filtered_response_entity)
                entity_match.extend(results_top10)
    #     # 打印结果，查看哪些元素被保留了
    #     print(filter_entities)
    # print(entity_match)

    #此时entity_match中存储了全部filter后的entities。

    if not len(entity_match):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_match]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_match]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(entity_match, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.

    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    #TODO：这里需要调整为entities间的路径信息
    entities_route_csv ,relation_csv , node_csv , routenum = await _find_route_between_entities_new(
        node_datas, query_param, knowledge_graph_inst, entity_match_list,workingdir
    )

    if(entities_route_csv == "no route detected."):
        routeflag = False

    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    entities_context = entities_context + node_csv
    
    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return entity_match_list, routenum, f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relation_csv}
```
-----Sources-----
```csv
{text_units_context}
```

"""


#因为node_data中已经包含的始末节点的信息，因此此处不再重复求解
async def _find_route_between_entities_new(    
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entity_match_list:list[list],
    workingdir:str
    ):
    route_list = []
    # print(workingdir)
    graph_dir = workingdir+"/graph_chunk_entity_relation.graphml"
    # print(graph_dir)
    graph = nx.read_graphml(graph_dir)
    nodes = list(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph)
    #TODO：st1:对于entity_match_list中的实体，执行组合求路径集合，并将路径集合进行存储
    import itertools
    
    if(len(entity_match_list) < 2):
        return "no route detected." , "" , "", 0
    # 使用 itertools.product 来找到所有不同列表之间的元素组合
    all_combinations = list(itertools.product(*entity_match_list))

    # 打印所有不同列表之间的元素组合
    for combination in all_combinations:
        src = nodes.index(combination[0])
        dst = nodes.index(combination[1])
        route = find_all_paths(adj_matrix, src, dst, 3)
        if(len(route) != 0):
            route_list.append(route)
    flattened_route_list = [item for sublist in route_list for item in sublist]
    routenum = len(flattened_route_list)
    if(routenum == 0):
        return "no route detected." , "" , "" , 0
    # print(flattened_route_list)

    #TODO：st2:对于求解得到的路径，找到其对应的node以及edge并以文本方式返回（自定义格式，包含text）
    edge_datas = []
    node_datas = []

    for route in flattened_route_list:
        for index, nodeindex in enumerate(route):
            if(index == 0):
                last_node = nodeindex
            elif(index == len(route) - 1):
                src = last_node
                dst = nodeindex
                #下面找边
                route_edge = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(nodes[src],nodes[dst])]
                )
                dic2 = {"src_id": nodes[src], "tgt_id": nodes[dst], "description": route_edge[0]["description"],"keywords":route_edge[0]["keywords"],"weight":route_edge[0]["weight"],"rank": 0}
                edge_datas.append(dic2)

            else:
                src = last_node
                dst = nodeindex
                dst_data = await asyncio.gather(
                    *[knowledge_graph_inst.get_node(nodes[dst])]
                )
                dic1 = { "entity_name": nodes[dst], "entity_type":dst_data[0]['entity_type'],"description": dst_data[0]['description']}
                node_datas.append(dic1)
                #下面找边
                route_edge = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(nodes[src],nodes[dst])]
                )
                dic2 = {"src_id": nodes[src], "tgt_id": nodes[dst], "description": route_edge[0]["description"],"keywords":route_edge[0]["keywords"],"weight":route_edge[0]["weight"],"rank": 0}
                edge_datas.append(dic2)
                last_node = nodeindex
    
    all_edges_data = zqm_truncate_list_by_token_size(
        edge_datas,
        max_token_size=3000,
    )


    all_entities_units = zqm_truncate_list_by_token_size(
        node_datas,
        max_token_size=2000,
    )

    relations_section_list = [["id", "source", "target", "description", "keywords", "weight", "rank"]]
    for i, e in enumerate(all_edges_data):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    
    
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(all_entities_units):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
            ]
        )



    relations_context = list_of_list_to_csv(relations_section_list)
    entities_context = list_of_list_to_csv(entites_section_list)

    return "", relations_context , entities_context , routenum


def zqm_truncate_list_by_token_size(list_data: list, max_token_size: int):
    """Truncate a list of data by token size"""
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        if(i != 0):
            tokens += len(encode_string_by_tiktoken(data["description"]))
            if tokens > max_token_size:
                return list_data[:i]
    return list_data




#因为node_data中已经包含的始末节点的信息，因此此处不再重复求解
async def _find_route_between_entities(    
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
    entity_match_list:list[list],
    workingdir:str,
    route_ranking:bool,
    user_query:str
    ):
    route_list = []
    # print(workingdir)
    graph_dir = workingdir+"/graph_chunk_entity_relation.graphml"
    # print(graph_dir)
    graph = nx.read_graphml(graph_dir)
    nodes = list(graph.nodes())
    adj_matrix = nx.to_numpy_array(graph)
    #TODO：st1:对于entity_match_list中的实体，执行组合求路径集合，并将路径集合进行存储
    import itertools
    
    if(len(entity_match_list) < 2):
        routenum = 0
        return routenum , "no route detected."
    # 使用 itertools.product 来找到所有不同列表之间的元素组合
    all_combinations = list(itertools.product(*entity_match_list))

    # 打印所有不同列表之间的元素组合
    for combination in all_combinations:
        src = nodes.index(combination[0])
        dst = nodes.index(combination[1])
        route = find_all_paths(adj_matrix, src, dst, 3)
        if(len(route) != 0):
            route_list.append(route)
    flattened_route_list = [item for sublist in route_list for item in sublist]
    routenum = len(flattened_route_list)
    if(routenum == 0):
        return routenum , "no route detected."
    # print(flattened_route_list)

    #TODO：st2:对于求解得到的路径，找到其对应的node以及edge并以文本方式返回（自定义格式，包含text）
    important_route =""
    route_information = ""

    if(route_ranking):

        rank_route_information = {
            "route information": [],
            "score": []
        }

        #计算user_query的embedding
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=120
        )
        # 创建聊天完成
        chat_completion = client.embeddings.create(
            model="text-embedding-3-small",
            input=user_query,
            encoding_format="float"
        )
        query_embeddings = (chat_completion.data[0].embedding)
        query_embeddings = np.array(query_embeddings)

    for route in flattened_route_list:
        if(len(route) == 2):
            src = route[0]
            dst = route[1]

            dst_data = await asyncio.gather(
                *[knowledge_graph_inst.get_node(nodes[dst])]
            )
            for i, n in enumerate(dst_data):
                dst_node_data = (
                    [
                        nodes[dst],
                        n.get("entity_type", "UNKNOWN"),
                        n.get("description", "UNKNOWN"),
                    ]
                )

            #下面找边
            route_edge = await asyncio.gather(
                *[knowledge_graph_inst.get_edge(nodes[src],nodes[dst])]
            )
            route_edge_data = {
                "source":str(nodes[src]) , "target":nodes[dst], "description": route_edge[0]["description"], "keyword": route_edge[0]["keywords"]
            }

            # print(route_edge_data)

            out_dst_data = to_csv(dst_node_data)
            route_data = to_csv(list(route_edge_data.items()))

            direct_route =  "source node:" + str(nodes[src]) + "   edge information:  " + route_data + "   end node:  "+ str(nodes[dst])

            if(route_ranking):
                #计算路径embedding：
                chat_completion = client.embeddings.create(
                    model="text-embedding-3-small",
                    input = direct_route,
                    encoding_format="float"
                )
                route_embeddings = (chat_completion.data[0].embedding)
                route_embeddings = np.array(route_embeddings)
                score = cosine_similarity(query_embeddings, route_embeddings)
                rank_route_information["route information"].append(direct_route)
                rank_route_information["score"].append(score)
            else:
                rank_route_information["route information"].append(direct_route)
                rank_route_information["score"].append(1)


        else:
            this_route = ""
            this_route_imformation = ""
            for index, nodeindex in enumerate(route):
                if(index == 0):
                    last_node = nodeindex
                    this_route = this_route + "Start from:" + nodes[nodeindex] + "  "
                elif(index == len(route) - 1):
                    this_route = this_route + "End with:" + nodes[nodeindex]

                    src = last_node
                    dst = nodeindex

                    #下面找边
                    route_edge = await asyncio.gather(
                        *[knowledge_graph_inst.get_edge(nodes[src],nodes[dst])]
                    )
                    route_edge_data = {
                        "source":str(nodes[src]) , "target":nodes[dst], "description": route_edge[0]["description"], "keyword": route_edge[0]["keywords"]
                    }
                    route_data = to_csv(list(route_edge_data.items()))

                    this_route_imformation += "source node: " + str(nodes[src]) + "  edge information: " + route_data + "  end node: "+ str(nodes[dst]) +"\n"

                else:
                    this_route = this_route + "Passing node:" + nodes[nodeindex] + "  "
                    src = last_node
                    dst = nodeindex

                    dst_data = await asyncio.gather(
                        *[knowledge_graph_inst.get_node(nodes[dst])]
                    )
                    for i, n in enumerate(dst_data):
                        dst_node_data = (
                            [
                                nodes[dst],
                                n.get("entity_type", "UNKNOWN"),
                                n.get("description", "UNKNOWN"),
                            ]
                        )

                    #下面找边
                    route_edge = await asyncio.gather(
                        *[knowledge_graph_inst.get_edge(nodes[src],nodes[dst])]
                    )
                    route_edge_data = {
                        "source":str(nodes[src]) , "target":nodes[dst], "description": route_edge[0]["description"], "keyword": route_edge[0]["keywords"]
                    }

                    out_dst_data = to_csv(dst_node_data)
                    route_data = to_csv(list(route_edge_data.items()))

                    this_route_imformation += "source node: " + str(nodes[src]) + "  edge information: " + route_data + "  next node information: "+ out_dst_data +"\n"
                    last_node = nodeindex
            route_information = this_route + "\n" + this_route_imformation
            
            if(route_ranking):
                #计算路径embedding：
                chat_completion = client.embeddings.create(
                    model="text-embedding-3-small",
                    input = route_information,
                    encoding_format="float"
                )
                route_embeddings = (chat_completion.data[0].embedding)
                route_embeddings = np.array(route_embeddings)
                score = cosine_similarity(query_embeddings, route_embeddings)
                rank_route_information["route information"].append(route_information)
                rank_route_information["score"].append(score)
            else:
                rank_route_information["route information"].append(route_information)
                rank_route_information["score"].append(0.5)


    # print(route_information)
    # print(len(flattened_route_list))
    # print(len(rank_route_information["route information"]))
    # exit()
    # 使用 zip 将两个列表组合在一起，并根据 score 降序排序
    sorted_pairs = sorted(zip(rank_route_information["score"], rank_route_information["route information"]), reverse=True)

    # 拆分排序后的结果
    sorted_scores, sorted_route_info = zip(*sorted_pairs)

    # # 更新字典
    # rank_route_information["score"] = list(sorted_scores)
    # rank_route_information["route information"] = list(sorted_route_info)

    out_route_csv = "\n".join(sorted_route_info)
    # with open("./111111111.txt" , "w" )as file:
    #     file.write(out_route_csv)
    # exit()

    out_route_csv = truncate_csv_by_token_size(out_route_csv,max_token_size=query_param.max_token_for_route_unit)
    return  routenum, out_route_csv


def truncate_csv_by_token_size(data:str, max_token_size: int):
    # 编码字符串
    encoded_tokens = encode_string_by_tiktoken(data)

    # 截断token列表到最大token大小
    truncated_tokens = encoded_tokens[:max_token_size]

    # 将截断后的token列表解码回字符串
    truncated_entity_level_routes = decode_tokens_by_tiktoken(truncated_tokens)

    # print(truncated_entity_level_routes)
    # exit()

    return truncated_entity_level_routes



def dfs_paths(adj, start, end, path, all_paths, visited, max_hops, current_hops=0):
    global routecount
    path.append(start)
    if start == end:
        if current_hops < max_hops:  # 只有当当前跳数小于最大跳数时才添加路径
            all_paths.append(list(path))
            print(routecount)
            routecount = routecount + 1
    else:
        for next_node in range(len(adj)):
            if adj[start][next_node] != 0 and next_node not in visited:
                visited.add(next_node)
                # 只有当当前跳数小于最大跳数时才递归
                if current_hops < max_hops:
                    dfs_paths(adj, next_node, end, path, all_paths, visited, max_hops, current_hops + 1)
                visited.remove(next_node)  # 回溯
    path.pop()

def find_all_paths(adj, start, end, max_hops):
    global routecount
    all_paths = []
    visited = set()  # 用于跟踪已访问的节点
    visited.add(start)  # 从起点开始
    dfs_paths(adj, start, end, [], all_paths, visited, max_hops)
    routecount = 0
    return all_paths

def to_csv(data):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(data)
    data = output.getvalue()
    data = data.replace("'source',","source:")
    data = data.replace("'target',","target:")
    data = data.replace("'description',","description:")
    data = data.replace("'keyword',","keyword:")
    data = data.replace("(" , "")
    data = data.replace(")","")
    
    return data


async def _build_zqm_relation_query_context(
    entity_match_list:list[list],   
    routenum, 
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    ):

    results = await relationships_vdb.query(keywords, top_k=40)
    flat_list = [item for sublist in entity_match_list for item in sublist]

    if(routenum > 5):
        #先去前五重要的边进行直接保留
        result_top_5 = results[:5]
        rest_result = results[5:]
    elif(routenum > 0):
        result_top_5 = results[:10]
        rest_result = results[10:]
    else:
        result_top_5 = results[:15]
        rest_result = results[15:]

    # 过滤 results，仅保留 src_id 或 tgt_id 在 entity_match_list 中的项
    filtered_results = [
        r for r in rest_result 
        if r["src_id"] in flat_list or r["tgt_id"] in flat_list
    ]

    # # 打印过滤后的结果
    # print(result_top_5)
    # print("-----------------------------------------------------------")
    # print(filtered_results)
    
    final_result = result_top_5 + filtered_results

    if not len(final_result):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in final_result]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in final_result]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(final_result, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    if(routenum <= 5):
        use_entities = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        )

        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(use_entities):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_of_list_to_csv(entites_section_list)
    else:
        use_entities = {}
        entities_context = ""
    logger.info(
        f"Global query uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)




    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)


    return f"""
        -----Entities-----
        ```csv
        {entities_context}
        ```
        -----Routes-----
        ```csv
        
        ```
        -----Relationships-----
        ```csv
        {relations_context}
        ```
        -----Sources-----
        ```csv
        {text_units_context}
        ```
        """


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = []
    seen = set()

    for e in edge_datas:
        if e["src_id"] not in seen:
            entity_names.append(e["src_id"])
            seen.add(e["src_id"])
        if e["tgt_id"] not in seen:
            entity_names.append(e["tgt_id"])
            seen.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units





async def _build_zqm_relation_query_context_new(
    entity_match_list:list[list],   
    routenum, 
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    ):

    results = await relationships_vdb.query(keywords, top_k=40)
    flat_list = [item for sublist in entity_match_list for item in sublist]

    if(routenum > 5):
        #先去前五重要的边进行直接保留
        result_top_5 = results[:5]
        rest_result = results[5:]
    elif(routenum > 0):
        result_top_5 = results[:10]
        rest_result = results[10:]
    else:
        result_top_5 = results[:15]
        rest_result = results[15:]

    # 过滤 results，仅保留 src_id 或 tgt_id 在 entity_match_list 中的项
    filtered_results = [
        r for r in rest_result 
        if r["src_id"] in flat_list or r["tgt_id"] in flat_list
    ]
    
    final_result = result_top_5 + filtered_results

    if not len(final_result):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in final_result]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in final_result]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(final_result, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )

    if( routenum <= 5):
        use_entities = await _find_most_related_entities_from_relationships(
            edge_datas, query_param, knowledge_graph_inst
        )

        entites_section_list = [["id", "entity", "type", "description", "rank"]]
        for i, n in enumerate(use_entities):
            entites_section_list.append(
                [
                    i,
                    n["entity_name"],
                    n.get("entity_type", "UNKNOWN"),
                    n.get("description", "UNKNOWN"),
                    n["rank"],
                ]
            )
        entities_context = list_of_list_to_csv(entites_section_list)
    else:
        use_entities = {}
        entities_context = ""
    logger.info(
        f"Global query uses {len(use_entities)} entities, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)




    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)


    return f"""
        -----Entities-----
        ```csv
        {entities_context}
        ```
        -----Relationships-----
        ```csv
        {relations_context}
        ```
        -----Sources-----
        ```csv
        {text_units_context}
        ```
        """


def cosine_similarity(A, B):
    # 计算点积
    dot_product = np.dot(A, B)
    
    # 计算模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    
    # 计算余弦相似度
    similarity = dot_product / (norm_A * norm_B)
    
    return similarity