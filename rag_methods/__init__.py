from .lightrag import (
    LightRAG,
    QueryParam as light_QueryParam
)
from .lightrag.llm import (
    gpt_4o_mini_complete as light_gpt_4o_mini_complete, 
    openai_embedding as light_openai_embedding,
    set_global_base_url as light_set_global_base_url,
    set_global_api_key as light_set_global_api_key
)
from .nano_graphrag import (
    GraphRAG as NanoRAG,
    QueryParam as nano_QueryParam
)
from .nano_graphrag._llm import (
    gpt_4o_mini_complete as nano_gpt_4o_mini_complete, 
    openai_embedding as nano_openai_embedding,
    set_global_base_url as nano_set_global_base_url,
    set_global_api_key as nano_set_global_api_key
)
from .fast_graphrag import (
    GraphRAG as FastRAG,
    QueryParam as fast_QueryParam
)
from .fast_graphrag._llm import (
    OpenAIEmbeddingService,
    OpenAILLMService
)
from ._domain import DOMAIN, ENTITY_TYPES, QUERIES
from env import BASEURL, APIKEY

base_url = BASEURL
api_key = APIKEY
ligth_api_key = ""
nano_api_key = ""
fast_api_key = ""

def set_base_url(url):
    global base_url
    base_url = url

def set_api_key(key):
    global api_key
    api_key = key

def set_light_api_key(key):
    global ligth_api_key
    ligth_api_key = key

def set_nano_api_key(key):
    global nano_api_key
    nano_api_key = key

def set_fast_api_key(key):
    global fast_api_key
    fast_api_key = key


def get_rag(rag_name, dataset_name, working_dir):
    if rag_name == "LightRAG" or rag_name == "NaiveRAG":
        light_set_global_base_url(base_url)
        if ligth_api_key == "":
            light_set_global_api_key(api_key)
        else:
            light_set_global_api_key(ligth_api_key)
        return LightRAG(working_dir=working_dir, 
                        embedding_func=light_openai_embedding, 
                        llm_model_func=light_gpt_4o_mini_complete,
                        enable_llm_cache=False
                    )
    
    elif rag_name == "NanoRAG":
        nano_set_global_base_url(base_url)
        if nano_api_key == "":
            nano_set_global_api_key(api_key)
        else:
            nano_set_global_api_key(nano_api_key)
        return NanoRAG(working_dir=working_dir, 
                       embedding_func=nano_openai_embedding, 
                       best_model_func=nano_gpt_4o_mini_complete, 
                       enable_naive_rag=True,
                       enable_llm_cache=False
                    )
    
    elif rag_name == "FastRAG":
        global fast_api_key
        fast_api_key = fast_api_key if fast_api_key != "" else api_key
        return FastRAG(working_dir=working_dir,        
                       domain=DOMAIN[dataset_name], 
                       entity_types=ENTITY_TYPES[dataset_name],
                       example_queries="\n".join(QUERIES),
                       config=FastRAG.Config(
                           embedding_service=OpenAIEmbeddingService(base_url=base_url, api_key=fast_api_key),
                           llm_service=OpenAILLMService(base_url=base_url, api_key=fast_api_key)
                        )
                    )
    else:
        raise ValueError(f"Invalid rag_name: {rag_name}")


def get_query(rag_name, dataset_name, working_dir, question, expected_length: int=0):
    rag = get_rag(rag_name, dataset_name, working_dir)
    result = ''
    if rag_name == "LightRAG":
        result = rag.query(str(question), param=light_QueryParam(mode="hybrid", expected_length=expected_length))

    elif rag_name == "NaiveRAG":
        result = rag.query(question, param=light_QueryParam(mode="naive", expected_length=expected_length))

    elif rag_name == "NanoRAG":
        result = rag.query(question, param=nano_QueryParam(mode="global", expected_length=expected_length))

    elif rag_name == "FastRAG":
        result = rag.query(question, params=fast_QueryParam(expected_length=expected_length)).response
    return result

