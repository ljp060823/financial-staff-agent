import warnings
warnings.filterwarnings('ignore')
import re
import traceback
import json
import random
import numpy as np
from pymilvus import connections, utility, Collection
from langchain import PromptTemplate
from openai import OpenAI
from py2neo import Graph
from model import RagEmbedding, RagLLM, QwenLLM
from prompt_cfg import rule_template, finance_template, keyword_prompt


llm = RagLLM()
embedding_model = RagEmbedding()
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_db"
connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
zhidu_db = Collection(COLLECTION_NAME)

graph = Graph("bolt://127.0.0.1:7687", user='neo4j', password='Ljp13249158398@',name='neo4j')

def run_chat(prompt, history=[]):
    client = OpenAI(
        base_url='http://localhost:8001/v1/',
        api_key='/data/qwen2.5-7b',
    )

    history_msg = []
    for idx, msg in enumerate(history):
        if idx == 0:
            continue
        history_msg.append({"role": "user", "content": msg[0]})
        history_msg.append({"role": "assistant", "content": msg[1]})
    # print(history_msg)

    chat_completion = client.chat.completions.create(
        messages=history_msg+[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        max_tokens=4096, 
        stream=True,    
        model='qwen2:72b',
        temperature=0.1,  
        top_p=0.9,
    )
    return chat_completion

def run_rag_pipline(query, context_query, k=3, context_query_type="query", 
                    stream=True, prompt_template=rule_template,
                    temperature=0.1):
    
    if context_query_type == "doc":
        related_docs = context_query if isinstance(context_query, list) else [context_query]
    
    elif context_query_type == "vector":
        query_vector = context_query
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = zhidu_db.search(      
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["page_content", "metadata"]
        )
        related_docs = _milvus_hits_to_docs(results)
    
    else:
        embed_func = embedding_model.get_embedding_fun()
        query_vector = embed_func(context_query)
        
   
        if isinstance(query_vector, (list, tuple)) and len(query_vector) > 0:
            if isinstance(query_vector[0], (list, tuple, np.ndarray)):
                query_vector = query_vector[0]
        
        # Milvus 搜索
        search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        results = zhidu_db.search(       
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["page_content", "metadata"]
        )
        related_docs = _milvus_hits_to_docs(results)

    context = "\n".join([f"上下文{i+1}: {doc.page_content} \n" 
                        for i, doc in enumerate(related_docs)])

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_template,
    )
    llm_prompt = prompt.format(question=query, context=context)
    
    if stream:
        response = llm(llm_prompt, stream=True)
        return (response, context)
    else:
        response = llm(llm_prompt, stream=False, temperature=temperature)
        return (response, context)



def _milvus_hits_to_docs(results):
    """Milvus search 结果转为与 LangChain Document 结构一致的假对象"""
    related_docs = []
    for hits in results:
        for hit in hits:
            related_docs.append(type('Doc', (), {
                'page_content': hit.entity.get('page_content', ''),
                'metadata': hit.entity.get('metadata', {})
            })())
    return related_docs

def parse_query(query, max_keywords=3):
    
    prompt_template = PromptTemplate(
                input_variables=["query_str", "max_keywords"],
                template=keyword_prompt,
            )
    
    final_prompt = prompt_template.format(max_keywords='3',
                                          query_str=query)
    response = llm(final_prompt)
    keywords = response.split('\n')[0].split('^')
    return keywords

def get_node(keyword, node_type):
    query = f"""
    MATCH (n:{node_type})
    where n.name CONTAINS "{keyword}"
    RETURN n.name as name
    """
    fetch_node = None
    print(query)
    results = graph.run(query)
    
    for record in results:
        return record['name']
    return fetch_node

def gen_contexts(investor_condition, 
                 company_condition, 
                 even_type_condition,
                 query_level=1,
                 exclude_content=False):
    if query_level == 1:
        query = f"""
        MATCH (i:Investor)-[:INVEST]->(c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {investor_condition} {company_condition} {even_type_condition}
        RETURN i.name as investor,  c.name as company_name, e.name as even_type, r as relation
        """
    else:
        query = f"""
        MATCH (c:Company)-[r:HAPPEN]->(e)
        WHERE 1=1 {investor_condition} {company_condition} {even_type_condition}
        RETURN c.name as company_name, e.name as even_type, r as relation
        """
    print(query)
    results = graph.run(query)
    contexts = []
    for record in results:
        context = ''
        record = dict(record)
        if 'investor' in record:
            context += f"{record['investor']} 投资了 {record['company_name']} \n"
        context = context + f"{record['company_name']} 发生了 {record['even_type']} \n 详细如下："
        for key, value in dict(record['relation']).items():
            if exclude_content:
                if key in ["title", "content"]:
                    continue
            context = context + f"\n  {key}: {value}"

        contexts.append(context)
    return contexts

def get_even_detail(keyword, exclude_content=False):
    investor = get_node(keyword, "Investor")
    company = get_node(keyword, "Company")
    even_type = get_node(keyword, "EventType")
    
    investor_condition = ""
    company_condition = ""
    even_type_condition = ""
    if investor:
        investor_condition = f' and i.name = "{investor}"'
    if company:
        company_condition = f' and c.name = "{company}"'
    if even_type:
        even_type_condition = f' and e.name = "{even_type}"'
        
    print(f"investor={investor_condition} company={company_condition} even_type={even_type_condition}")
    if investor_condition or company_condition or even_type_condition:
        contexts = gen_contexts(investor_condition, 
                                company_condition, 
                                even_type_condition,
                                query_level=1,
                                exclude_content=exclude_content)
        if len(contexts) == 0:
            contexts = gen_contexts(investor_condition, 
                                    company_condition, 
                                    even_type_condition,
                                    query_level=2,
                                    exclude_content=exclude_content)
        return contexts
    else:
        return []

def graph_rag_pipline(query, exclude_content=True, stream=True, temperature=0.1):

    keywords = parse_query(query, max_keywords=3)
    contexts = []
    ignore_words = ['公司', '分析', '投资']
    for keyword in keywords:
        if keyword in ignore_words:
            continue
        contexts.extend(get_even_detail(keyword=keyword, exclude_content=exclude_content))
    
    prompt = PromptTemplate(
                        input_variables=["question", "context"],
                        template=finance_template,)
    context = "\n========================\n".join(contexts)
    llm_prompt = prompt.format(question=query, context=context)
    print(llm_prompt)
    
    if stream:
        response = llm(llm_prompt, stream=True)
        return (response, context)
    else:
        response = llm(llm_prompt, stream=False, temperature=temperature)
        return (response, context)
        


def extract_tables_and_remainder(text):
    pattern = r'<table.*?>.*?</table>'
    tables = re.findall(pattern, text, re.DOTALL)
    remainder = re.sub(pattern, '', text, flags=re.DOTALL).strip()
    return tables, remainder


