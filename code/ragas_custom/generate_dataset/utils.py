from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import pandas as pd

def classify_language(text):
    english_count = len(re.findall(r'[a-zA-Z]', text))
    korean_count = len(re.findall(r'[가-힣]', text))

    if english_count >= korean_count:
        return 'english'

    return 'korean'


def translate(testset_df):
    template = """
        You are an expert translator specializing in English to Korean translation.

        Translate the following English text into natural Korean.  
        Only output the translated Korean text.  
        If a term is a proper noun or a commonly used English term (e.g., "clean and jerk"), transliterate it into Korean and include the original English in parentheses.

        Text:  
        {input_text}
    """ 

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    testset_df['language_user_input'] = testset_df['user_input'].apply(lambda x : classify_language(x))
    testset_df['language_reference'] = testset_df['reference'].apply(lambda x : classify_language(x))

    user_data = testset_df.loc[(testset_df['language_user_input'] == 'english'), 'user_input'].tolist()
    reference_data = testset_df.loc[(testset_df['language_reference'] == 'english'), 'reference'].tolist()

    translate_user = chain.batch(user_data, config={'max_concurrency': 5})
    translate_reference = chain.batch(reference_data, config={'max_concurrency': 5})

    testset_df.loc[(testset_df['language_user_input'] == 'english'), 'user_input'] = translate_user
    testset_df.loc[(testset_df['language_reference'] == 'english'), 'reference'] = translate_reference

    return testset_df.iloc[:, :4]


def make_chunk_dict(knowledge_graph, heading: str = 'heading1'):
    chunk_dict = {}
    chunk_id_dict = {}
    chunk_summary_embedding = {}
    for node in knowledge_graph.nodes:
        section = node.properties['document_metadata']['heading'][heading]
        chunk_id = node.properties['document_metadata']['chunk_id']
        summary_embedding = node.properties['summary_embedding']
        page_content = node.properties['page_content']

        chunk_dict[page_content] = section
        chunk_id_dict[page_content] = chunk_id
        chunk_summary_embedding[page_content] = summary_embedding
    return chunk_dict, chunk_id_dict, chunk_summary_embedding

def regular_expression(reference_contexts, chunk_dict):
    if len(reference_contexts) == 1:
        return [chunk_dict[reference_contexts[0]]]
    else:
        return [chunk_dict[re.sub(r"<\d+-hop>\n\n", "", text)] for text in reference_contexts] 
    
    
def analyze_synthetic_data(kg, testset_df, heading:str = 'heading1'):
    chunk_dict, chunk_id_dict, chunk_summary_embedding = make_chunk_dict(kg, heading)
    testset_df['reference_contexts_section'] = testset_df['reference_contexts'].apply(lambda x : regular_expression(x, chunk_dict))

    return testset_df