from langchain_openai import ChatOpenAI
from langchain_core.prompts import load_prompt
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from typing import List
from pydantic import BaseModel, Field

from utils import load_chat_prompt
from rateLimit import handle_rate_limits

@handle_rate_limits
def summary_chain(data_batches, model_name='gpt-4o-mini', current_api_key=None, max_concurrency=5) -> List[str]:
    '''
    document에 text가 없을 수 있지만 heading이 있어, 요약이 무조건 됨
    '''
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)

    stuff_prompt = load_prompt('../prompt/summary/stuff_20250317_02.yaml')
    stuff_chain = stuff_prompt | llm | StrOutputParser()

    stuff_result = stuff_chain.batch(data_batches, config={"max_concurrency": max_concurrency})

    return stuff_result

@handle_rate_limits
def image_information_chain(data_batches, id_list, model_name='gpt-4o', current_api_key=None, max_concurrency=5) -> List:
    '''
    이미지 정보 추출 체인
    '''
    class ImageInformationExtract(BaseModel):
        detail: str = Field(description='key insights and information visible in the IMAGE')
        hypotheticalQuestions: List[str] = Field(description="relevant hypothetical questions")
    
    parser = PydanticOutputParser(pydantic_object=ImageInformationExtract)
    prompt = load_chat_prompt('../prompt/information_image/image_20250304_01.yaml')
    chat_prompt = ChatPromptTemplate.from_messages(prompt['messages'])
    chat_prompt = chat_prompt.partial(format=parser.get_format_instructions(), language='KOREAN')

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)
    image_chain = chat_prompt | llm

    results = image_chain.batch(data_batches, config={"max_concurrency": max_concurrency})
    
    # 결과 처리
    image_result = []
    
    for idx, result in enumerate(results):
        document_id, element_id = id_list[idx]
        
        try:
            parsed_result = parser.parse(result.content)
            
            image_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "detail": parsed_result.detail,
                    "hypotheticalQuestions": parsed_result.hypotheticalQuestions
                }
            }
        except Exception as e:
            # 파싱 실패 시 원본 결과 저장
            image_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "raw_result": result.content,
                }
            }
        
        image_result.append(image_info)
    
    return image_result

@handle_rate_limits
def chart_information_chain(data_batches, id_list, model_name='gpt-4o', current_api_key=None, max_concurrency=5) -> List:
    '''
    차트 정보 추출 체인
    '''
    class ChartInfomationExtract(BaseModel):
        detail: str = Field(description='key insights and information visible in the Chart')
        hypotheticalQuestions: List[str] = Field(description="relevant hypothetical questions")
    
    parser = PydanticOutputParser(pydantic_object=ChartInfomationExtract)
    prompt = load_chat_prompt('../prompt/information_chart/chart_20250304_01.yaml')
    chat_prompt = ChatPromptTemplate.from_messages(prompt['messages'])
    chat_prompt = chat_prompt.partial(format=parser.get_format_instructions(), language='KOREAN')

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)
    chart_chain = chat_prompt | llm

    results = chart_chain.batch(data_batches, config={"max_concurrency": max_concurrency})
    
    # 결과 처리
    chart_result = []
    
    for idx, result in enumerate(results):
        document_id, element_id = id_list[idx]
        
        try:
            parsed_result = parser.parse(result.content)
            
            chart_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "detail": parsed_result.detail,
                    "hypotheticalQuestions": parsed_result.hypotheticalQuestions
                }
            }
        except Exception as e:
            # 파싱 실패 시 원본 결과 저장
            chart_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "raw_result": result.content,
                }
            }
        
        chart_result.append(chart_info)
    
    return chart_result

@handle_rate_limits
def normal_table_conversion_chain(data_batches, model_name='gpt-4o', current_api_key=None, max_concurrency=5) -> List:
    '''
    일반 테이블 이미지를 구조화된 JSON으로 변환하는 체인
    '''
    prompt = load_chat_prompt('../prompt/information_table/table_raw_20250326_01.yaml')
    chat_prompt = ChatPromptTemplate.from_messages(prompt["messages"])
    
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)
    normal_extract_chain = chat_prompt | llm
    
    results = normal_extract_chain.batch(data_batches, config={"max_concurrency": max_concurrency})
    return [result.content for result in results]

@handle_rate_limits
def cutoff_table_conversion_chain(data_batches, model_name='gpt-4o', current_api_key=None, max_concurrency=5) -> List:
    '''
    잘린 테이블 이미지를 구조화된 JSON으로 변환하는 체인
    '''
    prompt = load_chat_prompt('../prompt/information_table/table_cutoff_preprocessed_20250304_01.yaml')
    chat_prompt = ChatPromptTemplate.from_messages(prompt["messages"])
    chat_prompt = chat_prompt.partial(language='KOREAN')
    
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)
    cutoff_extract_chain = chat_prompt | llm
    
    results = cutoff_extract_chain.batch(data_batches, config={"max_concurrency": max_concurrency})
    return [result.content for result in results]

@handle_rate_limits
def table_information_chain(data_batches, id_list, model_name='gpt-4o', current_api_key=None, max_concurrency=5) -> List:
    '''
    테이블 정보 추출 체인
    
    Args:
        data_batches: 테이블 JSON, 제목, 컨텍스트를 포함한 데이터 배치
        id_list: (document_id, element_id) 튜플 목록 또는 (document_id, [element_ids]) 형식
        is_normal: 일반 테이블(True) 또는 잘린 테이블(False) 여부
    '''
    class TableInformationExtract(BaseModel):
        detail: str = Field(description='key insights and information visible in the table')
        hypotheticalQuestions: List[str] = Field(description="relevant hypothetical questions")
    
    parser = PydanticOutputParser(pydantic_object=TableInformationExtract)
    prompt = load_chat_prompt('../prompt/information_table/table_info_20250319_01.yaml')
    chat_prompt = ChatPromptTemplate.from_messages(prompt['messages'])
    chat_prompt = chat_prompt.partial(format=parser.get_format_instructions(), language='KOREAN')

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=current_api_key)
    information_extract_chain = chat_prompt | llm

    results = information_extract_chain.batch(data_batches, config={"max_concurrency": max_concurrency})
    
    # 결과 처리
    table_result = []
    
    for idx, result in enumerate(results):
        document_id, element_id = id_list[idx]
        table_json = data_batches[idx]['table_json']
        
        try:
            parsed_result = parser.parse(result.content)
            
            table_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "table_json": table_json,
                    "detail": parsed_result.detail,
                    "hypotheticalQuestions": parsed_result.hypotheticalQuestions
                }
            }
        except Exception as e:
            # 파싱 실패 시 원본 결과 저장
            table_info = {
                "document_id": document_id,
                "element_id": element_id,
                "information": {
                    "table_json": table_json,
                    "raw_result": result.content,
                }
            }
        
        table_result.append(table_info)
    
    return table_result