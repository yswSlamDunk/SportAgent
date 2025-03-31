import os
import json
from typing import List, Dict, Tuple, Set
from copy import deepcopy

from base import BaseNode
from state import GraphState
from utils import crop_make_dir, sanitize_filepath
from utils_image import merge_images, pdf_to_image, crop_image, encode_image, encode_pil_image
from chain import (summary_chain, chart_information_chain, 
                    image_information_chain, normal_table_conversion_chain, 
                    cutoff_table_conversion_chain, table_information_chain)


class OrganizeRelationNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, state: GraphState) -> Dict:
        '''
        relatedID가 [] 요소를 정렬하는 함수로, 두가지로 분류되며, 
        1. category가 caption인 경우
            - relatedID element에 {caption: 요소의 text} 형식으로 추가
            - 추가된 요소는 제거
        2. category가 caption이 아닌 경우
            - relatedID element의 바로 뒤에 요소를 복사
            - 복사된 요소는 제거
        '''
        self.log('OrganizeRelationNode 실행')
        organized_data = []
        relations = []
        
        # 관계가 있는 요소와 없는 요소 분리
        for element in state['documents']:
            if element['relatedID']:  # 빈 리스트가 아닌 경우
                relations.append(element)
            else:
                organized_data.append(element)
        
        # 관계가 있는 요소 처리
        for relation in relations:
            for related_id in relation['relatedID']:
                related_element_index = next(
                    (i for i, elem in enumerate(organized_data) if elem['id'] == related_id), 
                    None
                )
                
                if related_element_index is None:
                    continue
                    
                if relation['category'] == 'caption':
                    # 캡션인 경우 관련 요소에 캡션 추가
                    text = sanitize_filepath(relation['text'])
                    if text != '':
                        organized_data[related_element_index]['caption'] = text
                else:
                    # 아닌 경우 관련 요소 뒤에 삽입
                    organized_data.insert(related_element_index + 1, relation)
        
        self.log(f"Relations organized: {len(relations)} related elements processed")
        self.log('-'*30 +'\n')
        return {'documents': organized_data}

class RemoveUnusedCategoryNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, state: GraphState) -> Dict:
        '''
        state['unused_elements']에 있는 요소를 제거하는 함수.
        '''
        self.log('RemoveUnusedCategoryNode 실행')
        original_count = len(state['documents'])
        
        # 직접 사용하지 않는 방식으로 구현 (remove는 루프 중 오류 발생 가능)
        filtered_documents = [
            element for element in state['documents'] 
            if (element['category'], element['class']) not in state['unused_elements']
        ]
        
        removed_count = original_count - len(filtered_documents)
        self.log(f"Elements removed: {removed_count} elements of unused categories")
        self.log('-'*30 +'\n')
        
        return {'documents': filtered_documents}
    
class DocumentExtractNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, state: GraphState) -> Dict:
        """문서를 heading을 기준으로 chunking하는 함수
        
        Returns:
            Dict: {'documents': List[Dict]} 형태로 반환
            각 document는 다음 구조를 가짐:
            {
                'meta': {
                    'heading': Dict,  # 현재 문서의 heading 상태
                    'index': int     # chunk의 순서
                },
                'content': List[Dict]  # chunk에 포함된 elements
            }
        """
        self.log('DocumentExtractNode 실행')
        elements = state['documents']
        current_heading = {heading: None for heading in state['heading_structure']}
        
        chunk_index = 0
        chunks = []
        contents = []

        for i, doc in enumerate(elements):
            if 'heading' in doc['category']:
                # 기존 내용이 있으면 청크 저장
                if contents:
                    chunks.append({
                        'meta': {
                            'filepath': state['filepath'],
                            'heading': deepcopy(current_heading),  # 깊은 복사로 참조 문제 방지
                            'index': chunk_index
                        },
                        'content': contents
                    })
                    contents = []
                    chunk_index += 1
                
                # 현재 heading 업데이트    
                current_heading[doc['class']] = doc['text']
                
                # 하위 heading을 모두 None으로 초기화
                for heading in state['heading_structure'][state['heading_structure'].index(doc['class']) + 1:]:
                    current_heading[heading] = None
            else:
                contents.append(doc)
        
        # 마지막 내용이 있으면 청크 저장
        if contents:
            chunks.append({
                'meta': {
                    'filepath': state['filepath'],
                    'heading': deepcopy(current_heading),
                    'index': chunk_index
                },
                'content': contents
            })
        
        self.log(f"Documents extracted: {len(chunks)} chunks created from {len(elements)} elements")
        self.log('-'*30 +'\n')
        return {'documents': chunks}
    
class DocumentSummaryNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2
    
    def _make_dataset(self, state: GraphState) -> List[Dict]:
        stuff_dataset = []
        
        for document in state['documents']:
            heading = []
            for heading_key in state['heading_structure']:
                sentence = document['meta']['heading'][heading_key]
                if sentence != None:
                    heading.append(sentence.replace('\n', ''))
            heading = "\n".join(heading)

            text = []
            for element in document['content']:
                if element['category'] == 'paragraph':
                    text.append(' '.join([sentence.replace('\n', '') for sentence in element['text']]))

            text = " ".join(text)
            stuff_dataset.append({'document': heading + "\n" + text})

        return stuff_dataset

    def execute(self, state: GraphState) -> Dict:
        self.log('DocumentSummaryNode 실행')
        stuff_dataset = self._make_dataset(state)
        
        summary_result = []
        if len(stuff_dataset) // self.max_concurrency >= 1:
            for i in range(len(stuff_dataset) // self.max_concurrency):
                self.log(f"실행: {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(stuff_dataset)}")
                data = stuff_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                summary_result.extend(summary_chain(data, max_concurrency=self.max_concurrency))

        if len(stuff_dataset) % self.max_concurrency != 0:
            self.log(f"실행: {len(stuff_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(stuff_dataset)} // {len(stuff_dataset)}")
            data = stuff_dataset[len(stuff_dataset) // self.max_concurrency * self.max_concurrency:]
            summary_result.extend(summary_chain(data, max_concurrency=self.max_concurrency))

        result = state['documents'].copy()
        for i, summary in enumerate(summary_result):
            result[i]['meta']['summary'] = summary
        
        self.log(f"Document summary completed: {len(result)} documents processed")
        self.log('-'*30 +'\n')
        return {'documents': result}


class CropNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def execute(self, state: GraphState) -> Dict:
        '''
        category가 ['figure', 'chart', 'table']에 해당하는 element를 저장하고 path를 기록하는 과정
        '''
        self.log('CropNode 실행')
        folder_path_dict = crop_make_dir(state['filepath_pdf'])
        new_document = state['documents'].copy()

        for i, element in enumerate(state['documents']):
            for j, ele in enumerate(element['content']):
                if ele['category'] in ['figure', 'chart', 'table']:
                    page_index = ele.get('page') - 1
                    coordinates = ele['coordinates']
                    coordinates = (coordinates[0]['x'], coordinates[0]['y'], coordinates[2]['x'], coordinates[2]['y'])

                    page_img = pdf_to_image(state['filepath_pdf'], page_index)
                    
                    file_name = ele.get('caption', f"{ele['id']}")
                    output_path = os.path.join(folder_path_dict[ele['category']], file_name + '.jpg')
                    crop_image(page_img, coordinates, output_path)

                    new_document[i]['content'][j]['file_path'] = output_path
        
        return {'documents': new_document}
        

class ImageInformationNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2
    
    def _prepare_image_data(self, documents) -> tuple:
        '''
        문서에서 이미지 데이터 추출 및 처리할 데이터셋 준비
        '''
        id_list = []  # (document_id, element_id) 쌍 목록
        base64_list = []
        imageType_list = []
        title_list = []
        context_list = []
        
        for document in documents:
            document_id = document['meta']['index']
            document_summary = document['meta'].get('summary', '')
            for element in document['content']:
                if element['category'] == 'figure':
                    element_id = element['id']
                    base64, image_type = encode_image(element['file_path'])
                    
                    id_list.append((document_id, element_id))
                    base64_list.append(base64)
                    imageType_list.append(image_type)
                    context_list.append(document_summary)
                    title_list.append(element.get('caption', ''))
        
        image_dataset = [
            {'base64_image': base64, 'image_type': imageType, 'context': context, 'title': title} 
            for base64, imageType, context, title in zip(base64_list, imageType_list, context_list, title_list)
        ]
        
        return id_list, image_dataset
    
    def execute(self, state: GraphState) -> Dict:
        '''
        1. state['documents']에서 category가 Image인 element를 포함하는 document를 추출
            - 이 때, 해당 document id와 element id를 기록
        2. llm_batch에 적합하게 데이터셋 생성
        3. 뽑힌 document의 Image element를 대상으로 batch를 사용하여 정보를 추출
        4. document id, element id, 추출 정보를 state['image_result']에 맞게 저장 및 반환
        '''
        self.log('ImageInformationNode 실행')
        # 이미지 데이터 준비
        id_list, image_dataset = self._prepare_image_data(state['documents'])
        
        # 추출할 이미지가 없는 경우 빈 결과 반환
        if not id_list:
            self.log("No images found")
            return {"image_result": []}
        
        # 체인 실행
        results = []
        if len(image_dataset) // self.max_concurrency >= 1:
            for i in range(len(image_dataset) // self.max_concurrency):
                self.log(f"실행: {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(image_dataset)}")
                image_data = image_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                id_data = id_list[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                results.extend(image_information_chain(image_data, id_data, max_concurrency=self.max_concurrency))

        if len(image_dataset) % self.max_concurrency != 0:
            self.log(f"실행: {len(image_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(image_dataset)} // {len(image_dataset)}")
            image_data = image_dataset[len(image_dataset) // self.max_concurrency * self.max_concurrency:]
            id_data = id_list[len(image_dataset) // self.max_concurrency * self.max_concurrency:]
            results.extend(image_information_chain(image_data, id_data, max_concurrency=self.max_concurrency))
        
        self.log(f"Processing completed: {len(results)} images processed")
        self.log('-'*30 +'\n')
        return {"image_result": results}


class ChartInformationNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2
    
    def _prepare_chart_data(self, documents) -> tuple:
        '''
        문서에서 차트 데이터 추출 및 처리할 데이터셋 준비
        '''
        id_list = []  # (document_id, element_id) 쌍 목록
        base64_list = []
        imageType_list = []
        title_list = []
        context_list = []
        
        for document in documents:
            document_id = document['meta']['index']
            document_summary = document['meta'].get('summary', '')
            for element in document['content']:
                if element['category'] == 'chart':
                    element_id = element['id']
                    base64, image_type = encode_image(element['file_path'])
                    
                    id_list.append((document_id, element_id))
                    base64_list.append(base64)
                    imageType_list.append(image_type)
                    context_list.append(document_summary)
                    title_list.append(element.get('caption', ''))
        
        chart_dataset = [
            {'base64_image': base64, 'image_type': imageType, 'context': context, 'title': title} 
            for base64, imageType, context, title in zip(base64_list, imageType_list, context_list, title_list)
        ]
        
        return id_list, chart_dataset
    
    def execute(self, state: GraphState) -> Dict:
        '''
        1. state['documents']에서 category가 chart인 element를 포함하는 document를 추출
            - 이 때, 해당 document id와 element id를 기록
        2. llm_batch에 적합하게 데이터셋 생성
        3. 뽑힌 document의 Chart element를 대상으로 batch를 사용하여 정보를 추출
        4. document id, element id, 추출 정보를 state['chart_result']에 맞게 저장 및 반환
        '''
        self.log('ChartInformationNode 실행')
        # 차트 데이터 준비
        id_list, chart_dataset = self._prepare_chart_data(state['documents'])
        
        # 추출할 차트가 없는 경우 빈 결과 반환
        if not id_list:
            self.log("No charts found")
            return {"chart_result": []}
        
        # 체인 실행
        results = []
        if len(chart_dataset) // self.max_concurrency >= 1:
            for i in range(len(chart_dataset) // self.max_concurrency):
                self.log(f"실행: {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(chart_dataset)}")
                chart_data = chart_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                id_data = id_list[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                results.extend(chart_information_chain(chart_data, id_data, max_concurrency=self.max_concurrency))
        
        if len(chart_dataset) % self.max_concurrency != 0:
            self.log(f"실행: {len(chart_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(chart_dataset)} // {len(chart_dataset)}")
            chart_data = chart_dataset[len(chart_dataset) // self.max_concurrency * self.max_concurrency:]
            id_data = id_list[len(chart_dataset) // self.max_concurrency * self.max_concurrency:]
            results.extend(chart_information_chain(chart_data, id_data, max_concurrency=self.max_concurrency))
        
        self.log(f"Processing completed: {len(results)} charts processed")
        self.log('-'*30 +'\n')
        return {"chart_result": results}
    

class ClassifyTableNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _process_document_tables(self, document):
        """
        문서 내 테이블 요소를 처리하고 일반/잘린 테이블로 분류
        """
        document_id = document['meta']['index']
        document_summary = document['meta'].get('summary', '')

        element_sequence = []
        element_ids = []
        element_paths = []
        
        for element in document['content']:
            element_type = 'c' if element['class'] == 'cutOff' else 't' if element['category'] == 'table' else 'e'
            element_sequence.append(element_type)
            element_ids.append(element.get('id', ''))
            element_paths.append(element.get('file_path', ''))

        cutoff_items = []
        normal_items = []
        
        current_type = []
        first_index = 0
        last_index = 0
        
        for i, element_type in enumerate(element_sequence):
            if not current_type:
                if element_type == 't':
                    current_type.append('t')
                    first_index = i
                    last_index = i
            else:  # current_type이 ['t'] 또는 ['t', 'c'] 이런 경우
                if element_type == 'c':
                    current_type.append('c')
                    last_index = i
                
                else:  # element_type 이 't' 또는 'e'인 경우
                    if len(current_type) > 1:
                        item = self._create_cutoff_item(document, document_id, document_summary, 
                                                     element_paths, first_index, last_index)
                        cutoff_items.append(item)
                    elif len(current_type) == 1:
                        item = self._create_normal_item(document, document_id, document_summary,
                                                     element_paths, first_index)
                        normal_items.append(item)

                    if element_type == 't':
                        current_type = ['t']
                        first_index = i
                        last_index = i
                    else:
                        current_type = []
        
        # 마지막에 남아있는 element에 대한 작업 필요
        if current_type:
            if len(current_type) > 1:
                item = self._create_cutoff_item(document, document_id, document_summary,
                                             element_paths, first_index, last_index)
                cutoff_items.append(item)
            else: 
                item = self._create_normal_item(document, document_id, document_summary,
                                             element_paths, first_index)
                normal_items.append(item)
                
        return normal_items, cutoff_items
    
    def _create_normal_item(self, document, document_id, document_summary, element_paths, index):
        """일반 테이블 항목 생성"""
        base64_image, image_type = encode_image(element_paths[index])
        return {
            'dataset': {
                'base64_image': base64_image, 
                'image_type': image_type, 
                'context': document_summary, 
                'title': document['content'][index].get('caption', '')
            },
            'ids': [(document_id, document['content'][index].get('id'))]
        }
    
    def _create_cutoff_item(self, document, document_id, document_summary, element_paths, start_index, end_index):
        """잘린 테이블 항목 생성 (이미지 병합)"""
        merge_image = merge_images(element_paths[start_index:end_index+1])
        base64_image, image_type = encode_pil_image(merge_image)
        
        return {
            'dataset': {
                'base64_image': base64_image, 
                'image_type': image_type, 
                'context': document_summary, 
                'title': document['content'][start_index].get('caption', '')
            },
            'ids': [(document_id, document['content'][index].get('id')) for index in range(start_index, end_index+1)]
        }
    
    def execute(self, state: GraphState) -> Dict:
        '''
        문서에서 테이블을 식별하고 일반 테이블과 잘린 테이블로 분류
        
        중요한 점:
        - cutoff는 앞에 테이블이 있어야 하고, 연속적으로 나올 수 있음
        - 이 과정에서 전처리까지 완료
        - cutoff는 이미지를 하나로 병합하여 처리
        
        반환: 
        - normal_table: 일반 테이블 리스트
        - cutoff_table: 잘린 테이블 리스트 (병합 처리됨)
        '''
        self.log('ClassifyTableNode 실행')
        normal_table = []
        cutoff_table = []
        
        for document in state['documents']:
            normal_items, cutoff_items = self._process_document_tables(document)
            normal_table.extend(normal_items)
            cutoff_table.extend(cutoff_items)
        
        self.log(f"Processing completed: {len(normal_table)} normal tables, {len(cutoff_table)} cutoff tables")
        self.log('-'*30 +'\n')
        return {
            'normal_table': normal_table,
            'cutoff_table': cutoff_table
        }



class TableNormalConvertingNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2

    def _prepare_normal_table_data(self, normal_table):
        """
        일반 테이블 데이터 처리를 위한 데이터셋 준비
        """
        normal_dataset = []
        for data in normal_table:
            dt = data['dataset']
            normal_dataset.append({
                'base64_image': dt['base64_image'], 
                'image_type': dt['image_type']
            })
        
        return normal_dataset
    
    def execute(self, state: GraphState) -> Dict:
        """
        일반 테이블 이미지를 구조화된 데이터(JSON)로 변환
        
        1. 일반 테이블 데이터셋 준비
        2. 체인을 사용하여 이미지에서 테이블 구조 추출
        3. 추출된 테이블 데이터를 원본 테이블 데이터에 추가
        """
        self.log('TableNormalConvertingNode 실행')
        # 테이블 데이터 없으면 빈 결과 반환
        if not state.get('normal_table'):
            self.log("No normal tables found")
            return {'normal_table': []}
        
        # 데이터셋 준비
        normal_dataset = self._prepare_normal_table_data(state['normal_table'])
        
        # 체인 실행
        results = []
        if len(normal_dataset) // self.max_concurrency >= 1:
            for i in range(len(normal_dataset) // self.max_concurrency):
                self.log(f"실행: {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(normal_dataset)}")
                normal_data = normal_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                results.extend(normal_table_conversion_chain(normal_data, max_concurrency=self.max_concurrency))

        if len(normal_dataset) % self.max_concurrency != 0:
            self.log(f"실행: {len(normal_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(normal_dataset)} // {len(normal_dataset)}")
            normal_data = normal_dataset[len(normal_dataset) // self.max_concurrency * self.max_concurrency:]
            results.extend(normal_table_conversion_chain(normal_data, max_concurrency=self.max_concurrency))
        
        # 결과 처리
        normal_table = state['normal_table'].copy()
        for i, table_json in enumerate(results):
            normal_table[i]['dataset']['table_json'] = table_json
        
        self.log(f"Processing completed: {len(results)} normal tables converted")
        self.log('-'*30 +'\n')
        return {'normal_table': normal_table}
    

class TableCutoffConvertingNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2

    def _prepare_cutoff_table_data(self, cutoff_table):
        """
        잘린 테이블 데이터 처리를 위한 데이터셋 준비
        """
        cutoff_dataset = []
        for data in cutoff_table:
            dt = data['dataset']
            cutoff_dataset.append({
                'base64_image': dt['base64_image'], 
                'image_type': dt['image_type']
            })
        
        return cutoff_dataset
    
    def execute(self, state: GraphState) -> Dict:
        """
        잘린 테이블 이미지를 구조화된 데이터(JSON)로 변환
        
        1. 잘린 테이블 데이터셋 준비
        2. 체인을 사용하여 이미지에서 테이블 구조 추출
        3. 추출된 테이블 데이터를 원본 테이블 데이터에 추가
        """
        self.log('TableCutoffConvertingNode 실행')
        # 테이블 데이터 없으면 빈 결과 반환
        if state.get('cutoff_table') == []:
            self.log("No cutoff tables found")
            return {'cutoff_table': []}
        
        # 데이터셋 준비
        cutoff_dataset = self._prepare_cutoff_table_data(state['cutoff_table'])
        
        # 체인 실행
        results = []

        if len(cutoff_dataset) // self.max_concurrency >= 1: 
            for i in range(len(cutoff_dataset) // self.max_concurrency):
                self.log(f"실행: {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(cutoff_dataset)}")
                cutoff_data = cutoff_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                results.extend(cutoff_table_conversion_chain(cutoff_data, max_concurrency=self.max_concurrency))

        if len(cutoff_dataset) % self.max_concurrency != 0:
            self.log(f"실행: {len(cutoff_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(cutoff_dataset)} // {len(cutoff_dataset)}")
            cutoff_data = cutoff_dataset[len(cutoff_dataset) // self.max_concurrency * self.max_concurrency:]
            results.extend(cutoff_table_conversion_chain(cutoff_data, max_concurrency=self.max_concurrency))
        
        # 결과 처리
        cutoff_table = state['cutoff_table'].copy()
        for i, table_json in enumerate(results):
            cutoff_table[i]['dataset']['table_json'] = table_json
        
        self.log(f"Processing completed: {len(results)} cutoff tables converted")
        self.log('-'*30 +'\n')
        return {'cutoff_table': cutoff_table}
    
class TableInformationNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_concurrency = 2

    def _prepare_normal_table_data(self, state: GraphState) -> Tuple[List, List]:
        """
        일반 테이블 데이터셋과 ID 목록 준비
        """
        dataset = []
        id_list = []
        
        if state.get('normal_table', []):
            for data in state.get('normal_table'):
                dt = data['dataset']
                dataset.append({
                    'table_json': dt['table_json'], 
                    'title': dt['title'], 
                    'context': dt['context']
                })
            
                document_id = data['ids'][0][0]
                element_id = data['ids'][0][1]
                id_list.append((document_id, element_id))
        
        return dataset, id_list
    
    def _prepare_cutoff_table_data(self, state: GraphState) -> Tuple[List, List]:
        """
        잘린 테이블 데이터셋과 ID 목록 준비
        """
        dataset = []
        id_list = []
        
        if state.get('cutoff_table', []):
            for data in state.get('cutoff_table'):
                dt = data['dataset']
                dataset.append({
                    'table_json': dt['table_json'], 
                    'title': dt['title'], 
                    'context': dt['context']
                })
                
                document_id = data['ids'][0][0]
                # 잘린 테이블은 요소 ID가 여러 개일 수 있음
                element_ids = [ids[1] for ids in data['ids']]
                id_list.append((document_id, element_ids))
        
        return dataset, id_list
    
    def execute(self, state: GraphState) -> Dict:
        """
        일반 테이블과 잘린 테이블의 정보를 추출하여 통합된 결과 반환
        """
        self.log('TableInformationNode 실행')
        # 일반 테이블 데이터 준비
        normal_dataset, normal_id_list = self._prepare_normal_table_data(state)
        
        # 잘린 테이블 데이터 준비
        cutoff_dataset, cutoff_id_list = self._prepare_cutoff_table_data(state)
        
        # 테이블이 없는 경우 빈 결과 반환
        if not normal_dataset and not cutoff_dataset:
            self.log("No tables found")
            return {'table_result': []}
        
        # 일반 테이블 처리
        normal_results = []
        if normal_dataset != []:
            if len(normal_dataset) // self.max_concurrency >= 1:
                for i in range(len(normal_dataset) // self.max_concurrency):
                    self.log(f"실행(normal): {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(normal_dataset)}")
                    normal_data = normal_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                    normal_data_id = normal_id_list[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                    normal_results.extend(table_information_chain(normal_data, normal_data_id, max_concurrency=self.max_concurrency))
            
            if len(normal_dataset) % self.max_concurrency != 0:
                self.log(f"실행(normal): {len(normal_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(normal_dataset)} // {len(normal_dataset)}")
                normal_data = normal_dataset[len(normal_dataset) // self.max_concurrency * self.max_concurrency:]
                normal_data_id = normal_id_list[len(normal_dataset) // self.max_concurrency * self.max_concurrency:]
                normal_results.extend(table_information_chain(normal_data, normal_data_id, max_concurrency=self.max_concurrency))
            self.log(f"Processed {len(normal_results)} normal tables")
        else:
            self.log("No normal tables found")
        
        # 잘린 테이블 처리
        cutoff_results = []
        if cutoff_dataset != []:
            if len(cutoff_dataset) // self.max_concurrency >= 1:
                for i in range(len(cutoff_dataset) // self.max_concurrency):
                    self.log(f"실행(cutoff): {i * self.max_concurrency + 1} ~ {(i + 1) * self.max_concurrency} // {len(cutoff_dataset)}")
                    cutoff_data = cutoff_dataset[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                    cutoff_data_id = cutoff_id_list[i * self.max_concurrency: (i + 1) * self.max_concurrency]
                    cutoff_results.extend(table_information_chain(cutoff_data, cutoff_data_id, max_concurrency=self.max_concurrency))
        
            if len(cutoff_dataset) % self.max_concurrency != 0:
                self.log(f"실행(cutoff): {len(cutoff_dataset) // self.max_concurrency * self.max_concurrency + 1} ~ {len(cutoff_dataset)} // {len(cutoff_dataset)}")
                cutoff_data = cutoff_dataset[len(cutoff_dataset) // self.max_concurrency * self.max_concurrency:]
                cutoff_data_id = cutoff_id_list[len(cutoff_dataset) // self.max_concurrency * self.max_concurrency:]
                cutoff_results.extend(table_information_chain(cutoff_data, cutoff_data_id, max_concurrency=self.max_concurrency))
            self.log(f"Processed {len(cutoff_results)} cutoff tables")
        else:
            self.log("No cutoff tables found")
        
        # 결과 통합
        all_results = normal_results + cutoff_results
        
        self.log(f"Processing completed: table_result {len(all_results)} tables analyzed")
        self.log('-'*30 +'\n')
        return {'table_result': all_results}

class AlignNode(BaseNode):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _create_info_maps(self, state: GraphState) -> Tuple[dict, dict, dict, Set]:
        """
        테이블, 이미지, 차트 정보 매핑을 생성하고 이미 처리된 요소 ID 세트 반환
        """
        # 테이블 정보 매핑 구성
        table_info_map = {}
        processed_element_ids = set()  # 이미 처리된 element_id 추적 세트
        
        if 'table_result' in state:
            for table_info in state['table_result']:
                doc_id = table_info['document_id']
                elem_id = table_info['element_id']
                
                # 단일 ID 또는 ID 리스트 모두 처리
                if isinstance(elem_id, list):
                    for i, eid in enumerate(elem_id):
                        if i == 0:
                            table_info_map[(doc_id, eid)] = table_info['information']
                        else:
                            processed_element_ids.add((doc_id, eid))
                else:
                    table_info_map[(doc_id, elem_id)] = table_info['information']
        
        # 이미지 정보 매핑 구성
        image_info_map = {}
        if 'image_result' in state:
            for image_info in state['image_result']:
                doc_id = image_info['document_id']
                elem_id = image_info['element_id']
                image_info_map[(doc_id, elem_id)] = image_info['information']
        
        # 차트 정보 매핑 구성
        chart_info_map = {}
        if 'chart_result' in state:
            for chart_info in state['chart_result']:
                doc_id = chart_info['document_id']
                elem_id = chart_info['element_id']
                chart_info_map[(doc_id, elem_id)] = chart_info['information']
        
        return table_info_map, image_info_map, chart_info_map, processed_element_ids
    
    def _extract_pages(self, document) -> List[int]:
        """
        문서의 모든 요소에서 페이지 번호를 추출하여 정렬된 고유 목록 반환
        """
        all_pages = []
        for element in document['content']:
            if 'page' in element:
                if isinstance(element['page'], list):
                    all_pages.extend(element['page'])
                else:
                    all_pages.append(element['page'])
        
        # 중복 제거 및 정렬
        return sorted(list(set(all_pages)))
    
    def _process_paragraph(self, document, start_idx) -> Tuple[Dict, int]:
        """
        연속된 paragraph를 병합하고 새 paragraph 정보와 다음 처리할 인덱스를 반환
        """
        element = document['content'][start_idx]
        
        # 병합할 paragraph 텍스트 수집
        merged_text = element.get('text', '')
        if isinstance(merged_text, list):
            merged_text = ' '.join(merged_text)
        
        next_idx = start_idx + 1
        
        # 연속된 paragraph 찾기
        while next_idx < len(document['content']):
            next_element = document['content'][next_idx]
            if next_element.get('category') == 'paragraph':
                next_text = next_element.get('text', '')
                if isinstance(next_text, list):
                    next_text = ' '.join(next_text)
                merged_text += ' ' + next_text
                next_idx += 1
            else:
                break
        
        paragraph_info = {
            'category': 'paragraph',
            'information': merged_text
        }
        
        return paragraph_info, next_idx
    
    def _process_media_element(self, category, element, doc_id, element_id, info_map):
        """
        이미지, 차트, 테이블과 같은 미디어 요소 처리
        """
        caption = element.get('caption', '')
        filepaths = [element.get('file_path', '')] if element.get('file_path') else []
        
        # 정보 가져오기
        element_info = info_map.get((doc_id, element_id), {})
        
        # 파싱 오류 처리
        if 'raw_result' in element_info:
            information = element_info.get('raw_result', {})
        else:
            information = element_info.copy()
        return {
            'category': category,
            'caption': caption,
            'filepaths': filepaths,
            'information': information
        }
    
    def _save_state_json(self, state, filepath, suffix):
        """
        상태 JSON을 파일로 저장
        """
        # state_json = json.dumps(state)
        output_path = os.path.splitext(filepath)[0] + suffix
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(state, f)
        self.log(f"State JSON saved to: {output_path}")
    
    def execute(self, state: GraphState) -> Dict:
        """
        문서 데이터를 최종 형태로 정리하는 함수
        - 불필요한 필드 제거 (coordinates, relatedID, class 등)
        - 연속된 paragraph 병합
        - 각 요소의 정보를 타입별로 적절히 구성
        - 메타데이터에 페이지 목록(pages) 추가
        - 파싱 오류가 있는 경우 적절히 처리
        """        
        self.log('AlignNode 실행')
        # self._save_state_json(state, state['filepath_pdf'], '_align_state.json')
        # 정보 매핑 생성
        table_info_map, image_info_map, chart_info_map, processed_element_ids = self._create_info_maps(state)
        
        result = []
        
        for document in state['documents']:
            doc_id = document['meta']['index']
            
            new_document = {
                'meta': document['meta'].copy(),
                'content': []
            }
            
            # 페이지 목록 추가
            new_document['meta']['pages'] = self._extract_pages(document)
            
            # 요소 그룹화 및 처리
            i = 0
            while i < len(document['content']):
                element = document['content'][i]
                category = element.get('category', '')
                element_id = element.get('id', '')
                
                # 이미 처리된 요소 건너뛰기
                if (doc_id, element_id) in processed_element_ids:
                    i += 1
                    continue
                
                # 1. 연속된 paragraph 처리
                if category == 'paragraph':
                    paragraph_info, next_idx = self._process_paragraph(document, i)
                    new_document['content'].append(paragraph_info)
                    i = next_idx
                    continue
                
                # 2. 테이블 처리
                elif category == 'table':
                    table_result = self._process_media_element(
                        'table', element, doc_id, element_id, table_info_map
                    )
                    new_document['content'].append(table_result)
                
                # 3. 차트 처리
                elif category == 'chart':
                    chart_result = self._process_media_element(
                        'chart', element, doc_id, element_id, chart_info_map
                    )
                    new_document['content'].append(chart_result)
                
                # 4. 이미지(figure) 처리
                elif category == 'figure':
                    figure_result = self._process_media_element(
                        'figure', element, doc_id, element_id, image_info_map
                    )
                    new_document['content'].append(figure_result)
                
                i += 1
            
            result.append(new_document)
        
        # 최종 상태 저장
        state['documents'] = result
        self._save_state_json(state, state['filepath_pdf'], '_dataset.json')
        
        self.log(f"Processing completed: {len(result)} documents aligned")
        self.log('-'*30 +'\n')
        return {'documents': result}