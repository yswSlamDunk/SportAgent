import os
import yaml
import random
import time
from langchain_core.runnables.graph import MermaidDrawMethod
import openai

def load_chat_prompt(yaml_file):
    with open(yaml_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

    
def crop_make_dir(pdf_path):
    path_split = pdf_path.split('/')
    name = path_split[-1].split('.')[0]

    path_base = '/'.join(path_split[0:-1] + [name])
    if not os.path.isdir(path_base):
        os.mkdir(path_base)

    folder_path_dict = {'figure': '',
                        'table': '',
                        'chart': ''}

    for folder in ['figure', 'table', 'chart']:
        folder_path = os.path.join(path_base, folder)
        
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        folder_path_dict[folder] = folder_path
    return folder_path_dict


def visualize_graph(graph, output_file_path=None):
    if output_file_path is None:
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'graph.png')

    png_data = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)


    with open(output_file_path, 'wb') as f:
        f.write(png_data)

    print(f"Graph saved to {output_file_path}")


def sanitize_filepath(filepath):
    """
    파일 경로에서 유효하지 않은 문자를 제거하고 안전한 경로로 변환
    """
    # 디렉토리와 파일명 분리
    directory, filename = os.path.split(filepath)
    
    # 파일명에서 유효하지 않은 문자 제거
    # 줄바꿈, 탭 등의 문자 제거
    filename = filename.replace('\n', '_').replace('\r', '_').replace('\t', '_')
    
    # 윈도우에서 금지된 파일명 문자 제거 또는 대체: \ / : * ? " < > |
    forbidden_chars = ['\\', '/', ':', '*', '?', '"', '<', '>', '|']
    for char in forbidden_chars:
        filename = filename.replace(char, '_')
    
    # 파일명 길이 제한 (윈도우는 보통 255자 제한)
    if len(filename) > 200:
        base, ext = os.path.splitext(filename)
        filename = base[:196] + ext  # 확장자 유지하면서 길이 제한
    
    # 역슬래시를 정방향 슬래시로 변환
    directory = directory.replace('\\', '/')
    
    # 디렉토리와 파일명 결합
    return os.path.join(directory, filename)