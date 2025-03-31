import operator
from typing import TypedDict, Annotated, List, Dict, Tuple

class GraphState(TypedDict):
    filepath: Annotated[str, 'filepath']
    filepath_pdf: Annotated[str, 'pdf filepath']
    
    originData: Annotated[List[Dict], 'originData']
    documents: Annotated[List[Dict], 'documents']
    
    heading_structure: Annotated[List[str], 'heading structure. last heading is borderline']
    unused_elements: Annotated[List[Tuple[str, str]], 'unused elements']

    image_result: Annotated[List[Dict], 'result of image information extractor ', operator.add]
    chart_result: Annotated[List[Dict], 'result of chart information extractor ', operator.add]
    table_result: Annotated[List[Dict], 'result of table information extractor ', operator.add]

    normal_table: Annotated[List[Dict], 'table where class is not a cutoff']
    cutoff_table: Annotated[List[Dict], 'table where class is a cutoff']
