from langgraph.graph import StateGraph, END, START
from state import GraphState
from extractor import (ClassifyTableNode, TableNormalConvertingNode, 
                        TableCutoffConvertingNode, TableInformationNode,
                        ImageInformationNode, DocumentExtractNode, 
                        DocumentSummaryNode, CropNode, ChartInformationNode, 
                        AlignNode, OrganizeRelationNode, RemoveUnusedCategoryNode)       
from utils import visualize_graph


def create_subgraph_table(verbose: bool=True, visualize: bool=False):
    classifyTableNode = ClassifyTableNode(verbose=verbose)
    tableNormalConvertingNode = TableNormalConvertingNode(verbose=verbose)
    tableCutoffConvertingNode = TableCutoffConvertingNode(verbose=verbose)
    tableInformationNode = TableInformationNode(verbose=verbose)

    workflow = StateGraph(GraphState)
    workflow.add_node('classifyTableNode', classifyTableNode)
    workflow.add_node('tableNormalConvertingNode', tableNormalConvertingNode)
    workflow.add_node('tableCutoffConvertingNode', tableCutoffConvertingNode)
    workflow.add_node('tableInformationNode', tableInformationNode)

    workflow.add_edge(START, 'classifyTableNode')
    workflow.add_edge('classifyTableNode', 'tableNormalConvertingNode')
    workflow.add_edge('classifyTableNode', 'tableCutoffConvertingNode')
    workflow.add_edge('tableNormalConvertingNode', 'tableInformationNode')
    workflow.add_edge('tableCutoffConvertingNode', 'tableInformationNode')
    workflow.add_edge('tableInformationNode', END)

    graph = workflow.compile()

    if visualize:
        visualize_graph(graph, output_file_path='./graph/table_graph.png')

    return graph


def create_parser_graph(verbose: bool=True, visualize: bool=False):
    organizeRelationNode = OrganizeRelationNode()
    removeUnusedCategoryNode = RemoveUnusedCategoryNode()
    documentExtractNode = DocumentExtractNode()
    documentSummaryNode = DocumentSummaryNode()
    cropNode = CropNode()
    imageInformationNode = ImageInformationNode()
    chartInformationNode = ChartInformationNode()
    alignNode = AlignNode()
    
    subgraph_table = create_subgraph_table(verbose=True, visualize=False)

    workflow = StateGraph(GraphState)
    workflow.add_node('organizeRelationNode', organizeRelationNode)
    workflow.add_node('removeUnusedCategoryNode', removeUnusedCategoryNode)
    workflow.add_node('documentExtractNode', documentExtractNode)
    workflow.add_node('documentSummaryNode', documentSummaryNode)
    workflow.add_node('cropNode', cropNode)
    workflow.add_node('imageInformationNode', imageInformationNode)
    workflow.add_node('chartInformationNode', chartInformationNode)
    workflow.add_node('subGraph_tabelExtractorNode', subgraph_table)
    workflow.add_node('alignNode', alignNode)
    
    workflow.add_edge(START, 'organizeRelationNode')
    workflow.add_edge('organizeRelationNode', 'removeUnusedCategoryNode')
    workflow.add_edge('removeUnusedCategoryNode', 'documentExtractNode')
    workflow.add_edge('documentExtractNode', 'documentSummaryNode')
    workflow.add_edge('documentSummaryNode', 'cropNode')
    workflow.add_edge('cropNode', 'imageInformationNode')
    workflow.add_edge('cropNode', 'chartInformationNode')
    workflow.add_edge('cropNode', 'subGraph_tabelExtractorNode')
    workflow.add_edge('imageInformationNode', 'alignNode')
    workflow.add_edge('chartInformationNode', 'alignNode')
    workflow.add_edge('subGraph_tabelExtractorNode', 'alignNode')
    workflow.add_edge('alignNode', END)

    graph = workflow.compile()

    if visualize:
        visualize_graph(graph, output_file_path='./graph/parser_graph.png')

    return graph
    
