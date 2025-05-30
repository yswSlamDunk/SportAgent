import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union


def analyze_relationship(
    kg: Any,
    rel_types: List[str] = None,
    figsize: tuple = (7, 5),
    colors: List[str] = ['#4F81BD', '#C0504D'],
    show_plot: bool = True,
    heading: str = 'heading1'
) -> Dict[str, Dict[str, int]]:
    """
    지식 그래프의 관계를 분석하고 시각화합니다.

    Args:
        kg: 지식 그래프 객체
        rel_types: 분석할 관계 타입 리스트
        figsize: 그래프 크기
        colors: 막대 그래프 색상
        show_plot: 그래프 표시 여부

    Returns:
        Dict[str, Dict[str, int]]: 관계 통계 정보
    """
    # 기본 관계 타입 설정
    if rel_types is None:
        rel_types = ['entities_overlap', 'cosine_similarity']

    # 노드 id → 섹션명 매핑
    node_sector = {}
    for node in kg.nodes:
        try:
            node_id = node.id.hex
            sector = node.properties['document_metadata']['heading'][heading]
            node_sector[node_id] = sector
        except (AttributeError, KeyError) as e:
            print(f"Warning: Node {node_id} has invalid structure: {e}")
            continue

    # 통계 변수 초기화
    stats = {rtype: {'total': 0, 'same_sector': 0, 'diff_sector': 0} for rtype in rel_types}
    sector_rel_count = defaultdict(int)

    # 관계 순회하며 통계 집계
    for rel in kg.relationships:
        try:
            rel_type = getattr(rel, 'type', rel.__dict__.get('type', ''))
            source_id = rel.__dict__['source'].id.hex
            target_id = rel.__dict__['target'].id.hex
            source_sector = node_sector.get(source_id, 'Unknown')
            target_sector = node_sector.get(target_id, 'Unknown')

            sector_rel_count[source_sector] += 1
            sector_rel_count[target_sector] += 1

            if rel_type not in stats:
                continue

            stats[rel_type]['total'] += 1
            if source_sector == target_sector:
                stats[rel_type]['same_sector'] += 1
            else:
                stats[rel_type]['diff_sector'] += 1
        except (AttributeError, KeyError) as e:
            print(f"Warning: Invalid relationship structure: {e}")
            continue

    if show_plot:
        for rtype in rel_types:
            plt.figure(figsize=figsize)
            same_count = stats[rtype]['same_sector']
            diff_count = stats[rtype]['diff_sector']
            total = stats[rtype]['total']

            plt.bar(['동일 섹션', '타 섹션'], 
                   [same_count, diff_count], 
                   color=colors)

            for i, v in enumerate([same_count, diff_count]):
                percentage = (v/total*100) if total > 0 else 0
                plt.text(i, v, f'{v} ({percentage:.1f}%)', 
                        ha='center', va='bottom', 
                        fontsize=11, fontweight='bold')

            plt.title(f'{rtype} 관계 분포({heading})', fontsize=14)
            plt.ylabel('관계 개수')
            plt.tight_layout()
            plt.show()

    return stats

def analyze_sector_connection_ratio(
    kg: Any,
    rel_types: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 6),
    color_maps: Optional[Dict[str, str]] = None,
    return_dfs: bool = True,
    show_plot: bool = True,
    heading: str = 'heading1'
) -> Union[Tuple[pd.DataFrame, ...], None]:
    """
    섹터 간 연결 비율을 분석하고 시각화합니다.

    Args:
        kg: 지식 그래프 객체
        rel_types: 분석할 관계 타입 리스트
        figsize: 그래프 크기
        color_maps: 관계 타입별 컬러맵 딕셔너리
        return_dfs: DataFrame 반환 여부
        show_plot: 그래프 표시 여부

    Returns:
        Optional[Tuple[pd.DataFrame, ...]]: 관계 타입별 연결 비율 DataFrame
    """
    # 기본값 설정
    if rel_types is None:
        rel_types = ['cosine_similarity', 'entities_overlap']
    
    if color_maps is None:
        color_maps = {rtype: 'YlOrRd' for rtype in rel_types}

    # 노드 id → 섹터명 매핑 및 섹터별 노드 id 집합
    node_sector = {}
    sector_nodes = defaultdict(set)
    
    try:
        for node in kg.nodes:
            node_id = node.id.hex
            sector = node.properties['document_metadata']['heading'][heading]
            node_sector[node_id] = sector
            sector_nodes[sector].add(node_id)
    except (AttributeError, KeyError) as e:
        print(f"Warning: Error processing nodes: {e}")
        return None

    sectors = sorted(sector_nodes.keys())
    n = len(sectors)

    # 관계 타입별 실제 관계 수 매트릭스
    actual_matrices = {rtype: np.zeros((n, n), dtype=int) for rtype in rel_types}
    # 가능한 최대 관계 수 매트릭스
    max_matrix = np.zeros((n, n), dtype=int)

    # 관계 집계
    for rel in kg.relationships:
        try:
            rel_type = getattr(rel, 'type', rel.__dict__.get('type', ''))
            source_id = rel.__dict__['source'].id.hex
            target_id = rel.__dict__['target'].id.hex
            source_sector = node_sector.get(source_id, 'Unknown')
            target_sector = node_sector.get(target_id, 'Unknown')
            
            if source_sector in sectors and target_sector in sectors:
                i = sectors.index(source_sector)
                j = sectors.index(target_sector)
                if rel_type in rel_types:
                    actual_matrices[rel_type][i, j] += 1
        except (AttributeError, KeyError) as e:
            print(f"Warning: Error processing relationship: {e}")
            continue

    # 가능한 최대 관계 수 계산
    for i, s1 in enumerate(sectors):
        for j, s2 in enumerate(sectors):
            max_matrix[i, j] = len(sector_nodes[s1]) * len(sector_nodes[s2])

    # 연결 비율(%) 매트릭스 생성
    ratio_matrices = {}
    ratio_dfs = {}
    
    for rtype in rel_types:
        ratio_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if max_matrix[i, j] > 0:
                    ratio_matrix[i, j] = actual_matrices[rtype][i, j] / max_matrix[i, j] * 100
        ratio_matrices[rtype] = ratio_matrix
        ratio_dfs[rtype] = pd.DataFrame(ratio_matrix, index=sectors, columns=sectors)

    if show_plot:
        # 히트맵 시각화
        fig, axes = plt.subplots(1, len(rel_types), figsize=figsize)
        if len(rel_types) == 1:
            axes = [axes]

        for idx, rtype in enumerate(rel_types):
            im = axes[idx].imshow(
                ratio_matrices[rtype], 
                cmap=color_maps[rtype], 
                vmin=0, 
                vmax=100
            )
            
            plt.colorbar(im, ax=axes[idx], label='연결 비율(%)')
            
            axes[idx].set_xticks(range(n))
            axes[idx].set_xticklabels(sectors, rotation=45, ha='right')
            axes[idx].set_yticks(range(n))
            axes[idx].set_yticklabels(sectors)
            axes[idx].set_title(f'{heading} 섹션 간 연결 비율(%) - {rtype}')

            # 값 표시
            for i in range(n):
                for j in range(n):
                    value = ratio_matrices[rtype][i, j]
                    color = 'black' if value < 60 else 'white'
                    axes[idx].text(
                        j, i, f"{value:.1f}%",
                        ha='center', va='center',
                        color=color,
                        fontsize=9,
                        fontweight='bold'
                    )

        plt.tight_layout()
        plt.show()

    if return_dfs:
        return tuple(ratio_dfs[rtype] for rtype in rel_types)
    return None