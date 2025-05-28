from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field

import typing as t
import logging
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from ragas.testset.graph import KnowledgeGraph, Node

from ragas.testset.synthesizers.multi_hop import MultiHopScenario
from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer


logger = logging.getLogger(__name__)

@dataclass
class FastMultiHopAbstractQuerySynthesizer(MultiHopAbstractQuerySynthesizer):
    name: str = "fast_multi_hop_abstract_synthesizer"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._neighbor_cache = {}
        self._cluster_cache = {}
        
    def _build_neighbor_map(self, knowledge_graph: KnowledgeGraph) -> dict:
        """선처리: 노드별 이웃 노드 맵 생성"""
        if not self._neighbor_cache:
            neighbor_map = defaultdict(set)
            # 한 번의 순회로 모든 관계 처리
            for rel in knowledge_graph.relationships:
                if rel.get_property("summary_similarity"):
                    neighbor_map[rel.source].add(rel.target)
            self._neighbor_cache = dict(neighbor_map)
        return self._neighbor_cache

    def _find_cluster_from_node(self, start_node: Node, neighbor_map: dict, max_depth: int = 2) -> set:
        """단일 노드에서 시작하는 클러스터 찾기"""
        # 캐시 확인
        cache_key = (start_node.id, max_depth)
        if cache_key in self._cluster_cache:
            return self._cluster_cache[cache_key]

        visited = {start_node}
        current_level = {start_node}
        
        # BFS 사용 (더 효율적인 메모리 사용)
        for depth in range(max_depth):
            next_level = set()
            for node in current_level:
                neighbors = neighbor_map.get(node, set())
                next_level.update(n for n in neighbors if n not in visited)
            visited.update(next_level)
            current_level = next_level
            if not current_level:  # 더 이상 확장할 노드가 없으면 중단
                break

        # 결과 캐싱
        self._cluster_cache[cache_key] = visited
        return visited

    def get_node_clusters(self, knowledge_graph: KnowledgeGraph) -> t.List[t.Set[Node]]:
        """최적화된 클러스터 찾기"""
        # 1. 이웃 노드 맵 구축 (캐시 활용)
        neighbor_map = self._build_neighbor_map(knowledge_graph)
        
        # 2. 병렬 처리를 위한 함수
        def process_node_chunk(nodes):
            return [self._find_cluster_from_node(node, neighbor_map) for node in nodes]

        # 3. 노드를 청크로 분할하여 병렬 처리
        chunk_size = max(1, len(knowledge_graph.nodes) // (4 * 2))  # CPU 코어 수의 2배 정도의 청크
        node_chunks = [
            list(knowledge_graph.nodes)[i:i + chunk_size]
            for i in range(0, len(knowledge_graph.nodes), chunk_size)
        ]

        # 4. 병렬 처리 실행
        all_clusters = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            chunk_results = list(executor.map(process_node_chunk, node_chunks))
            for chunk_result in chunk_results:
                all_clusters.extend(chunk_result)

        # 5. 중복 제거 및 최소 크기 필터링 (set 연산 사용)
        unique_clusters = set()
        min_cluster_size = 2  # 최소 클러스터 크기 설정
        
        for cluster in all_clusters:
            if len(cluster) >= min_cluster_size:
                frozen_cluster = frozenset(cluster)
                unique_clusters.add(frozen_cluster)

        logger.info(f"Found {len(unique_clusters)} unique clusters")
        return [set(cluster) for cluster in unique_clusters]