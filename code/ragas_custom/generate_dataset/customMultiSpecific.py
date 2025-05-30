from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import random

from ragas.testset.synthesizers.multi_hop import MultiHopQuerySynthesizer, MultiHopScenario
from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.base import QueryStyle, QueryLength
from ragas.prompt import PydanticPrompt
from langchain.callbacks.base import Callbacks

@dataclass
class MultiRelationMultiHopScenario(MultiHopQuerySynthesizer):
    """
    Multiple relation types를 처리하는 MultiHopQuerySynthesizer
    """
    name: str = None
    relation_types: List[str] = field(default_factory=list)
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    relation_list: List[str] = field(default_factory=list)
    heading: str = 'heading1'

    def __post_init__(
        self,
        name: str = None,
        relation_types: List[str] = None,
        relation_list: List[str] = None,
        heading: str = None,
    ):
        if name is not None:
            self.name = name
        if relation_types is not None:
            self.relation_types = relation_types
        if relation_list is not None:
            self.relation_list = relation_list
        if heading is not None:
            self.heading = heading


    def filter_keywords_by_statistics(
        self,
        count_dict: Dict[str, int],
        min_frequency: int = 2,
        percentile_threshold: float = 0.25
    ) -> Dict[str, int]:
        if not count_dict:
            return {}

        # 최소 빈도수로 필터링
        filtered_dict = {
            k: v for k, v in count_dict.items() 
            if v >= min_frequency
        }

        if not filtered_dict:
            return {}

        # 빈도수 값들의 평균과 표준편차 계산
        values = list(filtered_dict.values())
        mean = sum(values) / len(values)

        # 표준편차가 0인 경우 처리
        if len(set(values)) == 1:  # 모든 값이 동일한 경우
            return filtered_dict  # 모든 값을 그대로 반환
    
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        # z-score 계산 및 하위 percentile_threshold% 제거
        z_scores = {k: (v - mean) / std for k, v in filtered_dict.items()}
        cutoff_z = sorted(z_scores.values())[int(len(z_scores) * percentile_threshold)]
        
        # z-score가 cutoff_z보다 큰 키워드만 유지
        filtered_dict = {
            k: v for k, v in filtered_dict.items()
            if z_scores[k] > cutoff_z
        }
        
        return filtered_dict

    def get_node_clusters(self, knowledge_graph: KnowledgeGraph) -> List[Tuple]:
        """
        Get node clusters based on relation types and relation list
        """
        node_clusters = knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                rel.type in self.relation_types and 
                rel.source.id.hex in [r.split('_')[0] for r in self.relation_list] and
                rel.target.id.hex in [r.split('_')[1] for r in self.relation_list] and
                rel.source.properties['document_metadata']['heading'][self.heading] != rel.target.properties['document_metadata']['heading'][self.heading]
            )
        )

        # source-target 쌍별로 관계 타입을 그룹화
        node_groups = {}
        for node in node_clusters:
            source_id = node[0].id.hex
            target_id = node[2].id.hex
            rel_type = node[1].type
            key = (source_id, target_id)
            
            if key not in node_groups:
                node_groups[key] = set()
            node_groups[key].add(rel_type)
        
        # 모든 relation_types를 포함하는 source-target 쌍만 필터링
        valid_keys = {
            key for key, rel_types in node_groups.items() 
            if set(self.relation_types).issubset(rel_types)
        }
        
        return [
            node for node in node_clusters 
            if (node[0].id.hex, node[2].id.hex) in valid_keys
        ]

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: List[Persona],
        callbacks: Callbacks,
    ) -> List[MultiHopScenario]:
        # 1. 노드 클러스터 가져오기
        node_clusters = self.get_node_clusters(knowledge_graph)

        if len(node_clusters) == 0:
            print(f"{self.name}: No clusters found in the knowledge graph. Check relation types and relation list.")
            return []
            # raise ValueError(
            #     "No clusters found in the knowledge graph. Check relation types and relation list."
            # )
        
        # 2. 관계 타입별로 카운트 딕셔너리와 캐시 초기화
        count_dict = {rel_type: {} for rel_type in self.relation_types}
        overlapped_item_cache = {rel_type: {} for rel_type in self.relation_types}

        # 3. overlapped_items 수집 및 카운팅
        for rel in node_clusters:
            rel_type = rel[1].type
            overlap_items = rel[1].properties['overlapped_items']
            
            
            unique_items = list(set(sum(overlap_items, [])))
            overlap_item = random.choice(unique_items)
            
            if overlap_item in count_dict[rel_type]:
                count_dict[rel_type][overlap_item] += 1
                overlapped_item_cache[rel_type][overlap_item].append(rel)
            else:
                count_dict[rel_type][overlap_item] = 1
                overlapped_item_cache[rel_type][overlap_item] = [rel]

        # 4. 각 관계 타입별로 필터링
        keywords = []
        for rel_type in self.relation_types:
            count_dict[rel_type] = self.filter_keywords_by_statistics(
                count_dict[rel_type],
                min_frequency=2,
                percentile_threshold=0.25
            )
            overlapped_item_cache[rel_type] = {
                k: v for k, v in overlapped_item_cache[rel_type].items() 
                if k in count_dict[rel_type]
            }
            keywords.extend(list(count_dict[rel_type].keys()))
        
        # 5. 페르소나-테마 매칭
        prompt_input = ThemesPersonasInput(
            themes=keywords,
            personas=persona_list
        )
        persona_concepts = await self.theme_persona_matching_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks
        )

        # 6. 페르소나별 relation_type 키워드 매핑
        persona_relation_keywords = {}
        for persona_name, persona_themes in persona_concepts.mapping.items():
            persona_relation_keywords[persona_name] = {}
            
            for rel_type in self.relation_types:
                rel_type_keywords = count_dict[rel_type].keys()
                
                # 완벽한 키워드 매칭만 허용
                matching_keywords = [
                    keyword for keyword in rel_type_keywords 
                    if any(keyword.lower() == theme.lower() for theme in persona_themes)
                ]
                
                persona_relation_keywords[persona_name][rel_type] = matching_keywords

        # 7. 유효한 페르소나 필터링
        valid_personas = {
            persona_name: rel_types 
            for persona_name, rel_types in persona_relation_keywords.items()
            if all(len(keywords) > 0 for keywords in rel_types.values())
        }

        if not valid_personas:
            print(f"{self.name}: No valid persona-keyword combinations found. Check if the themes match with relation types.")
            return []
            # raise ValueError(
            #     "No valid persona-keyword combinations found. Check if the themes match with relation types."
            # )
        
        # 8. 시나리오 생성
        scenarios = []
        persona_scenario_counts = {persona_name: 0 for persona_name in valid_personas.keys()}

        # relation_list를 랜덤하게 섞기
        random.shuffle(self.relation_list)
        
        for relation in self.relation_list:
            source = relation.split('_')[0]
            target = relation.split('_')[1]

            # 9. 관계 검색
            searched_relations = knowledge_graph.find_two_nodes_single_rel(
                relationship_condition=lambda rel: (
                    rel.type in self.relation_types and 
                    rel.source.id.hex == source and
                    rel.target.id.hex == target
                )
            )

            # 10. 관계 타입별 overlapped_items 수집
            relation_items = {}
            for rel in searched_relations:
                source_node, relationship, target_node = rel
                rel_type = relationship.type
                overlap_items = relationship.properties['overlapped_items']

                unique_items = list(set(sum(overlap_items, [])))
                overlap_item = random.choice(unique_items)
                relation_items[rel_type] = {
                    'items': [overlap_item],
                    'nodes': (source_node, target_node)
                }

            # 11. 필요한 모든 relation_type이 있는지 확인
            if not relation_items or not all(rel_type in relation_items for rel_type in self.relation_types):
                continue

            # 12. 각 페르소나에 대해 시나리오 생성 시도
            for persona_name, rel_type_keywords in valid_personas.items():
                valid_combinations = {}
                
                # 13. 각 relation_type에 대해 완벽한 키워드 매칭 확인
                for rel_type, keywords in rel_type_keywords.items():
                    if rel_type in relation_items:
                        item = relation_items[rel_type]['items'][0]
                        matching_keyword = next(
                            (keyword for keyword in keywords 
                             if keyword.lower() == item.lower()),
                            None
                        )
                        if matching_keyword:
                            valid_combinations[rel_type] = {
                                'keyword': matching_keyword,
                                'nodes': relation_items[rel_type]['nodes']
                            }

                # 14. 모든 relation_type에 대해 유효한 조합이 있는 경우 시나리오 생성
                if all(rel_type in valid_combinations for rel_type in self.relation_types):
                    style = random.choice(list(QueryStyle))
                    length = random.choice(list(QueryLength))
                    
                    ordered_combinations = [
                        valid_combinations[rel_type]['keyword'] 
                        for rel_type in self.relation_types
                    ]
                    
                    selected_persona = next((p for p in persona_list if p.name == persona_name), None)
                    if selected_persona is None:
                        raise ValueError(f"Persona with name {persona_name} not found in persona_list")

                    scenario = MultiHopScenario(
                        nodes=[valid_combinations[self.relation_types[0]]['nodes'][0], 
                            valid_combinations[self.relation_types[0]]['nodes'][1]],
                        combinations=ordered_combinations,
                        persona=selected_persona,
                        style=style,
                        length=length
                    )
                    scenarios.append(scenario)
                    persona_scenario_counts[persona_name] += 1
                    break

            if len(scenarios) >= n:
                break

        return scenarios[:n]

@dataclass
class SingleRelationMultiHopScenario(MultiHopQuerySynthesizer):
    """
    Single relation type을 처리하는 MultiHopQuerySynthesizer
    """
    name: str = None
    relation_type: str = None
    relation_list: List[str] = field(default_factory=list)
    theme_persona_matching_prompt: PydanticPrompt = ThemesPersonasMatchingPrompt()
    heading: str = 'heading1'

    def __post_init__(
        self,
        name: str = None,
        relation_type: str = None,
        relation_list: List[str] = None,
        heading: str = None,
    ):
        if name is not None:
            self.name = name
        if relation_type is not None:
            self.relation_type = relation_type
        if relation_list is not None:
            self.relation_list = relation_list
        if heading is not None:
            self.heading = heading

    def get_node_clusters(self, knowledge_graph: KnowledgeGraph) -> List[Tuple]:
        """
        Get node clusters based on relation type and relation list
        """
        return knowledge_graph.find_two_nodes_single_rel(
            relationship_condition=lambda rel: (
                rel.type == self.relation_type and 
                rel.source.id.hex in [r.split('_')[0] for r in self.relation_list] and
                rel.target.id.hex in [r.split('_')[1] for r in self.relation_list] and
                rel.source.properties['document_metadata']['heading'][self.heading] != rel.target.properties['document_metadata']['heading'][self.heading]
            )
        )
    
    def filter_keywords_by_statistics(
        self,
        count_dict: Dict[str, int],
        min_frequency: int = 2,
        percentile_threshold: float = 0.25
    ) -> Dict[str, int]:
        if not count_dict:
            return {}

        # 최소 빈도수로 필터링
        filtered_dict = {
            k: v for k, v in count_dict.items() 
            if v >= min_frequency
        }

        if not filtered_dict:
            return {}

        # 빈도수 값들의 평균과 표준편차 계산
        values = list(filtered_dict.values())
        mean = sum(values) / len(values)
        # 표준편차가 0인 경우 처리
        
        if len(set(values)) == 1:  # 모든 값이 동일한 경우
            return filtered_dict  # 모든 값을 그대로 반환
        
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

        # z-score 계산 및 하위 percentile_threshold% 제거
        z_scores = {k: (v - mean) / std for k, v in filtered_dict.items()}
        cutoff_z = sorted(z_scores.values())[int(len(z_scores) * percentile_threshold)]
        
        # z-score가 cutoff_z보다 큰 키워드만 유지
        filtered_dict = {
            k: v for k, v in filtered_dict.items()
            if z_scores[k] > cutoff_z
        }
        
        return filtered_dict

    async def _generate_scenarios(
        self,
        n: int,
        knowledge_graph: KnowledgeGraph,
        persona_list: List[Persona],
        callbacks: Callbacks,
    ) -> List[MultiHopScenario]:
        # 1. 노드 클러스터 가져오기
        node_clusters = self.get_node_clusters(knowledge_graph)
        
        if len(node_clusters) == 0:
            print(f"{self.name}: No clusters found in the knowledge graph. Check relation types and relation list.")
            return []

        # 2. 카운트 딕셔너리와 캐시 초기화
        count_dict = {}
        overlapped_item_cache = {}

        # 3. overlapped_items 수집 및 카운팅
        for rel in node_clusters:
            overlap_items = rel[1].properties['overlapped_items']
            
            unique_items = list(set(sum(overlap_items, [])))
            overlap_item = random.choice(unique_items)
            
            if overlap_item in count_dict:
                count_dict[overlap_item] += 1
                overlapped_item_cache[overlap_item].append(rel)
            else:
                count_dict[overlap_item] = 1
                overlapped_item_cache[overlap_item] = [rel]

        # 4. 키워드 필터링
        count_dict = self.filter_keywords_by_statistics(
            count_dict,
            min_frequency=2,
            percentile_threshold=0.25
        )
        overlapped_item_cache = {
            k: v for k, v in overlapped_item_cache.items() 
            if k in count_dict
        }
        keywords = list(count_dict.keys())

        # 5. 페르소나-테마 매칭
        prompt_input = ThemesPersonasInput(
            themes=keywords,
            personas=persona_list
        )
        persona_concepts = await self.theme_persona_matching_prompt.generate(
            data=prompt_input,
            llm=self.llm,
            callbacks=callbacks
        )

        # 6. 페르소나별 키워드 매핑
        persona_keywords = {}
        for persona_name, persona_themes in persona_concepts.mapping.items():
            matching_keywords = [
                keyword for keyword in keywords 
                if any(keyword.lower() == theme.lower() for theme in persona_themes)
            ]
            if matching_keywords:
                persona_keywords[persona_name] = matching_keywords

        if not persona_keywords:
            print(f"{self.name}: No valid persona-keyword combinations found. Check if the themes match with relation types.")
            return []
            # raise ValueError("No valid persona-keyword combinations found.")

        # 7. 시나리오 생성
        scenarios = []
        persona_scenario_counts = {persona_name: 0 for persona_name in persona_keywords.keys()}

        # 모든 가능한 조합을 미리 생성
        valid_combinations = []
        for persona_name, keywords in persona_keywords.items():
            for keyword in keywords:
                if keyword in overlapped_item_cache:
                    for rel in overlapped_item_cache[keyword]:
                        valid_combinations.append({
                            'persona_name': persona_name,
                            'keyword': keyword,
                            'nodes': (rel[0], rel[2])
                        })

        # 조합을 랜덤하게 섞기
        random.shuffle(valid_combinations)

        # 시나리오 생성
        for combo in valid_combinations:
            if len(scenarios) >= n:
                break

            persona_name = combo['persona_name']
            keyword = combo['keyword']
            nodes = combo['nodes']

            selected_persona = next((p for p in persona_list if p.name == persona_name), None)
            if selected_persona is None:
                continue

            style = random.choice(list(QueryStyle))
            length = random.choice(list(QueryLength))

            scenario = MultiHopScenario(
                nodes=[nodes[0], nodes[1]],
                combinations=[keyword],
                persona=selected_persona,
                style=style,
                length=length
            )
            scenarios.append(scenario)
            persona_scenario_counts[persona_name] += 1

        return scenarios[:n]