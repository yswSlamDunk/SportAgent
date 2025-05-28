import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def analyze_node_entity(kg, entity_name=None):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 데이터 구조 초기화
    sector_null = {}
    sector_total = {}
    sector_entities = {}
    total_entities = []
    sectors_order = []
    total_node_count = len(kg.nodes)
    null_node_count = 0

    # 데이터 수집
    for node in kg.nodes:
        sector = node.properties['document_metadata']['heading']['heading1']
        if entity_name != None:
            entities = node.properties['exercise_entities'].get(entity_name, [])
        else:
            entities = node.properties.get('entities', [])
        if sector not in sectors_order:
            sectors_order.append(sector)
        if sector not in sector_null:
            sector_null[sector] = 0
            sector_total[sector] = 0
            sector_entities[sector] = []
        sector_total[sector] += 1
        if not entities:
            sector_null[sector] += 1
            null_node_count += 1
        else:
            sector_entities[sector].extend(entities)
            total_entities.extend(entities)

    # 전체 entity 없는 노드 비율 계산
    null_ratio = null_node_count / total_node_count * 100

    # 섹터별 entity 없는 노드 비율 계산
    sector_null_ratio = {sector: sector_null[sector] / sector_total[sector] * 100 for sector in sectors_order}

    # 전체 상위 10개 entity
    total_entity_counts = Counter(total_entities)
    top_total_entities = dict(sorted(total_entity_counts.items(), key=lambda x: x[1], reverse=True)[:10])

    # 섹터별 상위 5개 entity
    sector_top_entities = {}
    for sector in sectors_order:
        counts = Counter(sector_entities[sector])
        sector_top_entities[sector] = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5])

    # 시각화
    plt.figure(figsize=(15, 10))

    # 1. 섹터별 entity 없는 노드 비율 막대그래프
    plt.subplot(2, 2, 1)
    bars1 = plt.bar(sector_null_ratio.keys(), sector_null_ratio.values(), color='skyblue')
    plt.ylabel('비율(%)')
    plt.title(f'섹터별 Entity 없는 노드 비율\n(전체 노드 중 entity 없는 노드 비율: {null_ratio:.1f}%)')
    plt.xticks(rotation=45, ha='right')
    # 막대 위에 값 표시
    for bar in bars1:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}%', 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3pt 위에
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 2. 전체 상위 10개 entity 막대 그래프
    plt.subplot(2, 2, 2)
    bars2 = plt.bar(top_total_entities.keys(), top_total_entities.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('전체 상위 10개 Entity')
    # 막대 위에 값 표시
    for bar in bars2:
        height = bar.get_height()
        plt.annotate(f'{int(height)}', 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 3. 섹터별 상위 entity 히트맵 (깔끔하게)
    plt.subplot(2, 1, 2)
    sectors = list(sector_top_entities.keys())
    all_entities = sorted({e for ents in sector_top_entities.values() for e in ents})
    heatmap_data = np.zeros((len(sectors), len(all_entities)))
    for i, sector in enumerate(sectors):
        for j, entity in enumerate(all_entities):
            heatmap_data[i, j] = sector_top_entities[sector].get(entity, 0)

    im = plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')

    # 컬러바
    cbar = plt.colorbar(im, label='출현 빈도')
    cbar.ax.tick_params(labelsize=10)

    # x, y축 라벨
    plt.xticks(range(len(all_entities)), all_entities, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(sectors)), sectors, fontsize=10)
    plt.xlabel('Entity', fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    plt.title('섹터별 상위 Entity 분포', fontsize=14, pad=15)

    # value annotation (0이 아닌 값만)
    for i in range(len(sectors)):
        for j in range(len(all_entities)):
            value = int(heatmap_data[i, j])
            if value > 0:
                plt.text(j, i, str(value), ha='center', va='center', color='black', fontsize=9, fontweight='bold')

    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # 비율 텍스트로도 출력
    print(f"전체 노드 수: {total_node_count}")
    print(f"Entity가 없는 노드 수: {null_node_count}")
    print(f"전체 노드 중 entity가 없는 노드 비율: {null_ratio:.1f}%")