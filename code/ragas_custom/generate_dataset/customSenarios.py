from ragas.testset.graph import KnowledgeGraph
import pandas as pd

def make_scenarios(kg: KnowledgeGraph, heading: str = 'heading1'):
    relations = [relation for relation in kg.relationships if relation.type in ['exercise_entities_exercise_name_overlap', 'exercise_entities_exercise_phase_overlap', 'exercise_entities_body_part_overlap']]

    diff = []
    key = []
    rel_type = []

    for relation in relations:
        dict_va =  relation.__dict__
        rel_type.append(dict_va['type']) 
        
        source = dict_va['source'].id.hex
        target = dict_va['target'].id.hex

        key.append('_'.join(sorted([source, target])))
        
        source_head = dict_va['source'].properties['document_metadata']['heading'][heading]
        target_head = dict_va['target'].properties['document_metadata']['heading'][heading]

        diff.append(not (source_head == target_head))
        
    relation_df = pd.DataFrame({'key': key, 'rel_type':rel_type, 'diff': diff})
    df = relation_df.loc[(relation_df['diff'] == True)].groupby(['rel_type'])['key'].unique().reset_index()

    # 각 관계 타입에 대한 데이터를 안전하게 추출
    name_overlap = df[df['rel_type'] == 'exercise_entities_exercise_name_overlap']['key'].iloc[0] if len(df[df['rel_type'] == 'exercise_entities_exercise_name_overlap']) > 0 else set()
    body_overlap = df[df['rel_type'] == 'exercise_entities_body_part_overlap']['key'].iloc[0] if len(df[df['rel_type'] == 'exercise_entities_body_part_overlap']) > 0 else set()
    phase_overlap = df[df['rel_type'] == 'exercise_entities_exercise_phase_overlap']['key'].iloc[0] if len(df[df['rel_type'] == 'exercise_entities_exercise_phase_overlap']) > 0 else set()

    # 각 케이스에 대한 계산
    nameOnly = list(set(name_overlap) - set(body_overlap) - set(phase_overlap))
    bodyOnly = list(set(body_overlap) - set(name_overlap) - set(phase_overlap))
    namePhase = list(set(name_overlap).intersection(set(phase_overlap)))
    nameBody = list(set(name_overlap).intersection(set(body_overlap)))

    return nameOnly, bodyOnly, namePhase, nameBody