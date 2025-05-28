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

    nameOnly = list(set(df.iloc[1, 1]) - set(df.iloc[0, 1]) - set(df.iloc[2, 1]))
    bodyOnly = list(set(df.iloc[0, 1]) - set(df.iloc[1, 1]) - set(df.iloc[2, 1]))
    namePhase = list(set(df.iloc[1, 1]).intersection(set(df.iloc[2, 1])))
    nameBody = list(set(df.iloc[1, 1]).intersection(set(df.iloc[0, 1])))

    return nameOnly, bodyOnly, namePhase, nameBody