import numpy as np
import pandas as pd

def norm1(positionVector):
    magnitude = np.linalg.norm(positionVector)

    if magnitude == 0:
        return np.zeros_like(positionVector)
    
    return (positionVector / magnitude).tolist()

def makeBody(json, visual_json):
    body_dict = {}
    for frame in json:
        for key, valueList in visual_json.items():
            for value in valueList:
                name = f'{key}-{value}'

                vector = [frame.landmark[value].x - frame.landmark[int(key)].x,
                          frame.landmark[value].y - frame.landmark[int(key)].y,
                          frame.landmark[value].z - frame.landmark[int(key)].z]
                vector = norm1(vector)
                body_dict[name] = body_dict[name] + [vector] if name in body_dict.keys() else [vector]
            
    return body_dict

def dynamicTimeWarpingPart(vector_list1: list, vector_list2: list) -> np.array: 
    # 출력: 신체 부위에 대한 timewarping matrix
    # 입력: vector_list1이 수강생
    array1 = np.transpose(np.array(vector_list1 * len(vector_list2)).reshape(len(vector_list2), len(vector_list1), 3), (1, 0, 2))
    array2 = np.array(vector_list2 * len(vector_list1)).reshape(len(vector_list1), len(vector_list2), 3)

    distance_array = array1 - array2
    distance_array = np.sqrt(np.square(distance_array[:, :, 0]) + np.square(distance_array[:, :, 1]) + np.square(distance_array[:, :, 2]))

    distance_array_origin = distance_array.copy()

    shape1 = distance_array.shape[0]
    shape2 = distance_array.shape[1]

    for i in range(shape1):
        if i == 0:
            continue
        else:
            distance_array[i, 0] += distance_array[i-1, 0]

    for j in range(shape2):
        if j == 0:
            continue
        else:
            distance_array[0, j] += distance_array[0, j-1]
    
    for i in range(1, shape1):
        for j in range(1, shape2):
            distance_array[i, j] += np.min([distance_array[i-1, j],
                                            distance_array[i, j-1],
                                            distance_array[i-1, j-1]])

    return  distance_array, distance_array_origin

def findPath(result_list):
    result = {'shortestPath' : [],
              'shortestScoring' : []
              }

    result_list = np.array(result_list)
    result_sum = np.sum(result_list, axis=0)

    reference_x = result_sum.shape[1]-1 
    reference_y = result_sum.shape[0]-1
    for i in range(result_sum.shape[0]-1, -1, -1):
        for j in range(result_sum.shape[1]-1, -1, -1):
            if (j > reference_x) & (reference_x != 0):
                continue
            elif (reference_x == 0):
                for tmp_i in range(reference_y, -1, -1):
                    result['shortestPath'].append((tmp_i, 0))
                    result['shortestScoring'].append(result_sum[tmp_i, 0])

                result['shortestPath'].reverse()
                result['shortestScoring'].reverse()
                return result_sum, result

            else:
                tmp = [result_sum[i, j-1], result_sum[i-1, j-1], result_sum[i-1, j]]
                index = tmp.index(min(tmp))
                result['shortestPath'].append((i, j))
                result['shortestScoring'].append(result_sum[i, j])
                if index == 0:                    
                    reference_x = j-1
                    continue
                elif index == 1:
                    reference_x = j-1
                    reference_y = i-1
                    break
                else:
                    reference_y = i-1
                    break
    
    result['shortestPath'].reverse()
    result['shortestScoring'].reverse()
    return result_sum, result

def partScoring(result, body_name, result_origin_list):
    for k, name in enumerate(body_name):
        for i, path in enumerate(result['shortestPath']):
            value = result_origin_list[k][path[0]][path[1]]
            value = round(100 - (100 * (np.mean(value) / 2)), 2)
            result[name] = [value] if i == 0 else result[name] + [value]
            
    return result

def score(result):
    df = {}
    keys_list = list(result.keys())[2:]  # shortestPath, shortestScoring 제외
    
    for i, key in enumerate(keys_list):
        if i == 0:
            df['부위'] = [key]        
            df['평균점수'] = [round(np.mean(result[key]), 2)]
        else:
            df['부위'] += [key]
            df['평균점수'] += [round(np.mean(result[key]), 2)]

    df['부위'] += ['전체']
    df['평균점수'] += [round(sum(df['평균점수']) / len(df['평균점수']), 2)]
    
    return pd.DataFrame(df)


def poseScoring(json1, json2):
    visual_json = {"0": [1, 4], "1": [2], "2": [3], "3": [7], "4": [5], "5": [6], "6": [8], "9": [10], "11": [12, 13, 23], "12": [14, 24], "13": [15], "14": [16], "15": [17, 19, 21], "16": [18, 20, 22], "17": [19], "18": [20], "23": [24, 25], "24": [26], "25": [27], "26": [28], "27": [29, 31], "28": [30, 32], "29": [31], "30": [32]}
    body_dict1 = makeBody(json1, visual_json)
    body_dict2 = makeBody(json2, visual_json)

    result_list = []
    result_origin_list = []
    for body_part in body_dict1.keys():
        distance_array, distance_array_origin = dynamicTimeWarpingPart(body_dict1[body_part], body_dict2[body_part])
        result_list.append(distance_array)
        result_origin_list.append(distance_array_origin)

    result_sum, result = findPath(result_list)

    body_name = list(body_dict1.keys())
    result = partScoring(result, body_name, result_origin_list)  

    df = score(result)
    
    return result_sum, result, df

def analyze_pose(standard_pose, user_pose, standard_dict, standard_scores):  
    result_sum, result, score = poseScoring(standard_pose, user_pose)
    part_list = []
    
    part_mapping = {row['connection_index']: row['connection_name'] for row in standard_scores}

    
    for connection_index in standard_dict.keys():
        if connection_index in score['부위'].values:
            part_score = score.loc[score['부위'] == connection_index, '평균점수'].values[0]
            min_score = standard_dict[connection_index]
            
            if part_score < min_score:
                korean_name = part_mapping.get(connection_index, connection_index)
                part_list.append(korean_name)
        else:
            continue
    
    return result_sum, result, score, part_list