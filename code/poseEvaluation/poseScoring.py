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
    for i, key in enumerate(list(result.keys())[2:]):
        if i == 0:
            df['부위'] = [key]        
            df['평균점수'] = [round(np.mean(result[key]), 2)]
        else:
            df['부위'] += [key]
            df['평균점수'] += [round(np.mean(result[key]), 2)]

    df['부위'] += ['전체']
    df['평균점수'] += [round(sum(df['평균점수']) / len(df['평균점수']), 2)]

    return pd.DataFrame(df)


def poseScoring(json1, json2, visual_json):
    body_dict1 = makeBody(json1, visual_json)
    body_dict2 = makeBody(json2, visual_json)

    result_list = []
    result_origin_list = []
    for i, body_part in enumerate(body_dict1.keys()):
        distance_array, distance_array_origin = dynamicTimeWarpingPart(body_dict1[body_part], body_dict2[body_part])
        result_list.append(distance_array)
        result_origin_list.append(distance_array_origin)

    result_sum, result = findPath(result_list)

    body_name = list(body_dict1.keys())
    result = partScoring(result, body_name, result_origin_list)  

    df = score(result)
    
    return result_sum, result, df

def rename_point(part):
    points = {
        0: '코',
        1: '왼쪽 눈 (안쪽)',
        2: '왼쪽 눈',
        3: '왼쪽 눈 (바깥쪽)',
        4: '오른쪽 눈 (안쪽)',
        5: '오른쪽 눈',
        6: '오른쪽 눈 (바깥쪽)',
        7: '왼쪽 귀',
        8: '오른쪽 귀',
        9: '입 (왼쪽)',
        10: '입 (오른쪽)',
        11: '왼쪽 어깨',
        12: '오른쪽 어깨',
        13: '왼쪽 팔꿈치',
        14: '오른쪽 팔꿈치',
        15: '왼쪽 손목',
        16: '오른쪽 손목',
        17: '왼쪽 새끼손가락',
        18: '오른쪽 새끼손가락',
        19: '왼쪽 검지',
        20: '오른쪽 검지',
        21: '왼쪽 엄지',
        22: '오른쪽 엄지',
        23: '왼쪽 엉덩이',
        24: '오른쪽 엉덩이',
        25: '왼쪽 무릎',
        26: '오른쪽 무릎',
        27: '왼쪽 발목',
        28: '오른쪽 발목',
        29: '왼쪽 발뒤꿈치',
        30: '오른쪽 발뒤꿈치',
        31: '왼쪽 발끝',
        32: '오른쪽 발끝'
    }
    indices = part.split('-')
    if part == '전체':
        return '전체'
    return f"{points.get(int(indices[0]), '알 수 없는 부위')}-{points.get(int(indices[1]), '알 수 없는 부위')}"

def diagnose_clean(new_pose, old_pose, visual_json):
    result_sum, result, score = poseScoring(new_pose, old_pose, visual_json)

    standard_dict = {'11-12': 83.58,
                     '11-13': 83.94,
                     '11-23': 92.04,
                     '12-14': 81.97,
                     '12-24': 91.46,
                     '13-15': 82.98,
                     '14-16': 82.75,
                     '23-24': 83.17,
                     '23-25': 87.52,
                     '24-26': 86.59,
                     '25-27': 91.78,
                     '26-28': 91.83,
                     '전체': 85.99}

    part_list = []
    
    for part in standard_dict.keys():
        part_score = score.loc[score['부위'] == part, '평균점수'].values[0]
        min_score = standard_dict[part]

        if part_score < min_score:
            part_list.append(rename_point(part))

    return result_sum, result, score, part_list
    