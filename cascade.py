'''import numpy as np 
import config 

def cascade_fusion(p_search_results, c_vecs, p_vecs, embedding_db):
    final_results = []

    for p_res, c_v, p_v in zip(p_search_results, c_vecs, p_vecs):
       c_vec_1d = np.squeeze(c_v)  
       p_vec_1d = np.squeeze(p_v)
       refined_list = []

     # 1. 1차 필터링 결과 후보만 순회함
     for p_item in p_res:
        pid = p_item['paper_id']
        p_sim = p_item['score'] 

        target_vector = embedding_db.get(pid)
        if target_vector is None:
            continue

        # 2. context query 벡터와 후보 논문 벡터 간 유사도 계산 (코사인 유사도)
        c_sim = float(np.dot(c_vec_1d, target_vector))

        # 3. 가중합 (최종 점수)
        final_sim = config.PAPER_SIM_WEIGHT * p_sim + config.CONTEXT_SIM_WEIGHT * c_sim

        refined_list.append({
           "paper_id": pid,
            "sim": final_sim
        })

        # 4. 정렬 후 top-100 선발
        # 4. 정렬 후 Top-100 컷오프
        refined_list.sort(key=lambda x: x['sim'], reverse=True)
        top_100 = refined_list[:config.TOP_K_FINAL] # config에 100으로 설정되어 있다고 가정
        
        # 5. 결과 패키징
        placeholder_results = []
        q_id = p_res[0]['query_id'] if p_res else "UNKNOWN"
        for new_rank, cand in enumerate(top_100):
            placeholder_results.append({
                "query_id": q_id,
                "rank": new_rank + 1,
                "paper_id": cand['paper_id'],
                "sim": cand['sim']
            })
            
        final_results.append(placeholder_results)
        
    return final_results
'''

import numpy as np
import config 

def cascade_fusion(p_search_results, c_vecs, p_vecs, embedding_db):
    final_results = []
    matrix_cache = {} 
    
    for p_res, c_v, p_v in zip(p_search_results, c_vecs, p_vecs):
        c_vec_1d = np.squeeze(c_v)
        
        # 1. paper_query_top_k개의 후보 논문 ID와 FAISS 점수(p_sim)를 리스트로 분리
        p_ids = [item['paper_id'] for item in p_res]
        cache_key = tuple(p_ids)

        # 만약 이미 메모리에 올려둔 행렬이 있다면 새로 안만들고 바로 꺼내 씀 
        if cache_key in matrix_cache:
            target_matrix, valid_p_ids, valid_p_sims = matrix_cache[cache_key]
        else: 
            p_sims = np.array([item['score'] for item in p_res])
        
            # 2. DB에서 paper_query_top_k개의 벡터를 가져와 행렬로 
            target_vectors = []
            valid_indices = []
        
            for idx, pid in enumerate(p_ids):
                vec = embedding_db.get(pid)
                if vec is not None:
                    target_vectors.append(np.squeeze(vec))
                    valid_indices.append(idx)
                
            # 방어 로직 (만약 유효한 벡터가 하나도 없다면 패스)
            if not target_vectors:
                final_results.append([])
                continue
                
            # shape: (3000, 768) 형태의 2차원 행렬 생성
            target_matrix = np.array(target_vectors) 
            valid_p_ids = [p_ids[i] for i in valid_indices]
            valid_p_sims = p_sims[valid_indices]
            
            matrix_cache[cache_key] = (target_matrix, valid_p_ids, valid_p_sims)

        # 3. 3000(paper_query_top_k)번 for문을 돌지 않고, 행렬 곱셈으로 3000개의 유사도를 동시에 계산
        c_sims = np.dot(target_matrix, c_vec_1d) 
        
        # 3-1. paper 점수 0~1 정규화 
        p_min, p_max = np.min(valid_p_sims), np.max(valid_p_sims)
        p_norm = (valid_p_sims - p_min) / (p_max - p_min + 1e-8) # 작은 수 더해서 0으로 나누는 경우 방지 
        

        # 3-2. context 점수 0~1 정규화 
        c_min, c_max = np.min(c_sims), np.max(c_sims)
        c_norm = (c_sims - c_min) / (c_max - c_min + 1e-8) # 반환값 shape: (3000,)
        
        # 4. 가중합도 NumPy로 한 번에 처리
        final_sims = (config.PAPER_SIM_WEIGHT * p_norm) + (config.CONTEXT_SIM_WEIGHT * c_norm)
        
        # 5. 점수가 높은 순서대로 인덱스를 정렬하고 Top-100개만 자름 (초고속 정렬)
        top_indices = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]
        
        # 6. 결과 패키징
        placeholder_results = []
        q_id = p_res[0]['query_id'] if p_res else "UNKNOWN"
        
        for rank, idx in enumerate(top_indices):
            placeholder_results.append({
                "query_id": q_id,
                "rank": rank + 1,
                "paper_id": valid_p_ids[idx],
                "sim": float(final_sims[idx]) 
            })
            
        final_results.append(placeholder_results)
        
    return final_results