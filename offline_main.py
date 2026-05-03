'''
import numpy as np
import config 
import time 
from tqdm import tqdm
import pickle
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from fusion_var import rank_fusion_var
from evaluate import calculate_metrics
from cascade import cascade_fusion


def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db, paper_query_top_k = config.PAPER_QUERY_TOP_K):
    # paper_batch : eval_data에서 32개 논문 가져온 리스트 (json 형태)
    # 1. Flatten : 논문 32개 각각의 모든 context를 1차원 리스트로 모음
    paper_query_list = []
    context_query_list = []   # context query 저장 
    metadata_list = []        # 메타데이터 보관소 (QueryBuilder가 만든 딕셔너리 결과물)

    for item in paper_batch:
        paper_id = item.get('paper_id', '')

        # QueryBuilder를 통해 paper query 1개, context query N개 추출 
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), item.get('abstract',''), item.get('all_references', [])
        )


        # 해당 논문의 모든 인용구([CITE:])를 context_query_list에 저장
        for sample in context_queries:
            # [for 초기 데이터] db에 존재하는 진짜 정답만 추려냄 
            valid_targets = [tid for tid in sample['target_ids'] if tid in embedding_db]

            # [for 초기 데이터] 
            if not valid_targets: continue 

            # [for 초기 데이터] 
            sample['target_ids'] = valid_targets

            paper_query_list.append(paper_query)
            context_query_list.append(sample['context_query'])
            metadata_list.append(sample) 

    total_samples = len(context_query_list)
    print(f"context 개수: {len(context_query_list)}")

    # 가져온 논문 32개 모두 인용구 하나도 없다면 패스 
    if total_samples == 0: return []

    # 2. 배치 임베딩
    # 2-1. context query 한 번에 임베딩 
    # 이때, embedder 내부에서 batch_size(예: 64) 단위로 쪼개어 연산 후 붙여줌
    p_vectors = embedder.encode(paper_query_list)
    c_vectors = embedder.encode(context_query_list)
    query_ids = [m['query_id'] for m in metadata_list]

    # 2-2. 전역 쿼리는 중복 제거한 paper_batch(예: 32) 개수만큼 인코딩
    # 기존 : {"paper_id : paper_query", " : ", ...}
    # 적용 후 : [[, , ,], [, , ,], ...]

    # 3. 오프라인 필터링 
    # context로 전체 후보 풀 다 뒤지지 않고, paper query로 먼저 FAISS 검색해서 후보 풀 추림 (paper_query_top_k개)
    p_search_results = retriever.search(p_vectors, query_ids, source = ["paper"] * total_samples, top_k = paper_query_top_k)
    
    # ==========================================================
    # 🔍 [Stage 1 Debugging] 여기서 1차 성적표를 확인하자!
    # ==========================================================
    p_captured_count = 0
    for i in range(total_samples):
        # 1차 검색으로 가져온 논문 ID들 리스트업
        retrieved_ids = [res['paper_id'] for res in p_search_results[i]]
        # 진짜 정답(target_ids) 리스트
        gt_ids = metadata_list[i]['target_ids']
        
        # 정답 중 하나라도 1차 결과(예: 3000개) 안에 들어있는지 확인
        if any(tid in retrieved_ids for tid in gt_ids):
            p_captured_count += 1
            
    stage1_recall = p_captured_count / total_samples if total_samples > 0 else 0
    print(f"\n📢 [Stage 1 Debug] Recall@{paper_query_top_k}: {stage1_recall:.4f}")
    # ==========================================================

    # 4. 온라인 정밀 타격 (추려진 paper_query_top_k개 안에서만 내적하여 최종 top-100 선발)
    all_fused_results = cascade_fusion(p_search_results, c_vectors, embedding_db)
    
    final_output_for_next = [] # 다음 단계에 제공

    # 5. Soft Bias 적용 및 최종 피처 패키징 
    for i in range(total_samples):
        meta = metadata_list[i]
        
        candidates = all_fused_results[i]

        # FAISS DB에 존재하는 유저 인용 기록만 남김 
        raw_bibs = meta.get('bib_ids', [])
        valid_user_bibs = [b for b in raw_bibs if b in embedding_db] # db에 존재하는 bib만 남김

        # soft bias 점수 계산
        biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
        # sim, bib_score 정규화
        norm_sims = np.array([c['sim'] for c in biased])
        raw_bibs = np.array([c.get('bib_score', 0.0) for c in biased])
    
        # bib_score 정규화 (Min-Max)
        b_min, b_max = np.min(raw_bibs), np.max(raw_bibs)
        # 만약 bib_score가 전부 0이라서 max=0, min=0인 경우를 대비한 방어 로직
        if b_max == b_min:
            norm_bibs = np.zeros_like(raw_bibs)
        else:
            norm_bibs = (raw_bibs - b_min) / (b_max - b_min + 1e-9)

        # top-100 각 논문에 대해 필요한 피처만 추출
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id": cand['paper_id'],
                # "rrf_score": cand['rrf_score'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        query_packet = {
            "query_id": meta['query_id'],
            'target_ids': meta['target_ids'],
            'context': meta['context_query'],
            'candidates': clean_candidates
        }
        
        final_output_for_next.append(query_packet)

    return final_output_for_next

        





def run_pipeline(data_path, paper_batch_size):
    
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 동작
    
    print(f"[Offline 실험용 추천 파이프라인 가동 시작...] (데이터: {data_path}")
    start_time = time.time()

    # 1. 모듈 생성 
    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    # 2. 데이터셋 로드 (정답지 포함된 JSON 파일)
    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    all_processed_queries = [] # 모든 배치를 1차원으로 통합할 리스트 (할지말지 고민)

    print(f"총 논문 개수 : {total_papers}개 (논문 {paper_batch_size}개씩 묶어서 처리)")

    # 전체 데이터 global metrics 누적 변수 초기화 
    total_queries_so_far = 0
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "Recall@600": 0.0, "MRR": 0.0}

    # 3. 데이터셋 순회하며 파이프라인 실행 (paper_batch_size(예: 32) 단위로 쪼갬)
    for i in tqdm(range(0, total_papers, paper_batch_size), desc="논문 배치 처리중..."):
        paper_batch = eval_data[i : i + paper_batch_size]
        # 논문 100개, batch : 32일때 마지막 루프 i=96일땐 96~128(96+32)가 아닌 96~100이어야하므로 min 취함 
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        # 배치 단위 성능 평가 로직 
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            batch_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150":0.0, "Recall@600": 0.0, "MRR": 0.0}

            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']

                
                # 쿼리당 채점 
                metrics = calculate_metrics(predicted_ids, gt_ids)


                # 배치 및 global metrics에 누적 
                for key in global_metrics:
                    batch_metrics[key] += metrics[key]
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count

            # 배치 평균 성능 출력
            # print(f"[Batch 성능] Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
            print(f"[Batch 성능] Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | Recall@150: {batch_metrics['Recall@150'] / batch_queries_count:.4f} | Recall@600: {batch_metrics['Recall@600'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
        all_processed_queries.extend(batch_results)# 다음 파트에 합치기 (batch_results 이용할지 말지)
    
    # 모든 배치가 끝난 후 최종 전체 성능 평가 결과 출력
    if total_queries_so_far > 0:
        print("\n" + "="*45)
        print(f"최종 전체 성능 (Total Queries: {total_queries_so_far}개)")
        print("="*45)
        for key in global_metrics:
            final_avg = global_metrics[key] / total_queries_so_far
            print(f" - {key}: {final_avg:.4f}")
        print("="*45 + "\n")
   
    print(f"총 소요시간 : {time.time() - start_time: .2f}초")

    return all_processed_queries

    

if __name__ == "__main__":
    final_data = run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)
    utils.save_json(final_data, "offline_output.json") 
    print("'offline_output.json' 저장 완료")

'''

import numpy as np
import config 
import time 
from tqdm import tqdm
import pickle
import utils
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from evaluate import calculate_metrics

def process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db):
    final_output_for_next = []
    
    # [Step 1] 논문 단위로 순회 (중복 검색 방지)
    for item in paper_batch:
        paper_id = item.get('paper_id', '')
        
        # QueryBuilder를 통해 1개의 paper_query와 N개의 context_queries 추출
        paper_query, context_queries = query_builder.build_offline_query(
            paper_id, item.get('full_text',''), item.get('title', ''), 
            item.get('abstract',''), item.get('all_references', [])
        )

        # DB에 정답이 있는 유효한 문맥만 필터링
        valid_contexts = []
        for sample in context_queries:
            vt = [tid for tid in sample['target_ids'] if tid in embedding_db]
            if vt:
                sample['target_ids'] = vt
                valid_contexts.append(sample)

        if not valid_contexts:
            continue

        # [Step 2] Stage 1: 오프라인 필터링 (논문당 딱 1번 실행)
        # Title Boosting 적용: title [SEP] abstract (query_builder에서 처리)
        p_vec = embedder.encode([paper_query]) 
        p_res = retriever.search(p_vec, [paper_id], top_k=config.PAPER_QUERY_TOP_K)[0]
        
        # 후보 5,000개의 벡터를 DB에서 한꺼번에 추출 (배치 처리)
        p_ids = [res['paper_id'] for res in p_res]
        # 리스트 컴프리헨션으로 I/O 속도 극대화
        valid_data = [(i, embedding_db[pid]) for i, pid in enumerate(p_ids) if pid in embedding_db]
        
        if not valid_data:
            continue
            
        v_indices, t_vectors = zip(*valid_data)
        target_matrix = np.array(t_vectors).squeeze() # Shape: (5000, 768)
        valid_p_sims = np.array([p_res[i]['score'] for i in v_indices])
        valid_p_ids = [p_ids[i] for i in v_indices]

        # [Step 3] Stage 2: 온라인 정밀 타격 (행렬 연산으로 모든 문맥 한꺼번에 계산)
        c_queries = [ctx['context_query'] for ctx in valid_contexts]
        c_vecs = embedder.encode(c_queries) # 문맥들 배치 인코딩
        
        # 모든 문맥에 대해 한꺼번에 유사도 계산 (Matrix Multiplication)
        # c_sims_all shape: (문맥개수, 5000)
        c_sims_all = np.dot(c_vecs, target_matrix.T)

        # [Step 4] 문맥별로 최종 순위 계산 및 패키징
        for i, sample in enumerate(valid_contexts):
            c_sims = c_sims_all[i]
            
            # Z-Score 정규화 (안정적인 결합)
            p_norm = (valid_p_sims - np.mean(valid_p_sims)) / (np.std(valid_p_sims) + 1e-8)
            c_norm = (c_sims - np.mean(c_sims)) / (np.std(c_sims) + 1e-8)

            # 0.6 돌파를 위한 가중치 적용 (문맥에 압도적 비중)
            final_sims = (config.PAPER_SIM_WEIGHT * p_norm) + (config.CONTEXT_SIM_WEIGHT * c_norm)
            
            # Top-K 정렬
            top_idx = np.argsort(final_sims)[::-1][:config.TOP_K_FINAL]
            
            candidates = []
            for rank, idx in enumerate(top_idx):
                candidates.append({
                    "paper_id": valid_p_ids[idx],
                    "sim": float(final_sims[idx])
                })

            # Soft Bias (기존 로직 유지)
            raw_bibs = sample.get('bib_ids', [])
            valid_user_bibs = [b for b in raw_bibs if b in embedding_db]
            biased = bib_scorer.soft_bias(candidates, valid_user_bibs, embedding_db)
            
            # 최종 피처 정리
            norm_sims = np.array([c['sim'] for c in biased])
            raw_scores = np.array([c.get('bib_score', 0.0) for c in biased])
            
            b_min, b_max = np.min(raw_scores), np.max(raw_scores)
            norm_bibs = (raw_scores - b_min) / (b_max - b_min + 1e-9) if b_max > b_min else np.zeros_like(raw_scores)

            clean_candidates = [{
                "paper_id": cand['paper_id'],
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            } for idx, cand in enumerate(biased)]

            final_output_for_next.append({
                "query_id": sample['query_id'],
                "target_ids": sample['target_ids'],
                "context": sample['context_query'],
                "candidates": clean_candidates
            })

    return final_output_for_next

def run_pipeline(data_path, paper_batch_size):
    print(f"[최적화 파이프라인 가동] 데이터: {data_path}")
    start_time = time.time()

    query_builder = QueryBuilder()
    embedder = SpecterEmbedder()
    retriever = FaissRetriever()
    bib_scorer = SoftBiasScorer()

    eval_data = utils.load_json(data_path)
    with open(config.EMBEDDING_DB_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    total_papers = len(eval_data)
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "Recall@150": 0.0, "MRR": 0.0}
    total_queries_so_far = 0

    for i in tqdm(range(0, total_papers, paper_batch_size), desc="배치 처리 중"):
        paper_batch = eval_data[i : i + paper_batch_size]
        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            for q_data in batch_results:
                predicted_ids = [cand['paper_id'] for cand in q_data['candidates']]
                gt_ids = q_data['target_ids']
                metrics = calculate_metrics(predicted_ids, gt_ids)

                for key in global_metrics:
                    global_metrics[key] += metrics[key]
            
            total_queries_so_far += batch_queries_count
            print(f"현재 Recall@150: {global_metrics['Recall@150'] / total_queries_so_far:.4f}")

    print("\n" + "="*45)
    for key, val in global_metrics.items():
        print(f" 최종 {key}: {val / total_queries_so_far:.4f}")
    print("="*45)
    print(f"총 소요시간: {time.time() - start_time:.2f}초")

    return []

if __name__ == "__main__":
    run_pipeline(config.EVAL_DATA_PATH, config.PAPER_BATCH_SIZE)