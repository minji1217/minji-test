import numpy as np
import config 
import time 
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
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = raw_sims
        norm_bibs = utils.normalize(raw_bibs)

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
    '''
    [동작 방식] 전체 데이터셋을 논문 단위로 쪼개고, 논문 내에서도 context 단위로 쪼개어 동작
    '''
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
    global_metrics = {"Recall@50": 0.0, "Recall@100": 0.0, "MRR": 0.0}

    # 3. 데이터셋 순회하며 파이프라인 실행 (paper_batch_size(예: 32) 단위로 쪼갬)
    for i in range(0, total_papers, paper_batch_size):
        paper_batch = eval_data[i : i + paper_batch_size]
        # 논문 100개, batch : 32일때 마지막 루프 i=96일땐 96~128(96+32)가 아닌 96~100이어야하므로 min 취함 
        print(f"처리 중 ... 논문 [{i} ~ {min(i + paper_batch_size, total_papers)}] / {total_papers}")

        batch_results = process_paper_batch(paper_batch, query_builder, embedder, retriever, bib_scorer, embedding_db)
        
        # 배치 단위 성능 평가 로직 
        batch_queries_count = len(batch_results)
        if batch_queries_count > 0:
            batch_metrics = {"Recall@10": 0.0, "Recall@50": 0.0, "Recall@100": 0.0, "MRR": 0.0}

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
            print(f"[Batch 성능] Recall@50: {batch_metrics['Recall@50'] / batch_queries_count:.4f} | Recall@100: {batch_metrics['Recall@100'] / batch_queries_count:.4f} | MRR: {batch_metrics['MRR'] / batch_queries_count:.4f}")
        
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