# Recall@K, MRR 계산 (Retrieval 성능 점검 for 디버깅)
import numpy as np
import config

def calculate_metrics(predicted_ids, gt_ids, k_list = [50, 100, 150, 600]):
    
    # predicted_ids : RRF Score 통해 최종 top-k(100) 결과 리스트 
    # gt_ids : 실제 저자가 인용한 정답 타겟 논문 ID 리스트 
    

    metrics = {}
    targets = set(gt_ids) # for 집합 연산 

    # 정답지가 비어있는 경우 모든 점수 0으로 반환
    if not targets:
        for k in k_list:
            metrics[f"Recall@{k}"] = 0.0
        metrics["MRR"] = 0.0
        return metrics
    
    # 1. Recall@K 계산 : 추천한 상위 K개 안에 정답이 몇개 포함 ? 
    for k in k_list:
        top_k = set(predicted_ids[:k])
        # 후보 논문과 gt 논문 교집합 구함 -> gt 개수로 나눔
        hits = len(top_k.intersection(targets))
        metrics[f"Recall@{k}"] = hits / len(targets)

    # 2. MRR 계산 : 정답이 몇등에서 나왔는가 ? (-> main에서 평균냄)
    mrr = 0.0
    for rank, id in enumerate(predicted_ids, 1): # 1부터 시작
        # 후보 논문이 실제 정답 목록에 있다면, 그 순위의 역수값을 점수로 기록
        if id in targets:
            mrr = 1.0 / rank # 정답을 찾으면 역수 취하고 반복문 종료 
            break
    metrics["MRR"] = mrr

    return metrics

    
