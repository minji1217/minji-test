# 야호 
import numpy as np
import time 
import config
import utils
import pickle 
from query_builder import QueryBuilder
from embedder import SpecterEmbedder
from retriever import FaissRetriever
from soft_bias import SoftBiasScorer
from cascade import cascade_fusion

class OnlinePaperProcess:
    def __init__(self):
        # 1. 모든 모듈 로드
        print("[모듈 로드 중 ...]")
        start = time.time()
        self.query_builder = QueryBuilder()
        self.embedder = SpecterEmbedder()
        self.retriever = FaissRetriever()
        self.bib_scorer = SoftBiasScorer()
        with open(config.EMBEDDING_DB_PATH, "rb") as f:
            self.embedding_db = pickle.load(f)

        print(f"[로딩 완료 ({time.time() - start: .2f}초 소요)]")

    def run_pipeline(self, user_input):
        '''
        프론트엔드에서 온 데이터 받아서 최종 피처 반환
        - user_input : { "title" : str, "abstract" : str, "context" : str, "bib_ids" : list}
        '''
        req_id = "req_" + utils.get_timestamp()

        # 1. 쿼리 생성 
        p_query, c_query = self.query_builder.build_online_query(
            user_input_text=user_input.get('context', ''),
            title=user_input.get('title', ''),
            abstract=user_input.get('abstract', '')
        )

        # 2. 임베딩 연산
        vecs = self.embedder.encode([p_query, c_query])
        p_vec, c_vec = vecs[0:1], vecs[1:2]

        # 3. FAISS 고속 검색
        p_res = self.retriever.search(p_vec, [req_id], source=["paper"], top_k = config.PAPER_QUERY_TOP_K)
        
        print(f" [Debug] p_res 타입: {type(p_res)}, 길이: {len(p_res[0])}")

        # 4. RRF 융합
        # original
        # fused = rank_fusion(p_res, c_res)[0]
        
        # 가중합 필수 버전
        fused = cascade_fusion(p_res, [c_vec], self.embedding_db)[0]
        print(f" [Debug] fused 개수: {len(fused)}")

        # 5. Soft Bias 계산
        bib_ids = user_input.get('bib_ids', [])
        biased = self.bib_scorer.soft_bias(fused, bib_ids, self.embedding_db)
        print(f" [Debug] biased 개수: {len(biased)}")



        # 6. 피처 정규화 
        raw_sims = [c['sim'] for c in biased]
        raw_bibs = [c.get('bib_score', 0.0) for c in biased]

        norm_sims = utils.normalize(raw_sims)
        norm_bibs = utils.normalize(raw_bibs)

        # 7. 최종 데이터 형식 가공
        clean_candidates = []
        for idx, cand in enumerate(biased):
            clean_candidates.append({
                "paper_id" : str(cand['paper_id']),
                "sim": float(norm_sims[idx]),
                "bib_score": float(norm_bibs[idx])
            })

        return {
            "query_id": req_id,
            "context": user_input.get('context', ''),
            "candidates": clean_candidates
        }
    
if __name__ == "__main__":
    # 서버 객체 생성
    engine = OnlinePaperProcess()
    
    # 프론트엔드에서 받은 가상의 JSON 데이터
    sample = {
    "title": "Towards Autonomous LLM Agents for Recommendation",
    "abstract": "This paper investigates the potential of Large Language Models as autonomous agents in recommendation systems. We focus on overcoming existing controller limitations during autonomous interaction...",
    "context": "LLM 에이전트를 활용한 추천 시스템에서 가장 큰 과제 중 하나는 프로파일 기반 결정 로직을 넘어선 진정한 의미의 자율적 상호작용(autonomous interaction)을 구현하는 것이다. 특히 기존 컨트롤러의 한계를 지적한 연구들을 살펴보면 \\cite{",
    "bib_ids": ["Wang2023_AgentCF", "ToolRec2024"]
    }
    
    print("\n 실시간 피처 추출 시작...")
    start_time = time.time()
    
    # 파이프라인 가동!
    result = engine.run_pipeline(sample)
    
    print(f"처리 완료! 소요 시간: {time.time() - start_time:.4f}초")
    print(f"반환된 후보 개수: {len(result['candidates'])}개")
    
    
    # 리스트가 비어있을 때 IndexError가 나지 않도록 조건문 추가
    if len(result['candidates']) > 0:
        print(f"1번째 후보 샘플: {result['candidates'][0]}")
        print(f"2번째 후보 샘플: {result['candidates'][1]}")
        print(f"3번째 후보 샘플: {result['candidates'][2]}")
        print(f"4번째 후보 샘플: {result['candidates'][3]}")
        print(f"5번째 후보 샘플: {result['candidates'][4]}")
        
    else:
        print("후보가 없습니다! 위쪽 Debug 로그에서 어디서 0이 되었는지 확인하세요.")

    utils.save_json(result, "online_output.json")
    print("output.json에 저장")


    