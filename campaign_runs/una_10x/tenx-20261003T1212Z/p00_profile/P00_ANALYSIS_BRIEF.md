# P00 분석 브리핑 — 외부 최적화 리뷰용 (2026-10-04)

목적: ChatGPT 등 외부 분석기에 P00 측정 사실을 전달하고, **중요 지역의 최적화 방안**을 정리하기 위한 자기완결 브리핑.
모든 수치는 이 디렉터리의 JSON 증거에서 나왔으며, 출처 파일은 §6에 정리되어 있다.

## 1. 문맥과 목표

- 캠페인: `una_10x` — **설치된 공개 RunBatch API** 완전 처리량 ≥10x (관측 네트워크 코호트, 서로 다른 파라미터 job 32개 = OBS_COHORT).
- 관문: paired 윈도우 중앙값 + 일측 95% bootstrap 하한 ≥10 (seed 20261003, 20000 resamples).
- **비트 일치 필수** (1-ULP 불일치도 실패), 출력 의무 축소 금지, 모델 교체 금지, 약한-스레드 베이스라인 금지.
- 하드웨어: PACE 노드, affinity CPU 4개 (물리 코어 4, SMT 없음), MemAvailable ~222 GiB, cgroup 미스로틀.
- P00 임무: 잔여 서비스 프로파일 + 공동 스케줄 선택 (측정만, 런타임 패치 없음).

## 2. 어떤 최적화 제안도 지켜야 할 제약 (위반 시 기각)

1. 공개 API 경계(`RunBatch`/`RunAccessibility`) 밖의 가속은 인정 안 됨 — 내부 상수/파라미터로도 공개 서피스를 통과해야 함.
2. 출력 64파일/147,830,672 바이트 그대로 (X_k는 K번 존재 — 생략·병합·반올림 금지).
3. 비트 일치 (재연관, FP32 치환, 연산 순서 변경 금지; `logical_reduction_plan="canonical"` 유지).
4. "같은 답 캐시" 금지 — 입력이 바뀌면 재계산하는 실제 재사용만 인정.
5. 레거시 3종(unа_legacy, madina_legacy, corrected_v1) 분리 유지, 스크라이프 순서/스냅샷 이완 보존.
6. 측정은 공개 호출만 타이머 브래킷; 프로파일링은 타이밍과 분리.

## 3. 측정 요약 (전부 비트 일치 검증됨, 공개 RunBatch 경계 타이밍)

### 3.1 벽시계 (초) — OBS_COHORT 32 rows

| 태그 | 트리 | 구성 | wall_s |
|---|---|---|---|
| a_base_serial | 395cdc5 | serial | 173.31 |
| a_base_par_w4 | 395cdc5 | W4 | **92.11** (serial 대비 1.881x) |
| a_base_par_w2 | 395cdc5 | W2 | 103.89 (1.668x) |
| e_cand_serial | e7c9eda | serial | 193.09 |
| **e_cand_w4_c1** | e7c9eda | W4 + 코호트 | **100.96** |
| e_cand_w2_c4 | e7c9eda | W2 | 123.65 |
| e_cand_w4_c2_qd4 | e7c9eda | W4, queue_depth 4 | 104.95 |
| e_cand_w4_c3_wc2 | e7c9eda | W4, writer_concurrency 2 | 110.17 (C1 재현) |
| e_cand_w2_t2 | e7c9eda | W2, threads 2 | 125.67 |

- **응용 비율 armA-W4 ÷ cand-W4 = 0.912** (CPU 계측 쌍 0.922) → 코호트 라우트 단독으로는 아직 응용 승리 아님.
- 같은 구성 반복 spread **27.2%** (공동사용자 srun 부하) — 단일 윈도우 설정 서열은 무의미, paired 비율은 일관.

### 3.2 serial 서비스 분해 (cProfile self time, cand serial 기준)

| 지역 | self_s | 점유율 |
|---|---|---|
| 행별 GIS 로드/파싱 (pyogrio read_arrow + next + from_wkb) | 141.0 | **75.4%** |
| 라우팅+메트릭 (`integrated_scope_access`) | 15.6 | 10.9% |
| 익스포트 write | 8.2 | 4.4% |
| 토폴로지+기타 | — | 9.3% |

- 코호트 trace 공유가 없애는 슬라이스는 ~11%뿐. **지배 지역은 행별 입력 로딩 (행마다 같은 3개 입력을 다시 읽음).**

### 3.3 병렬 풀 효율 (50ms CPU 샘플러, 4 물리 코어 기준)

- 베이스라인 W4: 효율 **0.524** / 후보 W4: **0.459** — 양쪽 트리 모두 절반 idle.
- W4에서 fold/row가 0.35s(W2) → 0.58-0.83s(W4)로 상승 (동시성 하 경합).
- 원인 3 가설 (미해결): ① 행별 GDAL/arrow 로드의 공유 상태 직렬화 (serial에서도 lock self 7.3-7.5s) ② 코디네이터 commit/publish 구간의 풀 idle ③ fold 커널 대역폭 경합.
- 계측 도구 자체가 다음 지역 선결 과제: worker RowResult.timings/단계별 CPU를 배치 리포트로 노출 (런타임 오너 변경).

### 3.4 코호트 카운터 (실측)

- trace build 1회 2.5s (C11 identity 1개, refusal 0), produce-once 정상 (singleflight_lost 31/31).
- 웜 fold: 32 rows × 14,751 folds/row; 0.349 s/row (W2) ~ 0.581 s/row (W4).
- trace 스토어 실측 1.5 MiB vs 레지던시 캡 8.9-17.9 GiB.

### 3.5 메모리 (whole-tree peak)

serial 0.68 GiB / W2 1.46 GiB / W4 2.53 GiB (양 트리 대칭). 단일 job child maxrss 693 MiB.

## 4. 가산 도달 모델 (비율 곱셈 금지 — 지역은 벽위에서 합산)

arm A W4 ≈ 92s 에서 G=10이면 코호트 전체(~32 rows + 64 exports)가 **~9s** 안에 끝나야 함.
측정된 지역 상한 (serial 기준):

| 지역 | serial 점유 | 후보 수단 |
|---|---|---|
| 공유 입력 준비 P_shared | **145.1s (75%)** | 코호트당 1회 준비 + 행별 view 슬라이스 (모델상 P_shared 항) |
| 풀 효율 | 0.46-0.52 | idle/직렬화 복원 → 모든 병렬 단계 ~2x |
| 메트릭 fold | 15.6s self | per-metric 재사용 — reach/knn은 beta 불변 → 상한 fold 절반 (C31-NOTE-1) |
| 프로세스 오버헤드 | import 3.4-7.8s/process | 다중 코호트 윈도우로 상각 (~7s) |
| 익스포트 | 8.2s self (4.4%) | X_k 의무 유지 — 파이프라인 겹치기만 가능 (축소 금지) |

- Amdahl: s ≤ C=4인 프로세스 병렬만으로는 G=10 불가 (측정 효율 ~0.5이면 더욱).
- 네 지역(준비 공유·풀 복원·메트릭 재사용·윈도우 상각)이 **각각 측정되었고 서로 소속(disjoint)이며 G=10에 공동 필요**.

## 5. 분석기에게 묻는 질문 (우선순위 순)

1. **공유 입력 준비 설계 공간**: 행별 재로딩(75%)을 코호트당 1회로 바꾸는 설계들 — 메모리 상한(레지던시 법칙), 비트 일치 보장(동일 바이트 → 동일 해석), 공개 API 통과 경로. 어떤 분할/포맷/캐시 구조가 안전한가?
2. **풀 idle 3 가설의 분별 실험 설계**: worker 단계별 타이밍 노출 최소 변경이 무엇인지, 노출 후 어떤 측정으로 3 가설을 분별하는지.
3. **메트릭 재사용**: reach/knn beta-불변 사실의 안전한 활용 조건 (어떤 파라미터 변화까지 불변인지, 키는 무엇으로) + C31-NOTE-1 (result key가 weight dtype을 float64로 뭉갬 — 보고용) 처리.
4. **익스포트 겹치기**: 4.4% 지역의 파이프라인화 가능성과 commit/publish fsync 법칙과의 충돌 여부.
5. **윈도우 상각**: 다중 코호트 윈도우 구성이 BENCHMARKS.md 창 구성 법칙과 양립하는지, 오버헤드 상각의 상한.

## 6. 증거 파일 인덱스 (이 디렉터리)

| 파일 | 내용 |
|---|---|
| `stage_model.json` | 전체 측정 모델 (분해·효율·분산·메모리·Amdahl·지역 경계) |
| `configuration_screen.json` | 설정 축 스크린 결과 + 최종 구성 C1 |
| `candidate_register.json` | L3 후보 등록 (E-cohort-parallel) + ablation 지도 |
| `p00_receipt.json` | 완료 판정 + 예산 (15회 실행 1878.6s 과금) |
| `p00_env.json` | 호스트/트리/입력/참조 출력 핀 |
| `runs/*.json`, `runs/*.log` | 실행별 원시 증거 (wall, 카운터, 비트검증 결과) |
| `profiles/*.pstats` | cProfile 원본 (진단용 — 타이밍 증거 아님) |
| `out/` | 실행 출력 원본 1.7 GiB — **커밋 제외**, 해시로 고정됨 |

트리: 베이스라인 = 395cdc5 체크아웃, 후보 = perf/una-10x @ e7c9eda (P00은 코드 변경 없음 — 증거 전용).
