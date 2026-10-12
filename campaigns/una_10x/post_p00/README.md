# P00 이후 실행 지침: 공유 준비와 저비용 재생

이 문서는 새 캠페인이 아니라 `una_10x`의 P00 이후 설계 개정이다. 현재 작업 계보는
`perf/una-10x@6130769a5751bf7446734dc39b9f8e3763461cfe`다. 과거 main에서 재시작하거나
C10~C31 구현을 덮어쓰지 않는다. 실행 순서는 상위 `TASKS_CLAUDE.yaml`에 통합되어 있다.

## 결정
`행마다 GIS 읽기/준비 -> 공유 trace`를 다음으로 바꾼다.

```
공개 RunBatch
  -> 원래 행 검증과 의존성 판정
  -> 입력 identity별 준비 single-flight (decode 이전)
  -> PreparedBundle + 행별 독립 객체
  -> 기존 trace 생성/검증
  -> compiled replay, 검증된 경우만 metric별 재사용
  -> 기존 exporter로 모든 파일 생성
  -> 원래 순서의 검증/내구성/게시/상태/체크포인트
```

먼저 `AUTHORITY.md`, `SOURCE_AND_ERRATA.md`, `IMPLEMENTATION_MAP.md`, 설계 01~07,
`TEST_MATRIX.md`를 읽는다. `rollout.json`은 구현·검토·측정 관문을 요약한다.
자료는 `sources.json`에 고정되어 있다. 새 설계와 미검증 가설을 P00 실측으로 부르지 않는다.

## 재개 명령
저장소 루트에서:

```bash
python3 campaigns/una_10x/tools/validate_packet.py
python3 campaigns/una_10x/post_p00/tools/validate_post_p00.py
python3 -m unittest discover -s campaigns/una_10x/tests -v
python3 -m unittest discover -s campaigns/una_10x/post_p00/tests -v
```

새 run root에 `resume.json`을 작성하고 P00 원본과 로컬 원시 출력/PRIMARY 해시를 확인한다.
P00 숫자는 원격 기록의 관측이지 이번 개정이 재실행한 성능 결과가 아니다. 기존 `PRIMARY.json`,
`BENCHMARKS.md`, `CONTRACT.md`, raw evidence와 수치 oracle는 변경하지 않는다.
설정/비교 정책 보완은 별도 protocol amendment로 버전 관리하고 원래 고정 비교도 유지한다.

## 이번 개정의 범위
설계·작업 DAG·goal과 실행 가능한 설계 검증 도구만 제공한다. 도구는 실제 PreparedBundle,
Numba 최적화 또는 배치 runtime의 구현체가 아니다. production 모듈이 이 디렉터리를 import하면
안 된다. 실제 구현은 각 담당자가 독립 worktree에서 회귀 테스트와 검토를 거쳐 통합한다.
Native/GPU/Madina/science 지원 의무는 유지하되 P00의 10x 비용 모델에 미측정 배수를 넣지 않는다.

현재 후보 100.9639초, 기준 92.1141초라는 기록에서 계산한 임시 10x 예산은 9.21141초다.
이는 최신 최강 기준이나 신뢰구간을 대신하지 않는다. 최종 설치된 공개 호출의 paired 중앙값과
일측 95% 하한이 모두 10 이상이어야 한다. 목표 미달이면 `target_unmet`을 기록한다.
