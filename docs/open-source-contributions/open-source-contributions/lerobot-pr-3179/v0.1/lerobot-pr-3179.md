# LeRobot PR #3179 / video timestamp fix

> **Migrated from Quartz wiki** (`wiki/sources/lerobot-pr-3179.md`, updated 2026-09-26)  
> tags: opensource, lerobot, huggingface, pull-request  
> source: https://github.com/huggingface/lerobot/pull/3179  
> source: https://github.com/huggingface/lerobot/issues/3177  
> source: raw/2026-09-26-lerobot-3177/pr-3179.json  

## English

Pull request by [Joonmo Han (weedmo) / 한준모](/area/career/joonmo-han) (`weedmo`) fixing [LeRobot](/project/open-source-contributions/lerobot-issue-3177):

- **Title:** `fix(datasets): use actual video duration for timestamps instead of float accumulation`
- **URL:** https://github.com/huggingface/lerobot/pull/3179
- **Author:** weedmo
- **Status (as of 2026-09-26):** **open, not merged**
- **Linked issue:** https://github.com/huggingface/lerobot/issues/3177

**Change summary (from author comment on the issue):**

- `lerobot_dataset.py`: read `latest_duration_in_s` from the actual video via `get_video_duration_in_s()` instead of parquet metadata; recompute episode duration from total duration after concatenation
- `aggregate.py`: same pattern for dataset aggregation
- Regression test: 50 episodes with varying frame counts (7–17), assert `from_timestamp ==` actual video PTS at boundaries

**Outcome so far:** Issue closed as stale without merge. Collaborator reported suggested tests already pass on `main`. Contribution record: authored bug report + proposed fix PR (pending review/merge).

## 한국어

[LeRobot](/project/open-source-contributions/lerobot-issue-3177)를 고치기 위해 [Joonmo Han (weedmo) / 한준모](/area/career/joonmo-han) (`weedmo`)가 연 PR입니다.

- **제목:** `fix(datasets): use actual video duration for timestamps instead of float accumulation`
- **URL:** https://github.com/huggingface/lerobot/pull/3179
- **작성자:** weedmo
- **상태 (2026-09-26 기준):** **open, 미머지**
- **연결 이슈:** https://github.com/huggingface/lerobot/issues/3177

**변경 요약 (이슈 댓글 기준):**

- `lerobot_dataset.py`: parquet 메타 대신 실제 비디오에서 `get_video_duration_in_s()`로 duration 읽고, concat 후 총 length로 에피소드 duration 재계산
- `aggregate.py`: 동일 패턴
- 회귀 테스트: 프레임 수 7–17인 에피소드 50개, 경계에서 `from_timestamp`가 실제 PTS와 일치하는지 검증

**현재까지 결과:** 이슈는 stale로 닫힘, 미머지. Collaborator는 제안 테스트가 `main`에서도 통과한다고 함. 기여 기록: 버그 리포트 작성 + 수정 PR 제안 (리뷰/머지 대기).
