# LeRobot #3177 / from_timestamp drift

> **Migrated from Quartz wiki** (`wiki/sources/lerobot-issue-3177.md`, updated 2026-09-26)  
> tags: opensource, lerobot, huggingface, bug-report  
> source: https://github.com/huggingface/lerobot/issues/3177  
> source: https://github.com/huggingface/lerobot/pull/3179  
> source: raw/2026-09-26-lerobot-3177/  

## English

GitHub issue filed by [Joonmo Han (weedmo) / 한준모](/area/career/joonmo-han) (`weedmo`) against [Hugging Face LeRobot / 허깅페이스 LeRobot](/resource/robotics/huggingface-lerobot):

- **Title:** Cumulative floating-point drift in `from_timestamp` causes `FrameTimestampError` on concatenated videos
- **URL:** https://github.com/huggingface/lerobot/issues/3177
- **Opened:** 2026-03-17
- **Closed:** 2026-07-31 (`state_reason: not_planned`, closed by stale bot after 30 days idle)
- **Labels:** bug, dataset, performance, examples, training, stale
- **Reactions:** 2 × 👍

**Claimed problem:** When many episodes are concatenated into one `.mp4`, `decode_video_frames_torchvision` can raise `FrameTimestampError` on later episodes because three timestamp paths diverge (parquet `frame_index/fps`, video PTS, and accumulated `from_timestamp` from `get_video_duration_in_s()`).

**Maintainer response (pkooij, 2026-03-17):** Praised the investigation; invited a PR and a regression test.

**Follow-up by weedmo (2026-03-18):** Opened fix PR [LeRobot PR](/project/open-source-contributions/lerobot-pr-3179) (https://github.com/huggingface/lerobot/pull/3179).

**Collaborator reply (CarolinePascal, 2026-06-23):** Could not reproduce; PR tests passed on `main` even with many episodes; argued that float drift alone is unlikely to reach a full frame (0.0333s) under normal LeRobot recording scale. Asked for a reproducible case if it reappears.

**Local snapshot:** `raw/2026-09-26-lerobot-3177/` (issue JSON, comments JSON, PR JSON).

See also [Open source contributions / 오픈소스 기여](/project/open-source-contributions/open-source-contributions) and [LeRobot PR](/project/open-source-contributions/lerobot-pr-3179).

## 한국어

[Joonmo Han (weedmo) / 한준모](/area/career/joonmo-han) (`weedmo`)가 [Hugging Face LeRobot / 허깅페이스 LeRobot](/resource/robotics/huggingface-lerobot)에 올린 이슈입니다.

- **제목:** 이어 붙인 비디오에서 `from_timestamp` 누적 부동소수점 드리프트로 `FrameTimestampError` 발생
- **URL:** https://github.com/huggingface/lerobot/issues/3177
- **생성:** 2026-03-17
- **종료:** 2026-07-31 (30일 무활동 stale 봇, `not_planned`)
- **라벨:** bug, dataset, performance, examples, training, stale
- **반응:** 👍 2

**주장한 문제:** 에피소드를 하나의 `.mp4`로 concat할 때, 나중 에피소드에서 타임스탬프 경로 3개(parquet, video PTS, 누적 `from_timestamp`)가 어긋나 `FrameTimestampError`가 날 수 있다는 내용.

**멘테이너(pkooij, 2026-03-17):** 조사 칭찬, PR·회귀 테스트 요청.

**weedmo 후속(2026-03-18):** 수정 PR [LeRobot PR](/project/open-source-contributions/lerobot-pr-3179) 제출 (https://github.com/huggingface/lerobot/pull/3179).

**Collaborator(CarolinePascal, 2026-06-23):** 재현 실패. PR 테스트는 `main`에서도 통과. 일반 녹화 규모에서는 float 드리프트만으로 1프레임(0.0333s)까지 가기 어렵다고 봄. 재현되면 상세 공유 요청.

**로컬 스냅샷:** `raw/2026-09-26-lerobot-3177/`

관련: [Open source contributions / 오픈소스 기여](/project/open-source-contributions/open-source-contributions), [LeRobot PR](/project/open-source-contributions/lerobot-pr-3179).
