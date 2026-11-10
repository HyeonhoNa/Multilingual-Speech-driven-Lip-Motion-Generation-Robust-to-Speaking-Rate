# 서버 실행: 저장된 점수 조회와 새 생성·평가

이 문서는 모델·데이터·환경이 설치된 **완성된 로컬 배포본**에서 실행하는 안내입니다. GitHub 소스만 복제한 상태에서는 실행할 수 없습니다. AW 원본 코드와 가중치는 기술이전된 비공개 자산으로, 공개 저장소에 포함하지 않습니다. 필요한 외부 자산은 [ASSETS.md](ASSETS.md)를 참고하세요.

```bash
cd /path/to/tta-lip-motion
python3 run.py verify
python3 run.py list
```

`list`는 사용 가능한 음성 식별자·언어·속도 구간과 ID를 출력합니다. 현재 고정 구성은 **15언어·90음성 × 10 ID = 900조합**, 모델은 **9,000회 체크포인트**입니다. 실행 시 추가 학습이나 개별 fitting은 없습니다. `CUDA_VISIBLE_DEVICES`를 따로 설정하지 말고, 실제 사용 가능한 GPU 번호를 `--gpus`에 지정하세요.

## 1. 이미 생성한 결과의 점수 보기

확정해 보관한 900조합의 기존 측정값을 한 줄씩 읽고 마지막에 평균을 표시합니다. GPU는 사용하지 않으며 **새 측정이 아닌 저장된 점수 조회**입니다.

```bash
python3 run.py reference
```

다른 실행 폴더에 저장된 평가값을 보려면:

```bash
python3 run.py report --output outputs/tta900
```

저장된 영상에서 **공식 점수를 다시 계산**하려면, 원본과 다른 새 평가 폴더를 지정합니다. 영상은 재생성하거나 복사하지 않습니다. 각 점수가 계산되는 즉시 출력되고, 마지막에 평균이 표시됩니다.

```bash
python3 run.py rescore --source outputs/tta900 \
  --output outputs/tta900_rescore --gpus 0
```

`--source`에는 `run.json`, `generation.json`과 해당 영상들이 있는 완전한 생성 실행 폴더를 넣습니다. 원본 실행에 기록된 범위를 재평가하므로 원본이 일부 조합이면 그 일부만 측정합니다. 원본과 새 평가 폴더는 서로 포함되지 않는 별도 폴더여야 합니다. 코드 업데이트 이전의 결과도 모델·데이터·ID 및 영상 무결성을 검증한 뒤 재평가할 수 있습니다. 단순히 `reference_results/`의 CSV만 있는 경우에는 영상을 재계산할 수 없으므로 `reference`로 조회합니다. 이 차이는 출력의 `cached`/`scored` 표기로 구분합니다.

## 2. 특정 음성·ID를 새로 생성하고 점수 보기

```bash
python3 run.py run --gpus 0 \
  --clips fleurs_test_th_th_2007_r00358 \
  --identities lfw_Kurt_Schottenheimer \
  --output outputs/selected_new --no-average
```

`--clips`와 `--identities`에 쉼표로 여러 식별자를 넣을 수 있습니다. 지정한 음성 × ID 조합을 모두 생성하고, 평가 단계에서 **각 점수를 계산하는 즉시 출력하며 터미널 평균은 생략**합니다. 평균 등 집계 파일은 재현을 위해 저장합니다. 출력 폴더 이름을 바꾸면 새로 실행하고, 같은 명령·같은 출력 폴더로 실행하면 완료된 항목을 검증해 재사용합니다.

## 3. 전체를 새로 생성하고 점수 보기

```bash
python3 run.py run --gpus 0 --output outputs/tta900_new
```

전체 900조합을 생성한 뒤 공식 평가를 수행합니다. 평가 단계에서 각 결과가 완료될 때마다 D·C를 출력하고, **900조합이 모두 완료되면 전체 평균**을 출력합니다. GPU 두 개가 실제로 사용 가능할 때는 `--gpus 0,1`로 실행할 수 있습니다. 음성을 GPU별로 나누며, 여러 GPU에서 나오는 평가 점수도 즉시 합쳐서 표시합니다.

```text
[scored] 음성 / 인물  D=...  C=...
...
Complete: 900/900 cells (full fixed cohort).
Final mean: D=...  C=... (900/900 cells).
```

실행은 준비 → 전체 선택 결과 생성 → 공식 평가 순서입니다. 생성 도중에는 아직 D·C가 없으며, 상세 진행은 `logs/`에 기록됩니다. 실패·누락은 완료된 평가로 취급하지 않습니다. 중간 집계는 `Partial mean`으로 표시되며 완성된 전체 평균과 구분합니다.

## 저장 위치·재개

`--output`에 지정한 폴더 아래에 저장됩니다.

| 경로 | 내용 |
|---|---|
| `workers/*/videos/<identity>/<clip>.mkv` | 음성이 포함된 생성 영상 |
| `workers/*/motions/<clip>.npy` | FLAME 모션 |
| `cell.csv` | 조합별 D·C |
| `language.csv`, `identity.csv`, `utterance.csv` | 언어·ID·음성별 집계 |
| `summary.json` | 완료 수·평균·목표 비교 |
| `logs/`, `evaluation/` | 작업 로그와 공식 평가 기록 |

`rescore`의 영상은 원본 실행 폴더에 그대로 있고, 새 평가 폴더에는 수치와 평가 기록만 저장됩니다. 영상 플레이어가 MKV를 지원하지 않으면 별도의 감상용 MP4를 만들 수 있지만, 재압축본으로 공식 평가 입력을 대체하지 않습니다.

중단 후 **같은 명령**을 다시 실행하면 완료된 결과를 검증하고 이어갑니다. 이미 사용한 출력 폴더의 코드·설정·범위·GPU 구성을 바꾸면 재개를 거부하므로 새로운 폴더를 사용하세요. `rescore`는 원본 영상의 평가만 새로 수행하기 위한 별도 경로입니다. 재평가를 재개할 때에도 원본 `run.json`·`generation.json`과 영상이 변경되지 않아야 합니다.

백그라운드가 필요할 때 `run`에 `--background`를 붙입니다. 점수 출력은 실행 로그에 저장됩니다.

```bash
python3 run.py run --gpus 0 --output outputs/tta900_new --background
tail -f outputs/tta900_new/launcher.log
```

전체 영상은 수십 GiB가 필요할 수 있으므로 저장 여유 공간을 먼저 확인하세요. `git clean -fdx`는 사용하지 마세요. Git에서 제외한 모델·데이터·환경까지 삭제합니다.

## 수식과 출처

실행하는 공식 코드, LSE-D·C 수식, crop·시간 오프셋·전처리 조건과 논문은 [LSE_METRICS.md](LSE_METRICS.md)에 정리했습니다. 데이터셋 평균은 조합별 D·C의 동일 가중치 산술평균입니다. 목표는 **평균 D ≤ 6.5, 평균 C ≥ 8.0**이며 모든 조합의 개별 통과와는 구분합니다.
