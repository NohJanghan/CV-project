# FungiTastic Full Dataset Toxicity Experiment

작성일: 2026-06-04

## 1. 목적

FungiTastic full 300px split을 사용해 mushroom toxicity classification을 다시 실험했다. 이번 버전은 validation/test protocol을 명시적으로 분리한다.

- 기본 protocol은 `openset`이다.
- `closedset`은 별도 config로 실행한다.
- OpenSet과 ClosedSet metadata를 합치지 않는다.
- 라벨은 FungiTastic metadata의 `poisonous` column만 사용한다.
- Wikipedia, Wikidata, UCI Mushroom Classification 라벨은 사용하지 않는다.

full-size 원본 이미지는 약 50GB라 사용하지 않았고, full split의 300px 이미지를 사용했다. 프로젝트 컨벤션에 맞춰 `min_images_per_species=5`와 existing-image 검사를 유지했다.

## 2. 구현 변경

관련 파일:

- [experiments/fungitastic_toxicity_experiment.py](/home/acsl/projects/CV-project/experiments/fungitastic_toxicity_experiment.py)
- [configs/fungitastic_full_all_toxicity_experiment.yaml](/home/acsl/projects/CV-project/configs/fungitastic_full_all_toxicity_experiment.yaml)
- [configs/fungitastic_full_all_closedset_toxicity_experiment.yaml](/home/acsl/projects/CV-project/configs/fungitastic_full_all_closedset_toxicity_experiment.yaml)
- [experiments/export_fungitastic_concepts.py](/home/acsl/projects/CV-project/experiments/export_fungitastic_concepts.py)

새 옵션:

```bash
--evaluation_protocol {openset,closedset}
```

`dataset_variant: full`일 때 이 옵션에 따라 `FungiTastic-OpenSet-Val/Test.csv` 또는 `FungiTastic-ClosedSet-Val/Test.csv` 중 하나만 읽는다. `dataset_variant: mini`에서는 기존 Mini train/val/test split을 그대로 사용한다.

full run에서 patch token 전체를 저장하면 메모리와 디스크 사용량이 과도하게 커진다. 따라서 `store_patch_tokens: false`일 때는 DINO global feature만 cache하고, concept discovery와 activation은 필요한 patch token을 streaming으로 계산한다.

## 3. 데이터 구성

### OpenSet Full-All

Config: [configs/fungitastic_full_all_toxicity_experiment.yaml](/home/acsl/projects/CV-project/configs/fungitastic_full_all_toxicity_experiment.yaml)

| Split | Non-poisonous | Poisonous | Total |
| --- | ---: | ---: | ---: |
| train | 415,667 | 18,035 | 433,702 |
| val | 91,677 | 4,245 | 95,922 |
| test | 92,239 | 4,492 | 96,731 |
| total | 599,583 | 26,772 | 626,355 |

Manifest 검증:

- rows: 626,355
- unique image paths: 626,355
- duplicate image paths: 0

### ClosedSet Full-All

Config: [configs/fungitastic_full_all_closedset_toxicity_experiment.yaml](/home/acsl/projects/CV-project/configs/fungitastic_full_all_closedset_toxicity_experiment.yaml)

| Split | Non-poisonous | Poisonous | Total |
| --- | ---: | ---: | ---: |
| train | 415,667 | 18,035 | 433,702 |
| val | 85,439 | 4,220 | 89,659 |
| test | 87,368 | 4,464 | 91,832 |
| total | 588,474 | 26,719 | 615,193 |

Manifest 검증:

- rows: 615,193
- unique image paths: 615,193
- duplicate image paths: 0

## 4. 모델

| ID | Model | 설명 가능성 |
| --- | --- | --- |
| F1 | DINOv2 global linear | 이미지 전체 embedding을 쓰는 성능 baseline |
| F2 | DINOv2 PCA-64 linear | 차원 축소 control, concept 설명성은 낮음 |
| F3 | DINOv2 patch spherical concept bottleneck K=40 | patch centroid concept를 top activation 이미지로 사람이 naming |
| F4 | E7-style medoid prototype concept bottleneck K=40 | 실제 train patch prototype을 concept로 사용해 F3보다 inspection이 쉬움 |

## 5. 결과

### OpenSet Full-All

산출물:

- [REPORT.md](/home/acsl/projects/CV-project/output/fungitastic_full_all_openset_toxicity/REPORT.md)
- [results.json](/home/acsl/projects/CV-project/output/fungitastic_full_all_openset_toxicity/results.json)

| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | DINOv2 global linear | 0.3521 | 0.8694 | 0.6632 | 0.3598 |
| F2 | DINOv2 PCA-64 linear | 0.2943 | 0.8297 | 0.6390 | 0.3195 |
| F3 | spherical concept bottleneck | 0.1275 | 0.7854 | 0.5368 | 0.0870 |
| F4 | medoid prototype bottleneck | 0.1606 | 0.7854 | 0.5526 | 0.1278 |

### ClosedSet Full-All

산출물:

- [REPORT.md](/home/acsl/projects/CV-project/output/fungitastic_full_all_closedset_toxicity/REPORT.md)
- [results.json](/home/acsl/projects/CV-project/output/fungitastic_full_all_closedset_toxicity/results.json)

| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | DINOv2 global linear | 0.3572 | 0.8651 | 0.6664 | 0.3683 |
| F2 | DINOv2 PCA-64 linear | 0.2821 | 0.8267 | 0.6173 | 0.2664 |
| F3 | spherical concept bottleneck | 0.1897 | 0.7752 | 0.5763 | 0.1980 |
| F4 | medoid prototype bottleneck | 0.2095 | 0.7734 | 0.6189 | 0.3313 |

## 6. 해석

OpenSet과 ClosedSet 모두에서 AUROC 기준 F1이 가장 강하다. DINO global representation은 species-level toxicity와 관련된 시각적/분류적 신호를 가장 많이 보존한다.

F2는 DINO feature를 64차원으로 줄여도 상당한 AUROC를 유지한다. 하지만 PCA axis는 사람이 직접 이름 붙일 수 있는 concept가 아니므로, 설명 가능성 측면에서는 F3/F4와 다르다.

F3/F4는 concept bottleneck을 통해 사람이 top activation 예시를 보고 concept 후보를 해석할 수 있다. 대신 정보 병목이 강해서 F1보다 성능이 낮다. 특히 full-all은 poisonous prevalence가 낮아 고정 threshold 0.5에서 unsafe recall/F1이 낮아진다.

F4는 실제 train patch prototype을 쓰므로 F3 centroid보다 사람이 원형을 확인하기 쉽다. ClosedSet에서는 F4가 F3보다 unsafe F1과 recall이 높았지만, AUROC는 비슷하거나 약간 낮았다.

## 7. Concept 시각화

OpenSet F3 learned visual concepts:

![OpenSet F3 concept examples](../output/fungitastic_full_all_openset_toxicity/explainability_visuals/F3_fungitastic_concept_examples.png)

OpenSet F4 medoid prototype concepts:

![OpenSet F4 medoid concept examples](../output/fungitastic_full_all_openset_toxicity/explainability_visuals/F4_fungitastic_concept_examples.png)

ClosedSet F3 learned visual concepts:

![ClosedSet F3 concept examples](../output/fungitastic_full_all_closedset_toxicity/explainability_visuals/F3_fungitastic_concept_examples.png)

ClosedSet F4 medoid prototype concepts:

![ClosedSet F4 medoid concept examples](../output/fungitastic_full_all_closedset_toxicity/explainability_visuals/F4_fungitastic_concept_examples.png)

## 8. 결론과 다음 개선

FungiTastic full dataset에서도 aligned poisonous label을 쓰면 DINO global baseline의 AUROC는 안정적으로 높다. 하지만 full-all 설정에서는 class imbalance가 커서 고정 threshold 0.5 기준 unsafe F1/recall이 낮다.

다음 개선은 validation set에서 threshold를 calibration하는 것이다. 현재 표는 고정 threshold 0.5 기준이며, poisonous prevalence가 낮은 데이터에서는 threshold tuning 또는 비용 민감 decision rule을 추가해야 unsafe F1/recall을 공정하게 비교할 수 있다.

설명 가능성 측면에서는 F4가 F3보다 사람이 보기 쉬운 prototype을 제공하지만, 독성 여부가 이미지 한 장의 관찰 가능한 형태만으로 결정되지 않는 경우가 많다. 향후에는 `unknown/insufficient visual evidence` class, multi-view 정보, species-level prior를 명시적으로 결합하는 설계가 필요하다.

## 9. 실행 명령

```bash
uv run python download_fungitastic.py --subset full --size 300

tmux new-session -d -s fungitastic_full_all_openset \
  'UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/fungitastic_toxicity_experiment.py \
  --config_path configs/fungitastic_full_all_toxicity_experiment.yaml'

tmux new-session -d -s fungitastic_full_all_closedset \
  'UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/fungitastic_toxicity_experiment.py \
  --config_path configs/fungitastic_full_all_closedset_toxicity_experiment.yaml'

UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/export_fungitastic_concepts.py \
  --experiment_dir output/fungitastic_full_all_openset_toxicity --model_id F3

UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/export_fungitastic_concepts.py \
  --experiment_dir output/fungitastic_full_all_openset_toxicity --model_id F4

UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/export_fungitastic_concepts.py \
  --experiment_dir output/fungitastic_full_all_closedset_toxicity --model_id F3

UV_CACHE_DIR=/tmp/uv-cache uv run python experiments/export_fungitastic_concepts.py \
  --experiment_dir output/fungitastic_full_all_closedset_toxicity --model_id F4
```
