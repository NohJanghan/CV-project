# FungiTastic Toxicity Experiment Report

작성일: 2026-05-31

## 1. 변경 이유

기존 mushroom 실험은 두 가지 한계가 컸다.

1. `combined-kaggle-mushrooms-dataset`에는 독성 라벨이 없어 Wikipedia 기반 weak label에 의존했다.
2. UCI Mushroom Classification은 `concept -> edible/poisonous` tabular dataset이지만, species와 image가 없고 Dataset1의 species set과 정렬되지 않았다.

FungiTastic은 이 문제를 완화한다. metadata row에 image filename, species/taxonomy, habitat/substrate metadata, `poisonous` label이 함께 있으므로 같은 데이터셋 안에서 `image -> label` 실험을 할 수 있다. 따라서 이번 실험에서는 Wikipedia 라벨과 UCI를 사용하지 않았다.

## 2. 다운로드 정책

FungiTastic full archive는 약 50GB라 전체를 받지 않았다. 필요한 최소 조합만 사용했다.

새 스크립트:

- [download_fungitastic.py](/home/acsl/projects/CV-project/download_fungitastic.py)

다운로드/추출한 항목:

- `metadata.zip` 약 327MB
- `FungiTastic-Mini-train-300p.zip` 약 975MB
- `FungiTastic-Mini-val-300p.zip` 약 228MB
- `FungiTastic-Mini-test-300p.zip` 약 280MB

사용하지 않은 항목:

- full-size images
- full FungiTastic split images
- UCI Mushroom Classification
- Wikipedia/Wikidata 라벨

## 3. 데이터 구성

Mini metadata 전체:

| Split | Non-poisonous rows | Poisonous rows |
| --- | ---: | ---: |
| train | 40,999 | 5,843 |
| val | 7,670 | 1,742 |
| test | 8,915 | 1,823 |

Poisonous species는 13개로 species-level imbalance가 크다. 빠른 실험을 위해 공식 train/val/test split은 유지하되, 각 split에서 species cap과 class-balanced subsampling을 적용했다.

실험 subset:

| Split | Non-poisonous | Poisonous | Total |
| --- | ---: | ---: | ---: |
| train | 1,476 | 738 | 2,214 |
| val | 1,304 | 652 | 1,956 |
| test | 1,216 | 608 | 1,824 |

총 5,994장, 207 species를 사용했다.

## 4. 실험 모델

실험 스크립트:

- [experiments/fungitastic_toxicity_experiment.py](/home/acsl/projects/CV-project/experiments/fungitastic_toxicity_experiment.py)
- [configs/fungitastic_toxicity_experiment.yaml](/home/acsl/projects/CV-project/configs/fungitastic_toxicity_experiment.yaml)

모델:

| ID | Model | Role |
| --- | --- | --- |
| F1 | DINOv2 global linear | black-box image baseline |
| F2 | DINOv2 PCA-64 linear | dimension reduction control |
| F3 | DINOv2 patch spherical K-means K=40 | learned visual concept bottleneck |
| F4 | E7-style medoid prototype K=40 | human-inspectable prototype concept bottleneck |

## 5. 결과

| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe |
| --- | --- | ---: | ---: | ---: | ---: |
| F1 | DINOv2 global linear | 0.7674 | 0.9052 | 0.8322 | 0.8355 |
| F2 | DINOv2 PCA-64 linear | 0.7293 | 0.8868 | 0.7977 | 0.7401 |
| F3 | spherical concept bottleneck | 0.5427 | 0.7296 | 0.6542 | 0.5592 |
| F4 | medoid prototype bottleneck | 0.5677 | 0.7454 | 0.6743 | 0.5789 |

이전 combined+Wikipedia strict label 실험과 비교하면, aligned FungiTastic label 사용으로 F1 baseline이 크게 좋아졌다.

| Setup | Best image AUROC | Best image F1 unsafe |
| --- | ---: | ---: |
| combined images + Wikipedia weak labels | 0.7452 | 0.5512 |
| FungiTastic Mini aligned labels | 0.9052 | 0.7674 |

## 6. Concept 시각화

F3 spherical learned concepts:

![F3 FungiTastic concept examples](../output/fungitastic_toxicity/explainability_visuals/F3_fungitastic_concept_examples.png)

F4 medoid prototype concepts:

![F4 FungiTastic medoid examples](../output/fungitastic_toxicity/explainability_visuals/F4_fungitastic_concept_examples.png)

F4는 centroid가 아니라 실제 train patch medoid를 prototype으로 쓰기 때문에 사람이 concept 원형을 확인하기 더 쉽다. 성능도 F3보다 약간 좋았다.

## 7. 해석

FungiTastic으로 바꾸면서 label provenance와 dataset alignment 문제는 상당히 개선됐다. 더 이상 Wikipedia weak label이나 UCI의 species mismatch에 의존하지 않는다.

다만 concept bottleneck은 여전히 black-box DINO global feature보다 낮다. 독성은 species-level property이고, 단일 이미지 patch에서 항상 직접 관찰 가능한 morphological cue로 분해되지 않는다. 따라서 concept 모델은 “설명 가능성 후보”를 제공하지만, 성능을 유지하려면 더 풍부한 concept supervision이나 view/part-aware 설계가 필요하다.

## 8. 실행 명령

```bash
uv run python download_fungitastic.py --keep-zip
uv run python experiments/fungitastic_toxicity_experiment.py \
  --config_path configs/fungitastic_toxicity_experiment.yaml
uv run python experiments/export_fungitastic_concepts.py --model_id F3
uv run python experiments/export_fungitastic_concepts.py --model_id F4
```
