# Mushroom Toxicity Concept Experiment Report

작성일: 2026-05-31

## 1. 목적

MILK10K의 concept discovery 실험을 버섯 독성 분류 문제로 옮겼다. 핵심 차이는 MILK10K에는 `image -> MONET concept -> label` 관계가 있었지만, 이번 버섯 실험에는 두 데이터셋이 분리되어 있다는 점이다.

| Dataset | Available pair | Missing link |
| --- | --- | --- |
| Dataset1 combined Kaggle mushroom images | image -> species | species -> edible/unsafe |
| Dataset2 UCI Mushroom Classification | morphology concepts -> edible/poisonous | species, image |

따라서 UCI를 Dataset1의 라벨 소스로 쓰지 않는다. UCI 설명 자체가 “Agaricus/Lepiota family의 23종에 해당하는 hypothetical samples”이고, unknown/not recommended를 poisonous에 합쳤다고 밝히므로, Dataset1의 169개 species와 직접 정렬되는 ground truth가 아니다. 해결 방식은 다음처럼 분리했다.

1. Dataset1은 외부 species-level weak label을 수집해 image-toxicity 실험에 사용한다.
2. Dataset2 UCI는 morphology concept가 label을 얼마나 잘 예측하는지 보는 별도 tabular benchmark로만 사용한다.
3. 이미지에서 보이지 않는 morphology concept는 forced guess하지 않고 `unknown`으로 처리한다.

## 2. 데이터 수집

새 스크립트:

- [download_mushroom_data.py](/home/acsl/projects/CV-project/download_mushroom_data.py)

생성 파일:

- `data/mushrooms/raw/combined-kaggle-mushrooms-dataset.zip`
- `data/mushrooms/raw/mushroom-classification.zip`
- `data/mushrooms/uci/mushrooms.csv`
- `data/mushrooms/species_from_dataset1.txt`
- `data/mushrooms/species_toxicity_labels.csv`
- `data/mushrooms/species_toxicity_provenance.csv`

라벨은 하드코딩하지 않고 외부에서 재수집한다. 사용한 라벨 소스는 Wikidata가 아니라 다음이다.

- GBIF Species API: taxonomy/name normalization only
- Wikipedia species page: `Mycomorphbox`의 `howEdible`, `howEdible2`
- Wikipedia poisonous/deadly mushroom lists: unsafe evidence 보강

provenance에는 Wikipedia page title, revision id, revision timestamp, page URL, GBIF match confidence가 저장된다.

## 3. 라벨 정책

Strict binary mapping은 보수적으로 잡았다.

| Source value | Strict label |
| --- | --- |
| `choice`, `edible`만 존재 | `edible` |
| poisonous/deadly list에 있음 | `unsafe` |
| `deadly`, `poisonous`, `psychoactive`만 존재 | `unsafe` |
| `caution`, `edible when cooked`, `unknown`, `inedible`, `unpalatable`, mixed edible/unsafe | `exclude` |
| page missing 또는 external evidence 없음 | `exclude` |

최종 no-Wikidata 라벨 coverage:

| Strict label | Species count |
| --- | ---: |
| edible | 35 |
| unsafe | 23 |
| exclude | 111 |

Strict experiment 사용량:

| Split | Edible species | Unsafe species | Images |
| --- | ---: | ---: | ---: |
| train | 24 | 16 | 2,400 |
| val | 4 | 2 | 360 |
| test | 7 | 5 | 720 |

각 species는 최대 60장만 사용했고, split은 species-disjoint로 구성했다.

## 4. 실험 모델

### M1: DINOv2 Global Linear

```text
image -> DINOv2 global feature -> linear classifier -> edible/unsafe
```

이미지 기반 black-box baseline이다. 성능 기준점은 되지만, feature dimension을 사람이 직접 해석하기 어렵다.

### M2: DINOv2 PCA-64 Linear Control

```text
image -> DINOv2 global feature -> PCA 64 -> sparse linear classifier
```

“DINO dimension을 줄이면 concept bottleneck과 같은가?”를 확인하는 control이다. PCA 축은 사람이 이름 붙일 수 있는 concept가 아니므로 explanation 모델은 아니다.

### M3: Spherical Patch Concept Bottleneck

```text
image -> DINOv2 patch tokens -> spherical K-means K=40 -> sparse classifier
```

MILK10K E5와 유사한 learned visual concept 모델이다. concept 이름은 자동 생성되지 않으며, top activating examples를 사람이 보고 cap color, underside visibility, texture, morphology artifact 등으로 명명해야 한다.

![M3 mushroom concept examples](../output/mushroom_toxicity/explainability_visuals/M3_mushroom_concept_examples.png)

### M5: MILK10K E7-Style Medoid Prototype

```text
image -> DINOv2 patch tokens -> spherical K-means -> nearest real patch medoids -> sparse classifier
```

이번에 추가한 MILK10K E7 이식 모델이다. centroid 자체가 아니라 train patch 중 centroid에 가장 가까운 실제 patch를 prototype으로 사용한다. 따라서 M3보다 사람이 concept 원형을 검토하기 쉽다.

![M5 mushroom concept examples](../output/mushroom_toxicity/explainability_visuals/M5_mushroom_concept_examples.png)

### M4: UCI Concept Classifier With Unknown

```text
UCI morphology concepts -> unknown-aware sparse classifier -> edible/poisonous
```

UCI는 Dataset1과 직접 연결하지 않고 별도 tabular benchmark로 둔다. top-view 이미지에서 보이지 않을 수 있는 gill, stalk, ring, spore print, odor feature는 `__unknown__` category로 마스킹했다.

## 5. 결과

| ID | Model | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 | DINOv2 global linear | 0.5512 | 0.7349 | 0.6407 | 0.4933 |
| M2 | DINOv2 PCA-64 linear | 0.5412 | 0.7452 | 0.6443 | 0.4600 |
| M3 | DINOv2 patch concept K=40 | 0.3820 | 0.5611 | 0.4993 | 0.3533 |
| M5 | E7-style medoid prototype K=40 | 0.4651 | 0.5477 | 0.5179 | 0.5000 |
| M4 | UCI concepts, top-view unknown | 0.6749 | 0.7765 | 0.7145 | 0.6075 |

M4 full-concept UCI baseline:

| Setting | F1 unsafe | AUROC | Balanced accuracy | Recall unsafe |
| --- | ---: | ---: | ---: | ---: |
| all UCI concepts visible | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| only cap/ecology visible, others unknown | 0.6749 | 0.7765 | 0.7145 | 0.6075 |

## 6. 해석

M1과 M2가 비슷해졌다. no-Wikidata strict label로 바꾸면서 species 수와 class 구성이 바뀌었고, PCA-64가 AUROC에서는 M1과 비슷하지만 F1/recall에서는 여전히 낮다. PCA는 성능 control일 뿐 설명 가능한 concept bottleneck은 아니다.

M3는 이번 strict split에서 약했다. learned visual concept는 해석 후보를 제공하지만, 독성은 species-level property라 이미지 patch cluster와 완전히 대응하지 않는다.

M5는 M3보다 F1이 높고 recall도 높다. medoid prototype은 실제 이미지 patch라 사람이 concept 원형을 검토하기 쉽다는 장점이 있지만, 성능은 아직 M1보다 낮다.

M4는 UCI 내부에서는 강하지만 Dataset1 image model이 아니다. full concepts가 완벽한 것은 사람이 이미 morphology table을 제공한 상황이고, top-view unknown simulation에서는 성능이 크게 낮아진다. 이 결과는 “이미지만으로 concept를 알 수 없을 수 있다”는 문제를 지지한다.

## 7. 한계

1. Species-to-toxicity는 weak label이다. Wikipedia/GBIF 기반 외부 수집이지만 섭취 안전 판단용 라벨이 아니다.
2. Wikipedia `howEdible`는 모든 종을 포괄하지 않고, page 템플릿의 단순값이 문서 본문 caveat를 완전히 반영하지 못할 수 있다.
3. UCI concept는 Dataset1 이미지와 페어링되지 않는다. 따라서 UCI concept로 Dataset1 image concept를 supervised training할 수 없다.
4. Dataset1 image-toxicity는 species recognition shortcut을 배울 수 있다.
5. M3/M5 concept는 사람이 top activating examples를 보고 이름 붙이는 human-in-the-loop 단계가 필요하다.

## 8. 실행 명령

```bash
uv run python download_mushroom_data.py --skip-downloads --force-labels
uv run python experiments/mushroom_toxicity_experiment.py \
  --config_path configs/mushroom_toxicity_experiment.yaml
uv run python experiments/export_mushroom_concepts.py --model_id M3
uv run python experiments/export_mushroom_concepts.py --model_id M5
```
