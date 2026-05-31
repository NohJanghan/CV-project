# MILK10K Explainability and Two-Image Input Review

작성일: 2026-05-31

## 1. 결론 요약

현재 E1-E8 중 concept-level explainability를 직접 충족하는 모델은 E3-E8이다.

- E1/E2는 black-box baseline이다. 내부 concept가 없으므로 concept explanation을 직접 제공하지 못한다.
- E3는 사람이 이름 붙인 MONET concept를 직접 사용하므로 가장 직접적인 concept explanation이 가능하다.
- E4/E5/E6/E7은 DINOv2 patch token에서 발견한 latent visual concept를 top activating examples로 사람이 해석해야 한다.
- E8은 MONET와 discovered concept bank를 합친 모델이므로 bank별 provenance를 표시해야 한다.

MILK10K의 두 장 이미지 사용 여부:

- 현재 concept discovery 실험 E1-E8은 dermoscopy image만 사용한다.
- clinical close-up image는 현재 실험 모델 입력에 포함되지 않는다.
- 기존 `src/data/milk10k.py` 기반 Attention-CBM 파이프라인도 two-image pair model이 아니다. 한 번에 한 장의 이미지를 입력으로 받는 single-image 설계다.

## 2. 전체 Explainability Map

아래 그림은 각 모델이 어떤 방식으로 설명 가능한지 요약한다.

![Model explainability map](../output/milk10k_concept_discovery_improved/explainability_visuals/model_explainability_map.png)

## 3. E1: Black-box Linear Probe

### 현재 설명 가능성 상태

E1은 concept bottleneck이 없다. 입력은 DINOv2 global feature이고, classifier는 linear layer다.

```text
image -> DINOv2 global feature -> linear classifier -> diagnosis
```

따라서 E1에서 사람이 해석할 수 있는 "concept"는 모델 내부에 존재하지 않는다.

### 사람이 볼 수 있는 설명

E1에서 가능한 설명은 post-hoc explanation이다.

- Class weight direction 분석
  - 어떤 DINOv2 feature dimension이 특정 class logit에 영향을 주는지 볼 수 있다.
  - 단, feature dimension은 사람이 이름 붙일 수 있는 concept가 아니다.
- Nearest-neighbor evidence
  - 예측 대상 이미지와 DINOv2 global feature가 가까운 train image를 보여준다.
  - 사람이 "비슷한 병변 예시"를 보고 판단 근거를 추론한다.
- Saliency, attention rollout, occlusion map
  - 이미지의 어느 영역이 예측에 영향을 주는지 보여준다.
  - 그러나 영역이 어떤 concept인지 자동으로 이름 붙이지는 못한다.

### 판정

E1은 성능 baseline으로는 유효하지만, concept-level explainability 요구사항은 직접 충족하지 못한다. 논문/보고서에서는 "non-concept black-box baseline"으로 명확히 분류해야 한다.

## 4. E2: Black-box MLP

### 현재 설명 가능성 상태

E2도 concept bottleneck이 없다. E1과 차이는 classifier가 MLP라는 점뿐이다.

```text
image -> DINOv2 global feature -> MLP classifier -> diagnosis
```

### 사람이 볼 수 있는 설명

E2도 E1과 동일하게 post-hoc explanation만 가능하다.

- Nearest-neighbor evidence
- Saliency or occlusion heatmap
- Feature attribution

다만 MLP는 비선형 classifier이므로 E1보다 weight 해석이 더 어렵다.

### 판정

E2는 가장 중요한 performance baseline이지만, concept explanation 모델은 아니다. E2와 concept models를 비교할 때는 "성능은 강하지만 설명 가능성은 약하다"는 위치로 둬야 한다.

## 5. E3: MONET Pre-defined Concept Model

### Concept 정의

E3는 사람이 이미 이름 붙인 7개 MONET concept probability를 사용한다.

- ulceration/crust
- hair
- vessels
- erythema
- pigmentation
- gel/water/drop/liquid
- skin markings/pen ink

### 사람이 해석하는 방법

E3는 concept 이름이 이미 정해져 있으므로 가장 직접적으로 해석할 수 있다.

해석 절차:

1. 이미지별 MONET concept score를 본다.
2. 예측 class에 대한 classifier weight를 본다.
3. `weight[class, concept] * concept_score`로 contribution을 계산한다.
4. 높은 contribution concept를 설명으로 제시한다.

예시:

```text
Predicted class = BCC
Top positive concepts:
  pigmentation: +0.42
  ulceration/crust: +0.31
Top artifact concepts:
  hair: +0.18
```

이 경우 pigmentation과 ulceration/crust가 예측에 기여했고, hair도 artifact signal로 사용됐는지 확인해야 한다.

### 시각화

아래 grid는 각 MONET concept score가 높은 dermoscopy image 예시다. 사람이 각 concept가 실제로 보이는지 검토할 수 있다.

![E3 MONET concepts](../output/milk10k_concept_discovery_improved/explainability_visuals/E3_monet_concepts.png)

### 한계

MONET concept는 사람이 해석하기 쉽지만 7개뿐이다. MILK10K 11-class diagnosis를 설명하기에는 concept vocabulary가 좁다.

## 6. E4: Euclidean K-means Concept Model

### Concept 정의

E4 concept는 DINOv2 patch token을 Euclidean distance 기준으로 cluster한 centroid다.

```text
train patch tokens -> standardization -> Euclidean K-means -> centroids
```

각 centroid는 하나의 latent visual concept 후보로 본다.

### 사람이 해석하는 방법

E4 concept에는 처음부터 이름이 없다. 사람이 top activating examples를 보고 이름을 붙여야 한다.

해석 절차:

1. 특정 concept k에 대해 activation이 높은 image 또는 patch를 모은다.
2. top examples에서 반복되는 시각 패턴을 찾는다.
3. 사람이 concept name을 부여한다.
   - 예: dark pigment network
   - 예: hair occlusion
   - 예: blue-white structure
   - 예: shiny gel artifact
4. classifier weight와 activation을 곱해 class contribution을 계산한다.
5. artifact concept인지도 MONET correlation과 시각 검토로 확인한다.

### 시각화

아래 grid는 E4에서 class selectivity가 높은 concept들의 top activating image 예시다.

![E4 concept examples](../output/milk10k_concept_discovery_improved/explainability_visuals/E4_concept_examples.png)

### 한계

Centroid는 실제 patch가 아니라 평균 prototype이다. 따라서 사람이 concept를 해석하려면 centroid 자체보다 top activating examples를 봐야 한다.

## 7. E5: Spherical K-means Concept Model

### Concept 정의

E5 concept는 L2-normalized DINOv2 patch token을 cosine similarity 기준으로 cluster한 centroid다.

```text
train patch tokens -> L2 normalization -> spherical K-means -> normalized centroids
```

### 사람이 해석하는 방법

E5도 concept 이름은 자동으로 주어지지 않는다. top activating examples를 보고 사람이 이름을 붙인다.

E4와의 차이는 similarity 기준이다.

- E4: Euclidean distance
- E5: Cosine similarity

DINOv2 feature는 direction이 semantic similarity를 잘 담는 경우가 많으므로, E5 concept는 더 일관된 visual pattern으로 나타날 가능성이 있다.

해석 절차:

1. concept k의 top cosine activation images를 본다.
2. 반복되는 병변 구조나 artifact를 찾는다.
3. 사람이 concept 이름을 붙인다.
4. sparse classifier contribution으로 예측 근거를 계산한다.

### 시각화

아래 grid는 E5에서 class selectivity가 높은 concept들의 top activating image 예시다.

![E5 concept examples](../output/milk10k_concept_discovery_improved/explainability_visuals/E5_concept_examples.png)

### 이번 실험에서의 의미

E5는 단일 discovered concept bank 중 최고 성능을 보였다. 따라서 현재 설계에서는 가장 중요한 latent concept discovery candidate다.

## 8. E6: GMM + PCA Concept Model

### Concept 정의

E6 concept는 patch feature distribution의 Gaussian mixture component다.

```text
train patch tokens -> PCA -> diagonal GMM -> Gaussian components
```

각 component는 centroid 하나가 아니라 평균, 분산, mixture weight를 가진 density mode다.

### 사람이 해석하는 방법

E6에서는 concept를 "특정 patch 분포 mode"로 해석한다.

해석 절차:

1. component likelihood가 높은 image 또는 patch를 모은다.
2. top examples에서 공통 시각 패턴을 찾는다.
3. 사람이 component에 이름을 붙인다.
4. contribution은 classifier weight와 component activation score로 계산한다.

### 시각화

아래 grid는 E6에서 class selectivity가 높은 component들의 top activating image 예시다.

![E6 concept examples](../output/milk10k_concept_discovery_improved/explainability_visuals/E6_concept_examples.png)

### 한계

GMM score는 log-likelihood scale이라 E4/E5의 similarity score보다 직관성이 낮다. 사람이 해석할 때는 score 자체보다 top examples 중심으로 봐야 한다.

## 9. E7: Medoid Prototype Concept Model

### Concept 정의

E7 concept는 실제 train patch exemplar다. Spherical K-means centroid에 가장 가까운 실제 patch를 medoid prototype으로 선택한다.

```text
train patch tokens -> spherical K-means -> nearest real patch per centroid -> medoid concepts
```

### 사람이 해석하는 방법

E7은 discovered concept 중 사람이 해석하기 가장 쉽다. Prototype이 실제 patch이기 때문이다.

해석 절차:

1. concept medoid patch를 직접 본다.
2. medoid와 top activating images를 함께 본다.
3. 반복되는 시각 패턴을 기준으로 사람이 이름을 붙인다.
4. sparse classifier contribution을 계산한다.

### 시각화

아래 grid는 E7에서 class selectivity가 높은 concept들의 top activating image 예시다.

![E7 concept examples](../output/milk10k_concept_discovery_improved/explainability_visuals/E7_concept_examples.png)

### 장점

E7은 성능만 보면 E5보다 낮지만, exemplar explanation에는 유리하다. 보고서에서 사람에게 concept를 보여주는 용도로는 E7이 가장 직접적이다.

## 10. E8: Hybrid Concept Bank

### Concept 정의

E8은 다음 concept bank를 concat한 모델이다.

```text
E8 concept vector =
  [MONET 7 concepts,
   E4 Euclidean concepts,
   E5 Spherical concepts,
   E6 GMM concepts,
   E7 Medoid concepts]
```

개선 run 기준 dimension은 407이다.

```text
7 + 100 + 100 + 100 + 100 = 407
```

### 사람이 해석하는 방법

E8 설명은 bank-aware로 해야 한다. 단순히 concept index만 보여주면 해석이 불가능하다.

해석 절차:

1. 예측 class의 top contribution concept를 계산한다.
2. 각 concept의 source bank를 표시한다.
   - MONET concept인지
   - Euclidean prototype인지
   - Spherical prototype인지
   - GMM component인지
   - Medoid exemplar인지
3. source bank에 맞는 방식으로 concept를 보여준다.
4. artifact bank/concept인지 표시한다.

예시 형식:

```text
Predicted class = MEL

Top evidence:
1. E5_concept_037, contribution +0.41
   - Explanation type: spherical prototype
   - Human label: irregular dark network

2. MONET_pigmented, contribution +0.26
   - Explanation type: predefined concept

3. E7_concept_012, contribution +0.19
   - Explanation type: medoid exemplar
   - Human label: blue-white patch candidate
```

### 시각화

아래 grid는 E8 hybrid bank에서 class selectivity가 높은 source concepts의 top activating image 예시다.

![E8 hybrid concept examples](../output/milk10k_concept_discovery_improved/explainability_visuals/E8_hybrid_concept_examples.png)

### 한계

E8은 성능이 가장 좋지만 설명이 복합적이다. 최종 보고서에서는 반드시 concept source bank와 사람이 붙인 label을 함께 보여줘야 한다.

## 11. Two-Image Input Review

MILK10K metadata에는 같은 `lesion_id`에 대해 보통 두 image row가 있다.

- `image_type = clinical: close-up`
- `image_type = dermoscopic`

따라서 "MILK10K는 이미지 두 장을 받는다"는 말은 lesion-level로 보면 맞다.

### 현재 concept discovery experiment

현재 `experiments/milk10k_concept_discovery.py`는 dermoscopy-only로 설계되어 있다.

코드 흐름:

```text
metadata
  -> filter image_type == "dermoscopic"
  -> one image per lesion
  -> DINOv2 feature extraction
  -> E1-E8 models
```

결론:

- clinical close-up image는 사용하지 않는다.
- metadata의 MONET concept와 diagnosis label은 사용하지만, visual input은 dermoscopic image 한 장이다.
- 이 설계는 최초 실험 캔버스의 "기본 입력: dermoscopy-only, clinical close-up은 확장 실험"과 일치한다.

### 기존 `src/data/milk10k.py` Attention-CBM path

기존 프로젝트의 `MILK10KDataset`도 모델 입력 자체는 single image다.

```text
one metadata row -> one image tensor -> model
```

중요한 차이:

- `src/data/milk10k.py`는 기본적으로 `image_type`을 dermoscopic으로 필터링하지 않는다.
- 따라서 metadata CSV 전체를 쓰면 clinical image와 dermoscopic image가 각각 독립 sample처럼 들어갈 수 있다.
- 이 경우 두 이미지를 pair로 묶어 쓰는 것이 아니라, 한 장짜리 sample 두 개로 취급한다.
- 기존 split도 row shuffle 기반이라 clinical/dermoscopic pair가 서로 다른 split에 들어갈 수 있는 위험이 있다.

따라서 기존 Attention-CBM path도 true two-image model은 아니다.

## 12. Two-Image Model로 확장하려면 필요한 설계

MILK10K의 두 장 이미지를 모두 쓰려면 lesion-level paired dataset이 필요하다.

### Dataset 변경

현재 구조:

```text
row-level sample:
  image_path
  label
```

필요한 구조:

```text
lesion-level sample:
  dermoscopic_image_path
  clinical_closeup_image_path
  metadata
  label
```

Split은 반드시 `lesion_id` 기준으로 해야 한다.

### Model 변경 후보

#### Option A: Late Fusion

```text
derm image -> DINOv2 derm encoder -> z_derm
clinical image -> DINOv2 clinical encoder -> z_clinical
[z_derm, z_clinical, metadata] -> classifier
```

장점:

- 구현이 가장 단순하다.
- derm-only baseline과 비교하기 쉽다.

단점:

- modality별 concept를 명시적으로 분리하지 않으면 설명이 약하다.

#### Option B: Modality-specific Concept Bottleneck

```text
derm image -> derm concepts
clinical image -> clinical concepts
metadata -> metadata concepts
[derm concepts, clinical concepts, metadata concepts] -> sparse classifier
```

장점:

- 설명을 modality별로 분해할 수 있다.
- "dermoscopy concept"와 "clinical close-up concept"를 따로 제시할 수 있다.

단점:

- concept discovery와 visualization이 두 배로 필요하다.
- missing modality handling이 필요하다.

#### Option C: Hybrid E8 확장

```text
E8_derm = MONET + derm discovered concepts
E8_clinical = clinical discovered concepts
E8_pair = [E8_derm, E8_clinical, metadata]
```

이 프로젝트에서는 Option C가 가장 자연스럽다. 이미 E8이 concept bank concat 구조이므로 clinical branch를 추가하기 쉽다.

## 13. 현재 권장 기준

현재 결과를 보고서에 쓸 때는 다음처럼 명확히 써야 한다.

1. 본 실험의 visual input은 dermoscopy image 한 장이다.
2. MILK10K의 paired clinical close-up image는 아직 사용하지 않았다.
3. E1/E2는 explainable model이 아니라 black-box performance baseline이다.
4. E3-E8은 concept-level explanation이 가능하지만, E4-E8의 latent concept는 top activating examples를 사람이 보고 이름 붙여야 한다.
5. 최종 explainable model 후보는 E5와 E8이다.
   - E5: 단일 discovered concept model로 가장 간결하다.
   - E8: 성능이 가장 좋지만 bank-aware explanation이 필요하다.
