# MILK10K Model Design Details

작성일: 2026-05-31

이 문서는 MILK10K concept discovery 실험에서 사용한 baseline과 실험 모델 E1-E8의 설계를 자세히 설명한다.

## 1. 공통 실험 조건

모든 모델은 같은 데이터 split과 같은 11-class diagnosis target을 사용한다.

- Dataset: MILK10K training set
- Image subset: dermoscopy-only
- Split: lesion id 기준 stratified group split
- Split ratio: train/val/test = 70/10/20
- Target classes: `AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`
- Image encoder: DINOv2 ViT-S/14 via `timm`
- Default image size: 224
- Validation usage: model selection
- Test usage: final reporting only

공통 metric:

- Macro-F1 with argmax prediction
- Official-style macro-F1 with 0.5 threshold
- Macro AUROC
- Balanced accuracy
- Expected calibration error, ECE
- Malignancy-group recall
- Per-class F1 and recall

Concept 기반 모델은 추가로 다음을 계산한다.

- Top concept ablation
- MONET alignment proxy
- Artifact concept rate proxy
- Sparsity proxy
- Class selectivity proxy

## 2. E1: Black-box Linear Probe

### 목적

E1은 가장 단순한 DINOv2 feature baseline이다. Concept bottleneck 없이 DINOv2 global feature가 MILK10K 11-class diagnosis를 어느 정도 분리하는지 확인한다.

### 입력

- Dermoscopy image
- DINOv2 ViT-S/14 global feature
- Feature dimension: ViT-S 기준 384

### 모델 구조

```text
image
  -> DINOv2 ViT-S/14
  -> global feature z
  -> linear classifier
  -> 11-class logits
```

Classifier는 단일 linear layer다.

```text
logits = Wz + b
```

### 학습

- DINOv2는 frozen feature extractor로 사용한다.
- Linear classifier만 학습한다.
- Cross-entropy loss를 사용한다.
- 초기 run은 inverse-frequency class weight를 사용했다.
- 개선 run은 sqrt-inverse class weight와 train+val refit을 사용했다.

### 해석

E1은 concept bottleneck이 없는 하한선 성격의 black-box baseline이다. MLP보다 용량이 낮기 때문에 feature 자체의 선형 분리 가능성을 확인하는 데 적합하다.

### 장점

- 빠르다.
- 결과 해석이 단순하다.
- DINOv2 feature quality를 직접 평가할 수 있다.

### 한계

- Concept 설명이 없다.
- 비선형 decision boundary를 학습하지 못한다.
- Patch-level 시각 단서를 직접 분석하지 않는다.

## 3. E2: Black-box MLP

### 목적

E2는 concept bottleneck이 없는 가장 강한 black-box baseline이다. Concept 기반 모델이 이 모델 대비 어느 정도 성능 손실을 보이는지 비교 기준으로 사용한다.

### 입력

- Dermoscopy image
- DINOv2 global feature

### 모델 구조

```text
image
  -> DINOv2 ViT-S/14
  -> global feature z
  -> Linear
  -> ReLU
  -> Dropout
  -> Linear
  -> 11-class logits
```

개선 run에서는 hidden dimension을 256에서 384로 늘리고 dropout을 0.25로 설정했다.

### 학습

- DINOv2는 frozen이다.
- MLP classifier만 학습한다.
- Cross-entropy loss를 사용한다.
- Validation macro-F1 기준 best epoch를 선택한다.

### 해석

E2는 설명 가능성 제약이 없는 모델이다. 따라서 concept 기반 모델이 E2에 가까워질수록 "설명 가능성을 얻으면서 성능 손실을 줄였다"고 볼 수 있다.

### 장점

- 가장 강한 black-box baseline이다.
- Global feature의 비선형 조합을 학습할 수 있다.
- 구현과 학습이 단순하다.

### 한계

- 예측 근거가 concept 단위로 드러나지 않는다.
- Patch-level 단서나 artifact 의존성을 구조적으로 분리하지 못한다.
- Calibration은 개선 run에서 ECE가 높게 나왔다.

## 4. E3: Pre-defined MONET Sparse Logistic Regression

### 목적

E3는 사람이 미리 정의한 concept set만으로 diagnosis를 예측하는 baseline이다. "이미 있는 MONET concept 7개만으로 충분한가"를 검증한다.

### 입력

MILK10K metadata에 있는 MONET concept probabilities 7개를 사용한다.

- `MONET_ulceration_crust`
- `MONET_hair`
- `MONET_vasculature_vessels`
- `MONET_erythema`
- `MONET_pigmented`
- `MONET_gel_water_drop_fluid_dermoscopy_liquid`
- `MONET_skin_markings_pen_ink_purple_pen`

### 모델 구조

```text
MONET concept vector c in R^7
  -> sparse linear classifier
  -> 11-class logits
```

```text
logits = Wc + b
```

L1 penalty를 사용해서 classifier weight를 sparse하게 만든다.

### 학습

- Image encoder를 사용하지 않는다.
- Input은 metadata concept probability다.
- Sparse multinomial logistic regression 형태로 학습한다.

### 해석

E3의 weight는 concept와 class 사이의 직접 연결로 해석할 수 있다. 예를 들어 특정 class logit에 `hair` weight가 크면, 해당 class 예측이 hair artifact에 민감하다는 뜻이다.

### 장점

- 가장 해석하기 쉽다.
- Concept intervention과 ablation이 직접적이다.
- Artifact concept 의존성을 명확히 볼 수 있다.

### 한계

- 7개 concept만으로 11개 진단 class를 설명하기에는 정보가 부족하다.
- MONET concept가 진단 병변 morphology 전체를 포괄하지 않는다.
- Hair, gel, pen marking 같은 artifact concept가 diagnosis signal과 섞일 수 있다.

## 5. Concept Discovery 공통 구조

E4-E7은 모두 DINOv2 patch token에서 latent visual concept를 발견한다.

### 공통 입력

```text
image
  -> DINOv2 ViT-S/14
  -> patch tokens P in R^(N patches x D)
```

224 입력에서는 ViT-S/14 patch token이 대략 16x16 grid, 즉 256개 patch token이다. 각 token dimension은 384다.

### 공통 discovery rule

Concept discovery는 train split patch token만 사용한다.

```text
train images
  -> patch tokens
  -> sampled train patches
  -> clustering / prototype fitting
  -> K visual concepts
```

Test patch token은 concept fitting에 사용하지 않는다. 이는 test leakage를 막기 위한 조건이다.

### 공통 activation rule

각 image에 대해 K개 concept activation vector를 만든다.

```text
patch tokens of image i
  -> similarity or likelihood to each concept k
  -> top-k patch scores per concept
  -> mean pooling
  -> concept activation c_i in R^K
```

초기 run:

- K=50
- top-5 mean pooling

개선 run:

- K=100
- top-10 mean pooling

### 공통 classifier

Concept activation vector로 sparse linear classifier를 학습한다.

```text
concept activation c_i
  -> sparse linear classifier
  -> 11-class logits
```

이 구조에서 class y에 대한 concept k의 기여도는 다음처럼 해석한다.

```text
contribution_k = W[y, k] * c_i[k]
```

### 공통 faithfulness 평가

Top concept ablation을 사용한다.

```text
1. predicted class y 선택
2. contribution이 큰 top-m concept 선택
3. 해당 concept activation을 제거
4. P(y) 감소량 측정
```

감소량이 클수록 classifier가 해당 concept에 실제로 의존한다는 뜻이다.

## 6. E4: Euclidean K-means Concept Discovery

### 목적

E4는 가장 기본적인 prototype discovery baseline이다. Patch feature를 Euclidean space에서 K개 cluster로 나누고, cluster center를 concept prototype으로 사용한다.

### 설계

```text
train patch tokens
  -> standardization
  -> Euclidean K-means
  -> K centroids
```

Activation은 patch token과 centroid 사이의 negative squared distance로 계산한다.

```text
score(p, k) = -||standardize(p) - centroid_k||^2
```

Image-level concept activation은 concept별 top patch score의 평균이다.

### 해석

E4 concept는 "DINOv2 feature space에서 서로 가까운 patch들의 평균 패턴"이다. Euclidean distance는 feature magnitude와 direction을 모두 사용한다.

### 장점

- 가장 단순하고 재현하기 쉽다.
- Prototype center가 명확하다.
- 개선 run에서 AUROC가 높게 나왔다.

### 한계

- DINOv2 embedding은 cosine geometry가 더 적합할 수 있다.
- Feature magnitude 차이가 cluster를 지배할 수 있다.
- Centroid는 실제 patch가 아니므로 시각화 시 대표 crop을 별도로 찾아야 한다.

### 이번 결과

초기 run:

- Macro-F1: 0.2791
- AUROC: 0.8396

개선 run:

- Macro-F1: 0.3353
- AUROC: 0.8691

K=100과 class weight 완화로 성능이 명확히 개선됐다.

## 7. E5: Spherical K-means Concept Discovery

### 목적

E5는 DINOv2 patch token의 방향 정보를 중심으로 concept를 발견한다. DINOv2 feature는 cosine similarity 기반 검색/분류에 잘 맞는 경우가 많기 때문에 main proposed candidate로 설정했다.

### 설계

```text
train patch tokens
  -> L2 normalization
  -> spherical K-means
  -> K normalized centroids
```

Activation은 cosine similarity로 계산한다.

```text
score(p, k) = normalize(p) dot normalize(centroid_k)
```

### 해석

E5 concept는 "feature direction이 유사한 patch pattern"이다. 밝기, contrast, embedding norm 같은 magnitude 요인을 줄이고 semantic direction에 더 집중한다.

### 장점

- DINOv2 embedding geometry와 잘 맞을 가능성이 높다.
- 개선 run에서 단일 discovered concept 모델 중 최고 성능을 보였다.
- Activation score가 cosine similarity라 해석이 비교적 직관적이다.

### 한계

- Centroid가 실제 patch는 아니다.
- 서로 비슷한 방향의 concept가 중복될 수 있다.
- Artifact concept도 cosine cluster로 안정적으로 잡힐 수 있다.

### 이번 결과

초기 run:

- Macro-F1: 0.2603
- AUROC: 0.8180

개선 run:

- Macro-F1: 0.3726
- AUROC: 0.8540

Macro-F1이 +0.1123으로 가장 크게 개선됐다. 현재 단일 discovered concept bank 중 가장 유효한 설계다.

## 8. E6: GMM + PCA Concept Discovery

### 목적

E6는 hard clustering 대신 soft density model을 사용한다. Patch token이 여러 concept에 부분적으로 속할 수 있다는 가정을 반영한다.

### 설계

```text
train patch tokens
  -> PCA projection
  -> diagonal-covariance Gaussian mixture model
  -> K Gaussian components
```

PCA를 먼저 적용하는 이유:

- Full 384-dim covariance modeling은 비용이 크다.
- Diagonal GMM도 고차원에서는 불안정해질 수 있다.
- PCA로 주요 variation만 남기면 likelihood 계산이 더 안정적이다.

Activation은 각 Gaussian component의 log probability 또는 posterior-like score로 계산한다.

```text
score(p, k) = log N(PCA(p); mu_k, diag(var_k)) + log pi_k
```

### 해석

E6 concept는 prototype point라기보다 "patch feature 분포의 하나의 mode"다. 한 concept는 평균과 분산을 함께 가지므로 patch variation을 일부 반영한다.

### 장점

- Soft concept assignment에 가깝다.
- Cluster 내부 분산을 모델링한다.
- Artifact rate proxy는 개선 run에서 가장 낮았다.

### 한계

- Log-likelihood scale이 다른 method의 similarity score와 다르다.
- Diagonal covariance 가정은 patch feature correlation을 충분히 반영하지 못한다.
- Ablation 결과가 음수 또는 작게 나올 수 있어 contribution 해석이 까다롭다.

### 이번 결과

초기 run:

- Macro-F1: 0.2608
- AUROC: 0.8202

개선 run:

- Macro-F1: 0.3179
- AUROC: 0.8614

성능은 개선됐지만 E5보다 낮았다. 단독 모델보다는 hybrid bank의 보조 signal로 쓰는 편이 적합하다.

## 9. E7: Medoid Prototype Concept Discovery

### 목적

E7은 concept prototype을 실제 patch exemplar로 제한한다. 설명 시 "이 concept는 실제로 이런 patch와 유사하다"라고 보여주기 쉽게 만들기 위한 설계다.

### 설계

```text
train patch tokens
  -> L2 normalization
  -> spherical K-means
  -> each centroid nearest actual patch selected
  -> K medoid prototypes
```

Activation은 patch token과 medoid patch prototype 사이의 cosine similarity로 계산한다.

```text
score(p, k) = normalize(p) dot normalize(medoid_k)
```

### 해석

E7 concept는 실제 train patch 하나가 prototype이다. 따라서 centroid보다 visual explanation이 직접적이다.

### 장점

- Prototype이 실제 이미지 patch라 qualitative review에 유리하다.
- Exemplar-based explanation을 만들기 쉽다.
- Top activating patch export와 잘 맞는다.

### 한계

- 하나의 medoid patch가 cluster 전체 평균을 충분히 대표하지 못할 수 있다.
- Outlier patch가 선택되면 concept 품질이 떨어진다.
- 성능은 E5보다 낮았다.

### 이번 결과

초기 run:

- Macro-F1: 0.2598
- AUROC: 0.8099

개선 run:

- Macro-F1: 0.3350
- AUROC: 0.8503

성능은 E4와 유사하게 개선됐고, 설명 가능성 측면에서는 E4보다 유리하다.

## 10. E8: Hybrid MONET + Discovered Concept Bank

### 목적

E8은 pre-defined concept와 discovered concept를 결합해 성능과 설명 가능성의 균형을 개선하기 위한 모델이다.

### 입력

다음 feature를 concat한다.

- E3 MONET 7-dim concept vector
- E4 Euclidean K-means activation
- E5 Spherical K-means activation
- E6 GMM + PCA activation
- E7 Medoid prototype activation

개선 run 기준 concept dimension:

```text
7 + 100 + 100 + 100 + 100 = 407
```

### 모델 구조

```text
[MONET concepts, discovered activations]
  -> sparse linear classifier
  -> 11-class logits
```

### 해석

E8은 여러 concept bank 중 classifier가 유용한 signal을 선택하게 한다. Sparse linear classifier를 사용하므로 class별로 어떤 bank와 concept가 쓰였는지 분석할 수 있다.

### 장점

- 이번 실험에서 최고 overall 성능을 보였다.
- MONET의 사람이 정의한 concept와 discovered visual pattern을 동시에 사용한다.
- Sparse weight 분석으로 bank별 contribution을 분해할 수 있다.

### 한계

- 단일 discovery method보다 구조가 복합적이다.
- Explanation을 제시할 때 bank별 provenance를 명확히 표시해야 한다.
- Artifact concept가 여러 bank에 중복으로 들어갈 수 있다.

### 이번 결과

개선 run:

- Macro-F1: 0.3969
- AUROC: 0.8809
- Balanced accuracy: 0.4414
- ECE: 0.0346

E8은 black-box MLP보다 macro-F1과 AUROC가 높았다. 현재 결과 기준 가장 강한 모델이다.

## 11. 모델 간 비교 요약

| Model | Input | Main constraint | Explanation strength | Performance role |
| --- | --- | --- | --- | --- |
| E1 Linear | DINOv2 global | Linear boundary | Low | Simple black-box baseline |
| E2 MLP | DINOv2 global | No concept bottleneck | Low | Strong black-box baseline |
| E3 MONET | 7 predefined concepts | Very narrow concept set | High | Human-defined concept baseline |
| E4 Euclidean | Patch prototypes | Euclidean geometry | Medium | Basic discovery baseline |
| E5 Spherical | Patch prototypes | Cosine geometry | Medium | Best single discovery model |
| E6 GMM + PCA | Patch density modes | PCA + diagonal Gaussian | Medium-low | Soft concept baseline |
| E7 Medoid | Actual patch exemplars | Exemplar prototype | High | Exemplar explanation model |
| E8 Hybrid | MONET + all discovered banks | Larger sparse concept bank | Medium-high | Best overall model |

## 12. 현재 결론

현재 실험에서 가장 중요한 결론은 다음이다.

1. MONET 7개만으로는 부족하다.
2. DINOv2 patch token에서 발견한 latent concept는 diagnosis signal을 가진다.
3. 단일 discovered concept bank 중에서는 spherical K-means가 가장 좋다.
4. Concept bank를 넓히고 MONET와 결합하면 black-box MLP를 넘을 수 있다.
5. E8은 성능 기준으로 가장 좋지만, 설명 보고서에서는 bank별 contribution과 artifact reliance를 반드시 분해해야 한다.
