# MILK10K Concept Discovery Experiment Progress

작성일: 2026-05-31

## 1. 목표

MILK10K dermoscopy-only training set에서 다음 모델군을 비교했다.

- Black-box baseline: DINOv2 global feature 기반 linear probe, MLP
- Pre-defined concept baseline: MONET concept probability 기반 sparse logistic regression
- Concept discovery baseline: DINOv2 patch token 기반 prototype concept activation
- Improved concept model: MONET + discovered concept activation을 결합한 hybrid concept bank

평가 목표는 단순 정확도 비교가 아니라 성능 손실, concept bottleneck 설명 가능성, artifact concept 의존성, faithfulness ablation까지 확인하는 것이다.

## 2. 구현 현황

추가된 주요 파일:

- `docs/milk10k_model_designs.md`
  - E1-E8 모델 설계 상세 설명
  - 각 모델의 입력, 구조, 학습 방식, 해석 포인트, 장단점 정리
- `docs/milk10k_explainability_and_multimodal_review.md`
  - 모델별 concept 설명 가능성 정리
  - concept example 시각화 링크
  - MILK10K clinical/dermoscopy two-image 사용 여부 검토
- `experiments/milk10k_concept_discovery.py`
  - MILK10K metadata/ground-truth/supplement merge
  - dermoscopy-only filtering
  - lesion id 기준 stratified group split, 70/10/20
  - DINOv2 ViT-S/14 feature extraction
  - E1-E8 실험 실행
  - macro-F1, threshold macro-F1, AUROC, balanced accuracy, ECE, malignancy recall 계산
  - concept quality proxy, artifact correlation proxy, top concept ablation 계산
- `configs/milk10k_concept_discovery.yaml`
  - 최초 full experiment 설정
- `configs/milk10k_concept_discovery_improved.yaml`
  - 성능 개선용 follow-up 설정

실험 산출물:

- `output/milk10k_concept_discovery/`
  - 최초 full run 결과
- `output/milk10k_concept_discovery_improved/`
  - 개선 full run 결과
  - `REPORT.md`
  - `PERFORMANCE_ANALYSIS.md`
  - `summary_metrics.csv`
  - `results.json`
  - `explainability_visuals/`
  - DINOv2 feature cache
  - discovered concept activation cache

참고: `output/`은 `.gitignore` 대상이므로 결과 artifact는 Git 추적 대상이 아니다.

## 3. 최초 실험 결과 요약

최초 실험 설정:

- DINOv2 model: `vit_small_patch14_dinov2`
- Image size: 224
- Concept count: K=50
- Patch pooling: top-5 mean
- Class weighting: inverse-frequency
- Final fit: train split only

주요 결과:

| ID | Model | Macro-F1 | AUROC | Balanced Acc. | Malignancy Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | Black-box linear probe | 0.3421 | 0.8610 | 0.4230 | 0.7809 |
| E2 | Black-box MLP | 0.3819 | 0.8775 | 0.3903 | 0.8579 |
| E3 | MONET sparse logistic | 0.2173 | 0.7815 | 0.3327 | 0.6069 |
| E4 | Euclidean K-means | 0.2791 | 0.8396 | 0.3674 | 0.7663 |
| E5 | Spherical K-means | 0.2603 | 0.8180 | 0.3267 | 0.7942 |
| E6 | GMM + PCA | 0.2608 | 0.8202 | 0.3342 | 0.7756 |
| E7 | Medoid prototype | 0.2598 | 0.8099 | 0.3234 | 0.7477 |

초기 결론:

- Black-box MLP가 가장 높은 macro-F1과 AUROC를 보였다.
- MONET 7개 concept만으로는 진단 분류 정보가 부족했다.
- Discovered concept 단일 bank는 black-box 대비 성능 손실이 컸다.
- Artifact proxy rate는 discovered concept에서 14-26% 수준으로 관찰됐다.

## 4. 성능 저하 원인 분석

1. 클래스 불균형이 심하다.
   - Test split 기준 `MAL_OTH=2`, `BEN_OTH=9`, `VASC=9`, `INF=10`, `DF=11`이다.
   - macro-F1은 희소 클래스 실패에 매우 민감하다.

2. inverse-frequency class weight가 과보정될 수 있다.
   - 극소수 클래스에 지나치게 높은 weight가 부여되어 common class decision boundary가 흔들릴 가능성이 있다.

3. K=50 concept bottleneck이 좁다.
   - DINOv2 patch token 정보를 50개 prototype activation으로 압축하면서 분류 정보 손실이 컸다.

4. Validation 이후 최종 모델이 train split만 사용했다.
   - epoch 선택에는 validation을 쓰지만, 최종 classifier fit에는 validation 10%가 반영되지 않았다.

5. Pre-defined MONET concept set만으로는 충분하지 않다.
   - MONET concept는 artifact/시각 단서 일부를 설명하지만, 11-class diagnosis에 필요한 fine-grained 정보를 충분히 담지 못했다.

## 5. 개선 내용

개선 설정:

- Class weight: `inverse`에서 `sqrt_inverse`로 완화
- Final fitting: validation으로 best epoch 선택 후 train+val refit
- Concept capacity: K=50에서 K=100으로 증가
- Patch pooling: top-5에서 top-10 mean으로 변경
- L1 penalty: `1e-4`에서 `1e-5`로 완화
- MLP hidden dim: 256에서 384로 증가
- E8 추가:
  - MONET features + E4/E5/E6/E7 discovered activations를 결합한 sparse hybrid concept bank

## 6. 개선 실험 결과

| ID | Model | Macro-F1 | AUROC | Balanced Acc. | ECE | Malignancy Recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| E1 | Black-box linear probe | 0.3694 | 0.8378 | 0.3919 | 0.0550 | 0.8499 |
| E2 | Black-box MLP | 0.3801 | 0.8747 | 0.3622 | 0.1880 | 0.8858 |
| E3 | MONET sparse logistic | 0.2507 | 0.8037 | 0.2735 | 0.1525 | 0.8420 |
| E4 | Euclidean K-means | 0.3353 | 0.8691 | 0.3557 | 0.0669 | 0.8685 |
| E5 | Spherical K-means | 0.3726 | 0.8540 | 0.3985 | 0.0879 | 0.8738 |
| E6 | GMM + PCA | 0.3179 | 0.8614 | 0.3472 | 0.0750 | 0.8459 |
| E7 | Medoid prototype | 0.3350 | 0.8503 | 0.3628 | 0.0959 | 0.8552 |
| E8 | Hybrid concept bank | 0.3969 | 0.8809 | 0.4414 | 0.0346 | 0.8499 |

개선 폭:

| ID | Macro-F1 Delta |
| --- | ---: |
| E1 | +0.0274 |
| E2 | -0.0018 |
| E3 | +0.0334 |
| E4 | +0.0562 |
| E5 | +0.1123 |
| E6 | +0.0572 |
| E7 | +0.0753 |

최종 기준:

- 최고 overall 모델: E8 Hybrid concept bank
  - Macro-F1: 0.3969
  - AUROC: 0.8809
  - Balanced accuracy: 0.4414
  - ECE: 0.0346
- 최고 single discovered concept 모델: E5 Spherical K-means
  - Macro-F1: 0.3726
  - 기존 E5 대비 +0.1123

## 7. 해석

E8은 black-box MLP보다 macro-F1과 AUROC가 높다. 즉, 현재 설정에서는 concept bank를 충분히 넓히고 MONET concept를 결합하면 concept 기반 모델이 black-box baseline을 넘을 수 있다.

다만 E8은 단일 bottleneck concept discovery 모델이라기보다 hybrid concept bank 모델이다. 설명 가능성 측면에서는 각 bank별 contribution과 artifact concept 사용 여부를 별도로 해석해야 한다.

E5는 단일 discovered concept 모델 중 가장 성능이 좋다. Spherical K-means는 DINOv2 patch token의 cosine geometry와 잘 맞는 것으로 보인다.

## 8. 남은 한계

- `BEN_OTH`, `MAL_OTH`는 여전히 F1이 낮거나 0이다.
  - 특히 `MAL_OTH`는 test sample이 2개뿐이라 안정적 평가가 어렵다.
- Concept quality는 proxy metric 중심이다.
  - 사람이 top activating patches를 보고 이름 붙이는 qualitative review가 아직 필요하다.
- Stability는 현재 placeholder다.
  - seed 반복 또는 bootstrap concept matching이 필요하다.
- DINOv2 518px 입력 실험은 아직 수행하지 않았다.
  - 224px 대비 feature 품질 개선 가능성이 있지만 patch token 저장 비용이 커진다.

## 9. 다음 작업 후보

1. E8 contribution 분석
   - MONET vs E4/E5/E6/E7 bank별 기여도 분해
   - artifact concept contribution 비중 계산

2. Top activating patch export
   - 각 discovered concept별 top image/patch crop 저장
   - 사람이 concept 이름을 붙일 수 있는지 검토

3. Rare-class 개선
   - focal loss
   - class-aware sampling
   - rare class merge 또는 hierarchical evaluation

4. DINOv2 518px black-box baseline
   - patch token 저장 없이 global feature만 추출하는 경량 경로 추가

5. Stability 평가
   - seed 반복 K-means
   - centroid matching
   - concept activation correlation stability

## 10. 실행 명령

기본 실험:

```bash
uv run python experiments/milk10k_concept_discovery.py \
  --config_path configs/milk10k_concept_discovery.yaml
```

개선 실험:

```bash
uv run python experiments/milk10k_concept_discovery.py \
  --config_path configs/milk10k_concept_discovery_improved.yaml
```

긴 실험은 tmux에서 실행한다.

```bash
tmux new-session -d -s milk10k_cd_improved -c /home/acsl/projects/CV-project \
  'uv run python experiments/milk10k_concept_discovery.py --config_path configs/milk10k_concept_discovery_improved.yaml'
```
