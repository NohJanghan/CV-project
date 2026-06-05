# SkinCon E1-E7 Experiment Design

작성일: 2026-06-04

## 1. 목표

SkinCon의 dermatologist-annotated clinical concept label을 기존 MILK10K concept discovery 실험 구조와 맞춰 평가한다.

- Black-box DINOv2 diagnosis baseline과 concept bottleneck 모델의 성능 차이를 비교한다.
- SkinCon ground-truth concept annotation이 diagnosis 예측에 얼마나 유용한지 확인한다.
- DINOv2 patch token에서 discovery한 latent visual concept가 SkinCon clinical concept와 얼마나 정렬되는지 proxy metric으로 확인한다.

## 2. 데이터

기본 실험은 SkinCon Fitzpatrick17k annotation을 사용한다.

- SkinCon annotation: `data/skincon/annotations_fitzpatrick17k.csv`
- Fitzpatrick17k metadata: `data/skincon/fitzpatrick17k.csv`
- Local image directory: `data/skincon/fitzpatrick17k_images`

준비 명령:

```bash
uv run python download_skincon.py
```

주의: 이 스크립트는 annotation/metadata만 받는다. Fitzpatrick17k/DDI 원본 이미지는 각 데이터셋의 접근 정책을 확인한 뒤 `image_dir`에 별도로 준비해야 한다.

## 3. E1-E7 실험 구조

| ID | Model | 입력 | 역할 |
| --- | --- | --- | --- |
| E1 | DINOv2 global linear probe | DINOv2 global feature | 가장 단순한 black-box baseline |
| E2 | DINOv2 global MLP | DINOv2 global feature | 더 강한 black-box baseline |
| E3 | Oracle SkinCon sparse logistic | 48개 SkinCon ground-truth concept | predefined concept baseline |
| E4 | Euclidean k-means concept bottleneck | DINOv2 patch-token activation | discovered concept baseline |
| E5 | Spherical k-means concept bottleneck | DINOv2 patch-token activation | cosine geometry 기반 discovered concept |
| E6 | GMM + PCA concept bottleneck | DINOv2 patch-token likelihood | density 기반 discovered concept |
| E7 | Medoid prototype concept bottleneck | 실제 train patch medoid activation | 사람이 inspect하기 쉬운 prototype concept |

E3는 이미지에서 concept를 예측하지 않고, SkinCon annotation을 oracle concept vector로 사용한다. 따라서 E3는 “SkinCon concept set 자체가 diagnosis label을 얼마나 설명하는가”를 보는 upper/reference baseline이다.

E4-E7은 train split patch token만 사용해 concept/prototype을 fitting한다. 전체 이미지의 activation은 fitted concept bank로 계산하지만, concept discovery 자체에는 validation/test patch token을 사용하지 않는다.

## 4. Split 방식

현재 split은 Fitzpatrick17k diagnosis `label` 기준 image-level stratified split이다.

절차:

1. SkinCon annotation과 Fitzpatrick17k metadata를 `md5hash` 기준으로 inner join한다.
2. `Do not consider this image == 1`인 annotation row를 제거한다.
3. diagnosis class별 sample 수를 계산하고, `min_target_count`보다 적은 class를 제거한다.
4. 남은 각 diagnosis class 내부에서 image index를 seed로 shuffle한다.
5. class별로 약 70% train, 10% validation, 20% test에 배정한다.
6. class sample이 3개 이상이면 validation/test에 최소 1장씩 들어가도록 보정한다. 기본 설정은 `min_target_count: 20`이라 main run에서는 모든 class가 이 조건을 만족한다.

이 split이 보장하는 것:

- train/validation/test의 diagnosis class 분포가 크게 무너지지 않는다.
- E4-E7 concept discovery는 train split patch token만 사용하므로 test leakage를 피한다.
- validation은 epoch/model selection에만 사용하고, test는 최종 보고에만 사용한다.

이 split이 보장하지 못하는 것:

- SkinCon/Fitzpatrick17k metadata에는 안정적인 patient id 또는 lesion id가 없으므로 patient-level split이나 lesion-level group split은 보장할 수 없다.
- 동일 환자/동일 병변의 중복 이미지가 metadata에 숨어 있다면 image-level split만으로는 이를 막을 수 없다.
- concept label 분포까지 완전하게 stratify하지 않는다. stratification 기준은 diagnosis `label`이다.

## 5. Metrics

공통 diagnosis metric:

- accuracy
- macro-F1 with argmax prediction
- macro-AUROC one-vs-rest
- balanced accuracy
- per-class F1/recall

Concept model 추가 metric:

- top concept ablation: predicted class confidence가 top contribution concept 제거 후 얼마나 떨어지는지 측정
- SkinCon alignment proxy: discovered concept activation과 48개 SkinCon ground-truth concept label 사이의 최대 절대 상관
- diversity proxy: prototype/centroid 간 평균 절대 cosine similarity 기반 다양성

## 6. 실행 명령

```bash
uv run python experiments/skincon_concept_experiment.py \
  --config_path configs/skincon_concept_experiment.yaml
```

긴 실행은 tmux 사용을 권장한다.

```bash
tmux new-session -d -s skincon_e1_e7 -c /home/acsl/projects/CV-project \
  'uv run python experiments/skincon_concept_experiment.py --config_path configs/skincon_concept_experiment.yaml'
```

## 7. 검토 필요 사항

- Fitzpatrick17k/DDI 이미지 접근 정책과 재배포 가능 범위를 먼저 확인해야 한다.
- patient/lesion id가 없는 현재 metadata 기반 split의 한계를 결과 해석에 명시해야 한다.
- diagnosis label을 target으로 둘지, SkinCon concept prediction 자체를 별도 primary task로 둘지 실험 목적을 확정해야 한다.
- `min_target_count`, `k`, `topk_pool`, class weight 방식은 full run 전 최종 확정이 필요하다.
