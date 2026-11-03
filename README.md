# motorsim — 모터 모델링 · 시뮬레이션 · 파라미터 식별

DC 모터 / PMSM 모델을 설정 파일 하나로 구성해서 폐루프 시뮬레이션을 돌리고, 그 데이터로
**모터 파라미터(R, L, Kt, J)와 마찰 모델(쿨롱, 점성, Stribeck, LuGre)을 식별·검증**하는 프레임워크입니다.
시뮬레이션 데이터 대신 실제 측정 CSV를 넣으면 같은 파이프라인으로 실험 데이터 식별과 sim-vs-real 비교를 할 수 있습니다.

```
config (YAML) ──► simulate ──► 측정 데이터 (CSV/NPZ) ──┬─► LS 식별 ──► 마찰모델 비교(AIC/BIC) ──► OE 정밀화 ──► 검증
                     ▲                                 │
     실제 측정 CSV ───┼─────────────────────────────────┘
                     └─► dataset (파라미터 랜덤화) ──► CNN 학습 ──► predict
```

## 설치 / 실행

Python 3.10+ (현재 PC: `py -3.13`). 필요한 패키지: `numpy scipy pandas matplotlib pyyaml torch`(학습할 때만 torch 필요).

```powershell
py -3.13 -m pip install -r requirements.txt   # 대부분 이미 설치되어 있음
py -3.13 run.py <command> -c config/<file>.yaml
py -3.13 -m pytest tests -q                    # 물리/식별 sanity test (약 15초)
```

결과는 기본적으로 `export/<날짜시간>_<command>_<name>/` 에 저장됩니다(`-o` 로 지정 가능).

## 명령어

| 명령 | 하는 일 |
|---|---|
| `simulate -c cfg [--mode M]` | 폐루프 시뮬레이션 1회 (batch 가능) → `sim.csv`, `sim.npz`, `sim.png` |
| `demo -c cfg` | **합성 end-to-end**: 숨겨진 "진짜" 모터로 식별·검증용 실험 데이터 생성 → 식별 → 검증 → 정답 대비 오차 리포트 |
| `identify -c cfg --data id.csv [--val val.csv] [--fmap fmap.csv]` | 측정 데이터(실험 또는 시뮬레이션)로 LS → 마찰모델 선택 → OE → 검증 → `report.md`, `identified_params.yaml` |
| `validate -c cfg -p identified_params.yaml --data val.csv` | 측정 전압을 모델에 재생해서 전류·속도를 비교 (sim vs real) |
| `friction-map -c cfg [--data fmap.csv]` | 정속 계단 시험 → 정상상태 (ω, T) 맵 → 마찰모델 피팅/비교 |
| `dataset -c cfg [-n N]` | 파라미터를 랜덤화한 학습 데이터셋 생성 (`.npz`) |
| `train -d dataset.npz [--epochs E]` | 1D-CNN 파라미터 추정기 학습 → `.pt`, 정확도 표, 산점도 |
| `predict -c cfg -m model.pt --data rec.csv` | 학습된 네트워크로 녹화 데이터의 파라미터 추정 |

## 설정 파일 (config/)

| 파일 | 용도 |
|---|---|
| `dc_basic.yaml`, `pmsm_basic.yaml` | 모든 키를 주석으로 설명한 기본 시뮬레이션 |
| `demo_dc.yaml`, `demo_pmsm.yaml` | 합성 식별 연구 (진짜 모터 ≠ 제어기 공칭값, LuGre/Stribeck 마찰, 센서 노이즈) |
| `dataset_dc.yaml` | 학습용 랜덤화 데이터셋 |
| `real_template.yaml` | **실제 데이터용 템플릿** (CSV 컬럼 매핑 / 단위 스케일) |

주요 선택지:

- `motor.type`: `dc` | `pmsm` (dq 모델, Ld≠Lq 가능, 극쌍수 p)
- `control.mode`: `voltage` | `current` | `torque`(힘/토크 제어) | `speed` | `position`
  — 캐스케이드 PI, 공칭 모델 + 대역폭(`bandwidth_hz`)으로 게인 자동 설정, `gains`로 직접 지정 가능.
  제어기는 `control.nominal`(데이터시트 값)을 쓰고 플랜트는 `motor.params`(실제 값)를 씀 → 모델 불일치 재현.
- `friction.model`: `none` | `viscous` | `coulomb` | `coulomb_viscous` | `stribeck` | `lugre`
- `load`: 부하 토크, 코깅 토크(`amplitude`, `periods`)
- `sensors`: 전류 노이즈/오프셋, 엔코더 분해능(CPR), 속도 LPF
- `reference`: `step`, `steps`, `sine`, `square`, `triangle`, `chirp`, `multisine`, `prbs`, `random_steps`
  (리스트로 쓰면 합산, 숫자 대신 `[lo, hi]` → 배치마다 랜덤)

## 모델

단위는 SI, 각도/속도는 기계각 기준입니다.

- **DC**: `L di/dt = v − R i − Ke ω`, `J dω/dt = Kt i − T_f − T_L`
- **PMSM (dq)**: `Ld did/dt = vd − R id + ωe Lq iq`, `Lq diq/dt = vq − R iq − ωe(Ld id + ψ)`,
  `Te = 1.5 p (ψ iq + (Ld − Lq) id iq)`, 전압 제한원 `|v| ≤ Vdc/√3`, dq 디커플링 + anti-windup
- **마찰**
  - Coulomb+viscous: `Tc sgn(ω) + B ω`
  - Stribeck: `[Tc + (Tst − Tc) e^{−|ω/ωs|^δ}] sgn(ω) + B ω`
  - LuGre: `dz/dt = ω − σ0|ω| z / g(ω)`, `T_f = σ0 z + σ1 dz/dt + B ω` (bristle 상태는 implicit 적분)
  - 불연속 모델은 **Karnopp stick-slip** 처리: `|T_drive| ≤ 정지마찰`이면 실제로 멈춰 있음
- **적분**: 제어 주기 `Ts`마다 센서 → 제어기 → ZOH 전압, 플랜트는 RK4로 `substeps`개 구간 적분.
  모든 연산이 배치(N개 파라미터 세트 동시)로 벡터화되어 있음.

## 식별 방법

1. **LS (equation error)**
   - 전기: `v = R i + L di/dt + Ke ω` (PMSM은 d/q 두 식을 쌓아 `R, Ld, Lq, ψ`).
     모든 신호에 같은 zero-phase LPF를 적용하고, ZOH를 고려해 구간 중점 기준으로 회귀.
   - 기계: `Kt i = J dω/dt + T_f(ω)`. 엔코더 각도와 전류에 **같은 LTI 필터**를 적용한 뒤 미분해서 식의 일관성을 유지.
     마찰 후보 모델마다 비선형 LS로 피팅하고 RMSE / R² / AIC / BIC로 비교.
2. **마찰 맵 (정속 시험)**: 속도 계단의 정상상태 평균 토크. 저속 Stribeck 영역(Tst, ωs)은 이 시험으로 얻는 것이 가장 정확합니다.
   토크 환산에는 1단계에서 식별한 Kt를 씁니다(공칭 Kt를 쓰면 마찰 크기가 Kt 오차만큼 틀어짐).
3. **OE (output error / simulation error)**: 측정 전압을 개루프 모델에 재생하고 전류·속도 오차를 최소화.
   P+1개 파라미터 세트를 한 번의 배치 시뮬레이션으로 돌려 Jacobian을 계산합니다.
   `oe.friction_model: lugre`로 설정하면 동적 마찰 σ0, σ1도 피팅합니다.
4. **검증**: 식별에 쓰지 않은 실험 데이터로 fit% (MATLAB `compare` 방식), RMSE, R²를 계산하고,
   가장 잘 맞는 추정값을 최종 결과로 선택합니다.
5. **학습 기반 (amortized)**: 랜덤 파라미터 + 랜덤 여기로 만든 데이터셋에서 1D-CNN이 `log(p/p_nom)`을 회귀.

## 현재 결과 (export/ 에 저장됨)

| 실험 | 결과 |
|---|---|
| `demo_dc` (진짜 마찰 = LuGre, 노이즈/16-bit 엔코더) | R, L, Kt, J, Tc, B, Tst, ωs 모두 ≤ 0.7 % 오차, 검증 fit 99.2 % (i) / 99.9 % (ω) — 정답 모델 재생과 동일 수준 |
| `demo_pmsm` (Ld≠Lq, Stribeck) | R, Ld, Lq, ψ, J, Tc, Tst, ωs, B 모두 ≤ 1.3 % 오차 |
| CNN (3000 샘플, 30 epoch) | 중앙 상대오차 R 2.4 %, L 2.8 %, Kt 1.3 %, J 4.3 % / Tc 25 %, B는 식별 불가(R² < 0) |

해석: 물리 기반 LS + 마찰 맵이 가장 정확합니다. CNN은 한 번 학습해 두면 즉시 추정할 수 있지만,
현재 데이터 규모에서는 효과가 작은 마찰 파라미터(B, Tc)를 잘 잡지 못합니다.
OE는 LS가 이미 정답에 가까울 때는 개선 폭이 작고, Stribeck 형상(Tst, ωs)처럼 여기가 약한 파라미터를 풀게 두면 과적합합니다.
그래서 기본 설정에서는 이 두 파라미터를 OE 대상에서 뺐습니다.

## 실제 데이터 쓰는 법

1. `config/real_template.yaml`을 복사해서 모터 타입, 데이터시트 값, CSV 컬럼 매핑, 단위(`scale`)를 채웁니다.
   - `v`는 **실제 인가 전압**(duty × Vbus, 포화 후 값)이어야 합니다.
   - 위치는 엔코더 각도(rad)를 쓰고, 속도는 코드에서 zero-phase 미분으로 만듭니다.
   - 명령과 인가 사이에 PWM 갱신 지연이 있으면 `identification.ls.delay`(샘플 수)를 지정합니다.
2. 권장 실험 3종: (a) 식별용 — 속도제어 + `random_steps` + `multisine` (반전 포함), (b) 검증용 — 다른 신호,
   (c) 마찰 맵 — 속도 계단 ±(0.5 … 최대속도), 로그 간격, 각 단 0.3 s 이상 유지(`ref` 컬럼 필요).
3. `identify --data a.csv --val b.csv --fmap c.csv` → `report.md` 확인
   → `validate`로 다른 데이터에서도 맞는지 확인.

## 디렉터리

```
run.py                 CLI 실행기 (src를 path에 추가)
src/motorsim/
  config.py            기본값, YAML 로드, 파라미터 랜덤화
  models/motors.py     DC / PMSM 플랜트, RK4
  models/friction.py   마찰 모델 + Karnopp/LuGre
  control.py           PI(anti-windup), 캐스케이드 / FOC
  sensors.py           전류 노이즈, 엔코더 양자화, 속도 추정
  signals.py           기준/여기 신호
  simulator.py         배치 폐루프 시뮬레이터
  ident/ls.py          LS 식별 (전기 + 기계)
  ident/friction_fit.py  마찰모델 피팅/비교, 정상상태 추출
  ident/output_error.py  OE 식별, 전압 재생
  validate.py, data.py 검증 지표, CSV 로딩/매핑
  dataset.py, learn.py 데이터셋 생성, CNN 학습/추론
  pipeline.py, cli.py  end-to-end 파이프라인, CLI
tests/                 pytest
```

## 다음 단계 후보

- 마찰 비대칭(Tc+/Tc−), 위치 의존 마찰, 코깅 토크 식별
- 인버터 데드타임 / 전압 오차 모델 (실제 PMSM 저속 식별에서 중요)
- LuGre σ0, σ1 식별용 presliding 실험 (작은 진폭 정현파 위치 제어)
- CNN 개선: 데이터 증강, 더 많은 샘플, LS 결과를 입력으로 쓰는 하이브리드, 차분가능 시뮬레이터
