# 🏭 주조 공정 실시간 품질 모니터링 및 불량 예측 대시보드  
**LSBS Main Project 2 – Die Casting Realtime Monitoring**

---

## 📘 1. 프로젝트 개요 (Introduction)

본 프로젝트는 **주조(Die Casting) 공정의 실시간 데이터를 모니터링하고, 품질 이상 여부를 예측**하여 불량을 조기에 감지하기 위한 시스템입니다.  
센서 스트리밍 데이터를 기반으로 **주·야간 조별 생산 달성률, 불량 경향, 공정별 주요 변수 모니터링, 품질 예측 및 개선 분석**을 통합한 **Python Shiny 기반 실시간 대시보드**를 개발하였습니다.

---

## 📊 2. 데이터 설명 (Dataset Overview)

| 항목 | 설명 |
|------|------|
| 데이터 출처 | 실제 주조 설비의 센서 로그 및 품질 검사 이력 |
| 주요 파일 | `./data/train.csv`, `./data/test.csv` |
| 주요 변수 | `molten_temp`, `sleeve_temperature`, `EMS_operation_time`, `low_section_speed`, `high_section_speed`, `mold_code`, `passorfail` 등 |
| 표본 수 | 약 수천 건 (streaming 방식으로 row 단위 입력) |
| 특이사항 | 일부 결측치 및 이상치 존재 → `basic_fix()` 함수를 통한 전처리 수행 |

데이터는 주조 공정의 주요 단계를 반영하며, **실시간 센서 데이터 스트리밍 구조**를 모사하여 2초 간격으로 새로운 행이 주입됩니다.

---

## 🧠 3. 모델 구조 및 학습 (Modeling & Training)

- **모델 종류:** XGBoost 기반 품질 예측 모델 (`fin_xgb_pipeline.pkl`)  
- **특징 공학:**  
  - 속도·압력 비율, EMS 작동 시간, 주입 구간 온도차 등의 비율 및 조합 변수 생성  
  - `FeatureEngineer` 및 `DatetimeFeatureExtractor` 클래스를 통한 파생 변수 자동 생성  
- **평가 지표:** Accuracy, Precision, Recall, F-beta Score 등  
- **결과:** 약 90% 이상의 불량 탐지 정확도 달성  

모델은 `joblib`으로 직렬화되어 불러오며, **실시간 입력 데이터에 대해 즉시 예측 수행** 후 대시보드에서 시각적으로 표시됩니다.

---

## 🧩 4. 시스템 구조 및 주요 기능 (System Architecture & Features)

### 🔸 전체 구조
```
app.py
 ├─ RealTimeStreamer: 스트리밍 데이터 공급 클래스
 ├─ Shiny Reactive 시스템: 실시간 상태 업데이트
 ├─ Field Monitoring 탭
 │   ├─ 실시간 공정 상태 카드
 │   ├─ 주·야간 달성률 계산
 │   ├─ 센서 변수 SVG 시각화
 ├─ Quality Monitoring 탭
 │   ├─ 실시간 품질 예측 및 알림
 │   ├─ 예측 이력 모달 및 불량 원인 분석
 │   └─ 모델 모니터링 그래프
 ├─ Analysis 탭
 │   └─ 변수별 영향도 및 개선안 시각화
 ├─ Login / Logout / Navigation
 │   └─ 로그인 검증, 페이지 전환 관리
 └─ CSS, 이미지, 모델, 데이터 연동 모듈
```

### 🔹 주요 기능 요약

| 기능 | 설명 |
|------|------|
| **실시간 스트리밍 제어** | 시작 / 일시정지 / 초기화 / 배속 제어 (x1~x8) |
| **공정 상태 카드** | 조별 달성률, 품질 판정 결과, 실시간 상태 표시 |
| **품질 예측 알림** | 불량 발생 시 알림창에 즉시 표시 및 개별 삭제 기능 |
| **SVG 기반 공정 시각화** | 주조 공정 각 단계(용융, 충진, 냉각 등) 센서값 실시간 표시 |
| **모달 분석창** | 최근 불량 사례의 변수값 및 통계 비교 제공 |
| **데이터 필터링** | 3σ 기준 및 불량 필터 적용 기능 |
| **로그인/권한 관리** | 관리자 계정 기반 접근 제어 (`admin / 1234`) |

---

## 🖥️ 5. 대시보드 구성 (Dashboard Description)

| 탭 | 주요 내용 |
|-----|------------|
| **현장 모니터링 (Field Monitoring)** | 실시간 센서 데이터 스트리밍, 조별 달성률, 주조 단계별 변수 시각화 |
| **품질 모니터링 (Quality Monitoring)** | 실시간 예측 결과, 이상치 알림, 원인 분석 모달 |
| **예측 및 개선 (Prediction & Improvement)** | 변수 영향도 분석, 모델 예측 결과 비교, 개선 시뮬레이션 |
| **로그인 / 메뉴 / 로그아웃 UI** | 사용자 접근 제어 및 페이지 전환 관리 |

모든 탭은 **Shiny의 reactive 구조**로 연결되어 있으며, 실시간 반응형 데이터 흐름을 통해 대시보드의 모든 구성요소가 자동 업데이트됩니다.

---

## 📈 6. 성과 및 시사점 (Results & Implications)

- 실시간 스트리밍 기반 품질 예측 시스템을 Python Shiny로 구현하여,  
  **제조 현장에서 즉시 활용 가능한 수준의 시각화 및 반응성 확보**
- 불량 탐지 정확도 향상 및 조기 경보 시스템 구현  
- 주·야간 교대별 실적 관리 및 개선 포인트 시각화를 통한 **품질 관리 효율 극대화**
- 전통적인 SCADA 기반 공정관리 시스템 대비, **경량·저비용 웹 대시보드 솔루션**으로 대체 가능성 검증

---

## ⚙️ 7. 실행 방법 (How to Run)

### 1️⃣ 환경 구성
```bash
conda create -n shiny-env python=3.10
conda activate shiny-env
pip install -r requirements.txt
```

### 2️⃣ 프로젝트 구조
```
📦 LSBS_Main_Project_2_Die_Casting_Realtime_Monitoring
 ┣ 📂 data
 ┃ ┣ train.csv
 ┃ ┗ test.csv
 ┣ 📂 models
 ┃ ┗ fin_xgb_f20.pkl
 ┣ 📂 www
 ┃ ┣ diagram.svg / die-casting.gif / css 등 시각화 리소스
 ┣ 📜 app.py
 ┣ 📜 requirements.txt
 ┗ 📜 README.md
```

### 3️⃣ 실행
```bash
shiny run --reload app.py
```
또는
```bash
python app.py
```

브라우저에서 다음 주소로 접속합니다:
```
http://127.0.0.1:8000
```

---

## 🧰 8. 기술 스택 (Tech Stack)

| 구분 | 기술 |
|------|------|
| **Frontend** | Python Shiny UI, Bootstrap, FontAwesome, Custom CSS (`common.css`) |
| **Backend** | Python Shiny Reactive, pandas, numpy, joblib, XGBoost |
| **Visualization** | Plotly, SVG, Shiny UI Components |
| **Deployment** | shinyapps.io / Localhost |
| **Env** | Conda (Python 3.10 기반) |

---

## 📎 9. 참고사항 (Notes)

- `app.py`는 **단일 구조**로 작성되어 있으며, 각 탭과 reactive 흐름이 내부에서 관리됨  
- `RealTimeStreamer` 클래스를 통해 `test.csv`의 행이 2초 간격으로 스트리밍됨  
- **알림 시스템(`push_alert`)**은 불량 발생 시 실시간 경보를 표시하며, 개별 삭제 기능 제공  
- **모델 모니터링 그래프**는 스트리밍 데이터를 기반으로 주기적 갱신  
- 실제 현장 센서 연동 시 MQTT, OPC-UA, Kafka 등으로 확장 가능  
- 모델 버전 호환성을 위해 **scikit-learn 1.5.1 이상** 사용 권장
- 배포 주소 : https://spicywinter.shinyapps.io/main_proj_2_monitoring/

---

## ✨ 제작 및 기여 (Contributors)

- **개발 / 기획:** 김동균, 안태경, 윤해진, 이유진, 진소현
- **프로젝트명:** LSBS Main Project 2 – Die Casting Realtime Monitoring  
- **기간:** 2024.10.13. ~ 2025.10.22.  
- **주제:** 실시간 주조 공정 품질 예측 및 개선 대시보드  

---
 
> 코드, 모델 및 데이터 파일은 내부 프로젝트용으로 관리됩니다.*
