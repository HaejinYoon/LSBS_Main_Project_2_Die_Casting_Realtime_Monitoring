import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score, confusion_matrix
)

# ------------------------------------------------------------
# 🔧 basic_fix 함수 재정의 (모델과 동일하게)
# ------------------------------------------------------------
def basic_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tryshot_signal" in df.columns:
        df["tryshot_signal"] = df["tryshot_signal"].apply(lambda x: 1 if str(x).upper() == "D" else 0)
    if {"speed_ratio", "low_section_speed", "high_section_speed"} <= set(df.columns):
        df.loc[df["speed_ratio"] == float("inf"), "speed_ratio"] = -1
        df.loc[df["speed_ratio"] == -float("inf"), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -2
    if "pressure_speed_ratio" in df.columns:
        df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1
    return df


# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
MODEL_PATH = "./models/fin_xgb_f20.pkl"
META_PATH = "./models/fin_xgb_meta_f20.json"
TEST_PATH = "./data/fin_test_kf_fixed.csv"
TARGET = "passorfail"
DROP_COLS = ["team", "real_time"]


# ------------------------------------------------------------
# 2️⃣ 모델 & 메타 불러오기
# ------------------------------------------------------------
print("🔍 모델 및 메타 정보 불러오는 중...")
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

best_threshold = meta.get("best_threshold", 0.5)
print(f"✅ 모델 로드 완료 (threshold={best_threshold:.3f})")


# ------------------------------------------------------------
# 3️⃣ 테스트 데이터 로드
# ------------------------------------------------------------
def read_csv_safely(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

test = read_csv_safely(TEST_PATH)
print(f"✅ 테스트셋 로드 완료: {test.shape}")


# ------------------------------------------------------------
# 4️⃣ 예측 수행
# ------------------------------------------------------------
start = time.time()
X_test = test.drop(columns=[TARGET] + [c for c in DROP_COLS if c in test.columns], errors="ignore")
y_true = test[TARGET].values
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= best_threshold).astype(int)
elapsed = time.time() - start

# ------------------------------------------------------------
# 5️⃣ 전체 평가 출력
# ------------------------------------------------------------
print("\n📊 [전체 테스트셋 평가]")
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F0.5     : {fbeta_score(y_true, y_pred, beta=0.5):.4f}")
print(f"F1       : {fbeta_score(y_true, y_pred, beta=1.0):.4f}")
print(f"F2       : {fbeta_score(y_true, y_pred, beta=2.0):.4f}")
print(f"⏱ 예측 소요 시간: {elapsed:.2f}초")

# ------------------------------------------------------------
# 6️⃣ 몰드코드별 성능 분석
# ------------------------------------------------------------
if "mold_code" not in test.columns:
    print("\n⚠️ mold_code 컬럼이 없습니다.")
else:
    results = []
    for mold, group in test.groupby("mold_code"):
        y_t = group[TARGET].values
        X_t = group.drop(columns=[TARGET] + [c for c in DROP_COLS if c in group.columns], errors="ignore")
        y_p = model.predict_proba(X_t)[:, 1]
        y_p = (y_p >= best_threshold).astype(int)

        acc = accuracy_score(y_t, y_p)
        rec = recall_score(y_t, y_p, zero_division=0)
        prec = precision_score(y_t, y_p, zero_division=0)
        f1 = fbeta_score(y_t, y_p, beta=1.0)
        f2 = fbeta_score(y_t, y_p, beta=2.0)
        f05 = fbeta_score(y_t, y_p, beta=0.5)

        results.append({
            "mold_code": mold,
            "n_samples": len(group),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "F0.5": f05,
            "F1": f1,
            "F2": f2
        })

    df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\n📦 몰드코드별 성능 요약:")
    print(df_results.to_string(index=False, float_format="%.4f"))


# ------------------------------------------------------------
# 7️⃣ 성능 결과 파일 저장
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import joblib
import json
import time
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    fbeta_score
)

# ------------------------------------------------------------
# 🔧 기본 전처리 함수
# ------------------------------------------------------------
def basic_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "tryshot_signal" in df.columns:
        df["tryshot_signal"] = df["tryshot_signal"].apply(lambda x: 1 if str(x).upper() == "D" else 0)
    if {"speed_ratio", "low_section_speed", "high_section_speed"} <= set(df.columns):
        df.loc[df["speed_ratio"] == float("inf"), "speed_ratio"] = -1
        df.loc[df["speed_ratio"] == -float("inf"), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -2
    if "pressure_speed_ratio" in df.columns:
        df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1
    return df


# ------------------------------------------------------------
# 1️⃣ 설정
# ------------------------------------------------------------
MODEL_PATH = "./models/fin_xgb_f20.pkl"
META_PATH = "./models/fin_xgb_meta_f20.json"
TRAIN_PATH = "./data/fin_train.csv"
TEST_PATH = "./data/fin_test_kf_fixed.csv"
TARGET = "passorfail"
DROP_COLS = ["team", "real_time"]

# ------------------------------------------------------------
# 2️⃣ 모델 불러오기
# ------------------------------------------------------------
print("🔍 모델 로드 중...")
model = joblib.load(MODEL_PATH)
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)
best_threshold = meta.get("best_threshold", 0.5)
print(f"✅ 모델 로드 완료 (Threshold={best_threshold:.3f})")

# ------------------------------------------------------------
# 3️⃣ 데이터 로드
# ------------------------------------------------------------
def read_csv_safely(path):
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

train = read_csv_safely(TRAIN_PATH)
test = read_csv_safely(TEST_PATH)
print(f"✅ 데이터 로드 완료 | train: {train.shape} / test: {test.shape}")

# ------------------------------------------------------------
# 4️⃣ 예측 및 시간 측정 함수
# ------------------------------------------------------------
def evaluate_dataset(df, label="dataset"):
    df = basic_fix(df)
    X = df.drop(columns=[TARGET] + [c for c in DROP_COLS if c in df.columns], errors="ignore")
    y = df[TARGET].values

    # 예측 시간 측정 여러 번 수행 (평균 추정)
    times = []
    for _ in range(30):  # 30회 평균
        start = time.time()
        _ = model.predict_proba(X)[:, 1]
        times.append(time.time() - start)

    elapsed_mean = np.mean(times)
    elapsed_std = np.std(times)
    ucl = elapsed_mean + 3 * elapsed_std
    lcl = max(elapsed_mean - 3 * elapsed_std, 0)

    # 단일 예측 (성능 계산)
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= best_threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f05 = fbeta_score(y, y_pred, beta=0.5)
    f1 = fbeta_score(y, y_pred, beta=1.0)
    f2 = fbeta_score(y, y_pred, beta=2.0)

    return {
        "label": label,
        "n_samples": len(df),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "F0.5": f05,
        "F1": f1,
        "F2": f2,
        "elapsed_mean_sec": elapsed_mean,
        "elapsed_std_sec": elapsed_std,
        "UCL_3sigma": ucl,
        "LCL_3sigma": lcl
    }

# ------------------------------------------------------------
# 5️⃣ 학습/테스트 성능 계산
# ------------------------------------------------------------
print("\n📊 성능 계산 중...")
metrics_train = evaluate_dataset(train, label="train")
metrics_test = evaluate_dataset(test, label="test")

# ------------------------------------------------------------
# 6️⃣ 결과 요약 출력
# ------------------------------------------------------------
print("\n📦 [Train Dataset]")
for k, v in metrics_train.items():
    if k not in ["label"]:
        print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

print("\n📦 [Test Dataset]")
for k, v in metrics_test.items():
    if k not in ["label"]:
        print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# ------------------------------------------------------------
# 7️⃣ 파일로 저장
# ------------------------------------------------------------
os.makedirs("./results", exist_ok=True)

output = {
    "model_info": {
        "path": MODEL_PATH,
        "threshold": best_threshold,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    },
    "train_metrics": metrics_train,
    "test_metrics": metrics_test
}

output_path = "./results/model_performance_full.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print(f"\n💾 성능 결과 저장 완료 → {output_path}")
