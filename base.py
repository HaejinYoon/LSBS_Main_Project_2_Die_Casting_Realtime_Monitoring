#######################################################
# 몰튼 탬프 끝

import pandas as pd
import plotly.graph_objects as go
from pykalman import KalmanFilter

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# tryshot_signal D → 1, 아니면 0
train['tryshot_signal_D'] = train['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# 결측값 여부
train['is_nan'] = train['molten_temp'].isna()

# Kalman Filter 적용
kf = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=train['molten_temp'].dropna().iloc[0],
    observation_covariance=1,
    transition_covariance=0.1
)
state_means, _ = kf.filter(train['molten_temp'].fillna(method='ffill').values)
train['molten_temp_kf'] = state_means.flatten()

# 스무싱
train['molten_temp_kf_smooth'] = train['molten_temp_kf'].rolling(window=10, min_periods=1).mean()

# 시각화
fig = go.Figure()

# KF 보정 선 (진한 빨강)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_temp_kf'],
    mode='lines',
    name='Kalman Filtered',
    line=dict(color='#b30000', width=2.5)
))

# 스무싱 선 (진한 검정)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_temp_kf_smooth'],
    mode='lines',
    name='Smoothed (Rolling 10)',
    line=dict(color='black', width=2.5)
))

# passorfail=1 점 (연한 빨강)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==1, 'real_time'],
    y=train.loc[train['passorfail']==1, 'molten_temp_kf'],
    mode='markers',
    name='passorfail=1',
    marker=dict(color='#ff9999', size=6, opacity=0.7)
))

# passorfail=0 점 (연한 파랑)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==0, 'real_time'],
    y=train.loc[train['passorfail']==0, 'molten_temp_kf'],
    mode='markers',
    name='passorfail=0',
    marker=dict(color='#99ccff', size=6, opacity=0.7)
))

# 결측값 위치 X 마커 (검정)
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan'], 'real_time'],
    y=train.loc[train['is_nan'], 'molten_temp_kf'],
    mode='markers',
    name='NaN',
    marker=dict(color='black', size=6, symbol='x', opacity=0.9)
))

# tryshot_signal D 점 (초록)
fig.add_trace(go.Scatter(
    x=train.loc[train['tryshot_signal_D']==1, 'real_time'],
    y=train.loc[train['tryshot_signal_D']==1, 'molten_temp_kf'],
    mode='markers',
    name='tryshot_signal=D',
    marker=dict(color='green', size=5, opacity=0.7)
))

fig.update_layout(
    title='Molten Temp: KF + Smoothed with passorfail and NaN Highlighted',
    xaxis_title='Real Time',
    yaxis_title='Molten Temperature',
    template='plotly_white',
    height=600
)

fig.show()




##################################################################
train.isna().sum()

import pandas as pd
import plotly.graph_objects as go

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# 결측값 여부
train['is_nan'] = train['low_section_speed'].isna()

# 시계열 라인
fig = go.Figure()

# 원본 시계열 선
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['low_section_speed'],
    mode='lines+markers',
    name='low_section_speed',
    line=dict(color='#1f77b4', width=2),
    marker=dict(size=4, opacity=0.7)
))

# 결측값 X 표시
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan'], 'real_time'],
    y=[0]*train['is_nan'].sum(),  # 시각적으로 0 위치
    mode='markers',
    name='Missing Value',
    marker=dict(color='red', size=6, symbol='x', opacity=0.9)
))

fig.update_layout(
    title='Low Section Speed over Time',
    xaxis_title='Real Time',
    yaxis_title='Low Section Speed',
    template='plotly_white',
    height=500
)

fig.show()
train.isna().sum()


#################################################################################

import pandas as pd
import plotly.graph_objects as go
from pykalman import KalmanFilter

# 데이터 불러오기
train = pd.read_csv('./data/train.csv')

# tryshot_signal D → 1, 아니면 0
train['tryshot_signal_D'] = train['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)

# real_time → datetime
train['real_time'] = pd.to_datetime(train['real_time'])

# molten_volume 결측 여부
train['is_nan_volume'] = train['molten_volume'].isna()

# Kalman Filter 적용
kf_volume = KalmanFilter(
    transition_matrices=[1],
    observation_matrices=[1],
    initial_state_mean=train['molten_volume'].dropna().iloc[0],
    observation_covariance=1,
    transition_covariance=0.1
)
state_means_vol, _ = kf_volume.filter(train['molten_volume'].fillna(method='ffill').values)
train['molten_volume_kf'] = state_means_vol.flatten()

# 스무싱
train['molten_volume_kf_smooth'] = train['molten_volume_kf'].rolling(window=10, min_periods=1).mean()

# 시각화
fig = go.Figure()

# KF 보정 선 (진한 빨강)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_volume_kf'],
    mode='lines',
    name='Kalman Filtered',
    line=dict(color='#b30000', width=2.5)
))

# 스무싱 선 (진한 검정)
fig.add_trace(go.Scatter(
    x=train['real_time'],
    y=train['molten_volume_kf_smooth'],
    mode='lines',
    name='Smoothed (Rolling 10)',
    line=dict(color='black', width=2.5)
))

# passorfail=1 점 (연한 빨강)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==1, 'real_time'],
    y=train.loc[train['passorfail']==1, 'molten_volume_kf'],
    mode='markers',
    name='passorfail=1',
    marker=dict(color='#ff9999', size=6, opacity=0.7)
))

# passorfail=0 점 (연한 파랑)
fig.add_trace(go.Scatter(
    x=train.loc[train['passorfail']==0, 'real_time'],
    y=train.loc[train['passorfail']==0, 'molten_volume_kf'],
    mode='markers',
    name='passorfail=0',
    marker=dict(color='#99ccff', size=6, opacity=0.7)
))

# 결측값 위치 X 마커 (검정)
fig.add_trace(go.Scatter(
    x=train.loc[train['is_nan_volume'], 'real_time'],
    y=train.loc[train['is_nan_volume'], 'molten_volume_kf'],
    mode='markers',
    name='NaN',
    marker=dict(color='black', size=6, symbol='x', opacity=0.9)
))

# tryshot_signal D 점 (초록)
fig.add_trace(go.Scatter(
    x=train.loc[train['tryshot_signal_D']==1, 'real_time'],
    y=train.loc[train['tryshot_signal_D']==1, 'molten_volume_kf'],
    mode='markers',
    name='tryshot_signal=D',
    marker=dict(color='green', size=5, opacity=0.7)
))

fig.update_layout(
    title='Molten Volume: KF + Smoothed with passorfail and NaN Highlighted',
    xaxis_title='Real Time',
    yaxis_title='Molten Volume',
    template='plotly_white',
    height=600
)

fig.show()

#################################################################

import numpy as np
import pandas as pd
from pykalman import KalmanFilter

# 1️⃣ 데이터 불러오기
df = pd.read_csv('./data/train.csv')

# 2️⃣ KF로 결측치 채울 칼럼
kf_columns = ['molten_temp', 'molten_volume', 'upper_mold_temp3']

# 3️⃣ 각 칼럼마다 KF 적용 후 바로 기존 칼럼에 결측치 채우기
for col in kf_columns:
    if col in df.columns:
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=df[col].dropna().iloc[0],
            observation_covariance=1,
            transition_covariance=0.1
        )
        state_means, _ = kf.filter(df[col].fillna(method='ffill').values)
        mask = df[col].isna()
        df.loc[mask, col] = state_means.flatten()[mask]

# 4️⃣ 삭제할 칼럼만 제거
if 'lower_mold_temp3' in df.columns:
    df = df.drop(columns=['lower_mold_temp3'])


df.isna().sum()

df.dropna(subset= 'working', inplace=True)

# 5️⃣ CSV 저장
df.to_csv('./data/train_res.csv', index=False)

print("✅ KF로 결측치 바로 채우고, lower_mold_temp3만 제거 완료, train_res.csv 저장됨")


####################################################################################################################
# 모델 구현 및 저장


# 파생 변수 만들기 파트 주야 shift  global_count  monthly_count  speed_ratio  pressure_speed_ratio real_time

#######################################################################################################################
## 인코딩

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

train = pd.read_csv('./data/fin_train.csv')
test = pd.read_csv('./data/fin_test_kf.csv')

train

# 예시 컬럼
label_cols = ["working", "emergency_stop", "tryshot_signal", "shift"]
onehot_cols = ["mold_code", "heating_furnace"]

# -------------------------------
# 1️⃣ tryshot_signal 결측치 처리
# -------------------------------
# 최빈값으로 채우기 (문자형 'D' 포함 가능)
# tryshot_signal 이진화: D=1, 그 외=0
df['tryshot_signal'] = df['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)


# -------------------------------
# 2️⃣ Label 인코딩 대체 (OrdinalEncoder)
# -------------------------------
# handle_unknown 옵션으로 새로운 값도 허용
ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df[label_cols] = ordinal.fit_transform(df[label_cols])

# -------------------------------
# 3️⃣ One-hot 인코딩
# -------------------------------
# handle_unknown='ignore'로 새로운 값 무시
onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = onehot.fit_transform(df[onehot_cols])

# 컬럼명 복원
encoded_cols = onehot.get_feature_names_out(onehot_cols)
encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

# 기존 df와 결합
df = pd.concat([df.drop(columns=onehot_cols), encoded_df], axis=1)

# -------------------------------
# 4️⃣ speed_ratio 처리
# -------------------------------
# inf 값 처리
df.loc[df["speed_ratio"] == float("inf"), "speed_ratio"] = -1
# low_section_speed, high_section_speed 둘 다 0일 경우 처리
df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -2

# pressure_speed_ratio가 inf 또는 -inf이면 -1로 대체
df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1

df.isna().sum()

'''
team 이랑 real_time 은 뺴고 학습을 돌릴예정 파이프라인으로 해야하고 threshold 도 파라미터를 정해서 골라야하고 SMOTE NC로 해야할거같으면 SMOTE NC로 아닌거 같으면 SMOTE 둘중하나로 해주고 얼마나 늘려야하는지
XG 부스트도 하이퍼 파라미터 튜닝을 모든것을 다해주고 이걸 옵튜나에 베이지안 정리를 해서 해줬으면 좋겠어 그리고 cv도 튜닝다해줘 머가 좋은지 그리고 타겟은 passorfail 이고 test 데이터에도 이게 있어 둘다 뺴놓고 해야하고 test는 일단 사용하지말고
train데이터만 나눈다음 확인해주고 test데이터는 나중에 진행할 예정이니까 건들지 말아주고 test 데이터에 바로 사용할수 있게 피클로 저장해서 사용할수 있게 파이프라인을 만들어줘

'''



##################################################################################


# fin_xgb_pipeline_f2_optuna_full.py
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
from tqdm import tqdm
from xgboost import XGBClassifier
import joblib

# -------------------------------
# 0️⃣ 데이터 로드
# -------------------------------
train = pd.read_csv('./data/fin_train_fixed.csv')
test = pd.read_csv('./data/fin_test_kf_fixed.csv')  # 사용하지 않음

TARGET = 'passorfail'
DROP_COLS = ['team', 'real_time']

label_cols = ["working", "emergency_stop", "tryshot_signal", "shift"]
onehot_cols = ["mold_code", "heating_furnace"]

# -------------------------------
# 1️⃣ 규칙 기반 전처리 함수
# -------------------------------
def basic_fix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'tryshot_signal' in df.columns:
        df['tryshot_signal'] = df['tryshot_signal'].apply(lambda x: 1 if x == 'D' else 0)

    if {'speed_ratio', 'low_section_speed', 'high_section_speed'} <= set(df.columns):
        df.loc[df["speed_ratio"] == float("inf"), "speed_ratio"] = -1
        df.loc[df["speed_ratio"] == -float("inf"), "speed_ratio"] = -1
        df.loc[(df["low_section_speed"] == 0) & (df["high_section_speed"] == 0), "speed_ratio"] = -2

    if 'pressure_speed_ratio' in df.columns:
        df.loc[np.isinf(df["pressure_speed_ratio"]), "pressure_speed_ratio"] = -1

    return df

feature_fixer = FunctionTransformer(basic_fix, validate=False)

# -------------------------------
# 2️⃣ 데이터 분리
# -------------------------------
y = train[TARGET].values
X_raw = train.drop(columns=[TARGET] + [c for c in DROP_COLS if c in train.columns])

label_cols_eff = [c for c in label_cols if c in X_raw.columns]
onehot_cols_eff = [c for c in onehot_cols if c in X_raw.columns]
num_cols_eff = [c for c in X_raw.columns if c not in set(label_cols_eff + onehot_cols_eff)]

# -------------------------------
# 3️⃣ ColumnTransformer 설정
# -------------------------------
ct = ColumnTransformer(
    transformers=[
        ("label_enc", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), label_cols_eff),

        ("onehot", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), onehot_cols_eff),

        ("num", SimpleImputer(strategy="median"), num_cols_eff),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)

# -------------------------------
# 4️⃣ XGBoost 생성 함수
# -------------------------------
def make_clf(params: dict) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 6),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        min_child_weight=params.get("min_child_weight", 1.0),
        gamma=params.get("gamma", 0.0),
        scale_pos_weight=params.get("scale_pos_weight", 1.0),
        reg_lambda=params.get("reg_lambda", 1.0),
        reg_alpha=params.get("reg_alpha", 0.0),
        max_delta_step=params.get("max_delta_step", 0.0),
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
        use_label_encoder=False
    )

# -------------------------------
# 5️⃣ Optuna Objective (F2-score)
# -------------------------------
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 10.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 0.5, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "max_delta_step": trial.suggest_float("max_delta_step", 0.0, 10.0),
    }

    threshold = trial.suggest_float("threshold", 0.05, 0.95)

    clf = make_clf(params)

    # 파이프라인에 XGB 포함
    pipe = ImbPipeline(steps=[
        ("fix", feature_fixer),
        ("ct", ct),
        ("smote", SMOTE(random_state=42)),
        ("clf", clf)
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f2_scores = []

    try:
        for tr_idx, val_idx in skf.split(X_raw, y):
            X_tr, X_val = X_raw.iloc[tr_idx], X_raw.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            # 전체 파이프라인 학습
            pipe.fit(X_tr, y_tr)

            # validation 예측
            y_prob = pipe.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= threshold).astype(int)
            f2 = fbeta_score(y_val, y_pred, beta=2)
            f2_scores.append(f2)

        mean_f2 = np.mean(f2_scores)
        print(f"Trial {trial.number} | F2 = {mean_f2:.4f} | th={threshold:.2f}")
        return mean_f2

    except Exception as e:
        print(f"⚠️ Trial {trial.number} failed: {e}")
        return None


# -------------------------------
# 6️⃣ Optuna 실행 (진행상황 표시)
# -------------------------------
study = optuna.create_study(direction="maximize", study_name="xgb_f2_opt")
for _ in tqdm(range(50), desc="Optuna Trials Progress"):
    study.optimize(objective, n_trials=1, catch=(Exception,))

best_params = study.best_params
best_value = study.best_value
best_threshold = best_params.pop("threshold")

print("\n✅ Best F2:", best_value)
print("✅ Best Threshold:", best_threshold)
print("✅ Best Params:", best_params)

# -------------------------------
# 7️⃣ 최적 모델 학습
# -------------------------------
final_model = ImbPipeline(steps=[
    ("fix", feature_fixer),
    ("ct", ct),
    ("smote", SMOTE(random_state=42)),
    ("clf", make_clf(best_params))
])

final_model.fit(X_raw, y)
print("✅ Final model trained successfully")

# -------------------------------
# 8️⃣ 모델 및 메타데이터 저장
# -------------------------------
os.makedirs("./models", exist_ok=True)
joblib.dump(final_model, "./models/fin_xgb_pipeline.pkl")

meta = {
    "best_f2": best_value,
    "best_threshold": best_threshold,
    "best_params": best_params,
}
with open("./models/fin_xgb_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ 모델 및 파라미터 저장 완료.")



#################=======================================================================

import pandas as pd
import joblib
import json
from sklearn.metrics import (
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ 모델 및 메타정보 불러오기
# -------------------------------
model_path = './models/fin_xgb_pipeline.pkl'
meta_path = './models/fin_xgb_meta.json'

model = joblib.load(model_path)
with open(meta_path, 'r') as f:
    meta = json.load(f)

threshold = meta["best_threshold"]
print(f"✅ 모델 로드 완료 | threshold = {threshold:.3f}")

# -------------------------------
# 2️⃣ 데이터 불러오기
# -------------------------------
train = pd.read_csv('./data/fin_train.csv')
test = pd.read_csv('./data/fin_test_kf.csv')

X_train = train.drop(columns=['passorfail'])
y_train = train['passorfail'].values

if 'passorfail' in test.columns:
    X_test = test.drop(columns=['passorfail'])
    y_test = test['passorfail'].values
else:
    X_test = test.copy()
    y_test = None

# -------------------------------
# 3️⃣ 예측 함수 정의
# -------------------------------
def evaluate_model(model, X, y, threshold, dataset_name="data"):
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f2 = fbeta_score(y, y_pred, beta=2)
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)

    print(f"\n📊 [{dataset_name}] 성능 요약")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F2-score  : {f2:.4f}")
    print("\n혼동행렬:")
    print(cm)
    print("\n상세 리포트:")
    print(classification_report(y, y_pred, digits=4))

    # 혼동행렬 시각화
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    return {"acc": acc, "precision": precision, "recall": recall, "f2": f2}


# -------------------------------
# 4️⃣ Train/Test 평가
# -------------------------------
train_result = evaluate_model(model, X_train, y_train, threshold, "Train")

if y_test is not None:
    test_result = evaluate_model(model, X_test, y_test, threshold, "Test")
else:
    print("\n⚠️ Test 데이터에 passorfail 컬럼이 없어 평가 생략됨.")
    test_result = None

# -------------------------------
# 5️⃣ 비교 요약 출력
# -------------------------------
if test_result is not None:
    print("\n📈 Train vs Test 성능 비교")
    compare_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F2-score"],
        "Train": [
            train_result["acc"],
            train_result["precision"],
            train_result["recall"],
            train_result["f2"],
        ],
        "Test": [
            test_result["acc"],
            test_result["precision"],
            test_result["recall"],
            test_result["f2"],
        ],
    })
    print(compare_df.round(4))

    # 성능 비교 시각화
    compare_df_melted = compare_df.melt(id_vars="Metric", var_name="Dataset", value_name="Score")
    plt.figure(figsize=(6, 4))
    sns.barplot(data=compare_df_melted, x="Metric", y="Score", hue="Dataset")
    plt.title("Train vs Test Performance Comparison")
    plt.ylim(0, 1)
    plt.show()




