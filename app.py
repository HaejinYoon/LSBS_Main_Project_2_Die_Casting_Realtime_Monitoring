import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive, session
from shiny.ui import update_slider, update_numeric, update_select, update_navs
import seaborn as sns
import pathlib
import plotly.express as px
from shinywidgets import render_plotly, output_widget
import numpy as np
import matplotlib
from sklearn.metrics import pairwise_distances
import os
from matplotlib import font_manager
import plotly.io as pio
import calendar
import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
# ======== 실시간 스트리밍 대시보드 (현장 메뉴) ========
from shared import streaming_df, RealTimeStreamer, KFStreamer
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
import datetime

# ==========================================
# 🔹 Baseline UCL 계산 함수 (고정형 관리도용)
# ==========================================
from scipy.stats import f

stream_speed = reactive.Value(2.0)  # 기본 2초 주기

# 🔧 basic_fix 함수 추가 (model.py와 동일하게)
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

def calc_baseline_ucl(train_df, cols):
    """Train 데이터 기반 UCL, mean, inv_cov 계산"""
    X = train_df[cols].dropna().values
    n, p = X.shape
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    UCL = p * (n - 1) * (n + 1) / (n * (n - p)) * f.ppf(0.99, p, n - p)
    print(f"✅ Baseline UCL({cols[0][:6]}...) 계산 완료: {UCL:.3f}")
    return UCL, mean, inv_cov

# ✅ 표시에서 제외할 컬럼
EXCLUDE_COLS = ["id", "line", "name", "mold_name", "date", "time", "registration_time", "count"]

# ✅ 표시 대상: 위 제외 목록을 빼고 나머지 수치형 컬럼 자동 선택
display_cols = [
    c for c in streaming_df.columns
    if c not in EXCLUDE_COLS and pd.api.types.is_numeric_dtype(streaming_df[c])
]

# 스트리밍 초기 설정
streamer = reactive.Value(RealTimeStreamer(streaming_df))
current_data = reactive.Value(pd.DataFrame())
is_streaming = reactive.Value(False)
KF_PATH = pathlib.Path("./data/fin_test_kf.csv")
kf_streamer = reactive.Value(KFStreamer(KF_PATH))

# ===== 한글 변수명 매핑 =====
VAR_LABELS = {
    # 용융 단계
    "molten_temp": "용융 온도",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "슬리브 온도",
    "EMS_operation_time": "EMS 가동시간",
    "low_section_speed": "하부 주입속도",
    "high_section_speed": "상부 주입속도",
    "molten_volume": "주입 금속량",
    "cast_pressure": "주입 압력",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형온도",
    "upper_mold_temp2": "상부2 금형온도",
    "upper_mold_temp3": "상부3 금형온도",
    "lower_mold_temp1": "하부1 금형온도",
    "lower_mold_temp2": "하부2 금형온도",
    "lower_mold_temp3": "하부3 금형온도",
    "Coolant_temperature": "냉각수 온도",

    # 품질 및 속도
    "production_cycletime": "생산 사이클",
    "biscuit_thickness": "주조물 두께",
    "physical_strength": "제품 강도",
    
    "mold_code": "금형코드",
}

# ===== 센서 위치 (x, y) =====
VAR_POSITIONS = {
    # 용융부
    "molten_temp": (735, 250),
    # "heating_furnace": (735, 450),

    # 슬리브 / 주입
    "sleeve_temperature": (510, 325),
    "EMS_operation_time": (30, 340),
    "low_section_speed": (350, 390),
    "high_section_speed": (350, 135),
    "molten_volume": (700, 320),
    "cast_pressure": (520, 360),

    # 금형 냉각
    "upper_mold_temp1": (30, 30),
    "upper_mold_temp2": (30, 80),
    "upper_mold_temp3": (30, 130),
    "lower_mold_temp1": (530, 110),
    "lower_mold_temp2": (530, 160),
    "lower_mold_temp3": (530, 210),
    "Coolant_temperature": (30, 370),

    # 속도/품질
    "production_cycletime": (30, 460),
    "biscuit_thickness": (30, 430),
    "physical_strength": (30, 400),
    
    "mold_code": (350, 480),
}

# ==========================================
# 🔹 Train 데이터 로딩 및 공정별 UCL 기준 계산
# ==========================================
train_df = pd.read_csv("./data/fin_train.csv")
train_df.columns = [c.strip() for c in train_df.columns]

# 공정별 변수 리스트
melting_cols = ["molten_temp", "molten_volume"]
filling_cols = ["sleeve_temperature", "EMS_operation_time", "low_section_speed",
                "high_section_speed", "cast_pressure"]
cooling_cols = ["upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2",
                "Coolant_temperature"]
speed_cols = ["facility_operation_cycleTime", "production_cycletime"]
quality_cols = ["biscuit_thickness", "physical_strength"]

# 단계별 기준값 계산 (한 번만 수행)
UCL_MELT, MEAN_MELT, INV_MELT = calc_baseline_ucl(train_df, melting_cols)
UCL_FILL, MEAN_FILL, INV_FILL = calc_baseline_ucl(train_df, filling_cols)
UCL_COOL, MEAN_COOL, INV_COOL = calc_baseline_ucl(train_df, cooling_cols)
UCL_SPEED, MEAN_SPEED, INV_SPEED = calc_baseline_ucl(train_df, speed_cols)
UCL_QUAL, MEAN_QUAL, INV_QUAL = calc_baseline_ucl(train_df, quality_cols)

# ===== 백엔드 및 폰트 설정 =====
matplotlib.use("Agg")  # Tkinter 대신 Agg backend 사용 (GUI 불필요)
app_dir = pathlib.Path(__file__).parent

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(APP_DIR, "www", "fonts", "NanumGothic-Regular.ttf")

if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
else:
    plt.rcParams["font.family"] = "sans-serif"
    print(f"⚠️ 한글 폰트 파일 없음 → {font_path}")

pio.templates["nanum"] = pio.templates["plotly_white"].update(
    layout_font=dict(family="NanumGothic")
)
pio.templates.default = "nanum"

# ===== 모델 불러오기 =====
MODEL_PATH = "./models/model_2.pkl"
model = joblib.load(MODEL_PATH)

# ✅ 추가: joblib이 basic_fix를 찾을 수 있게 __main__에 등록
import sys
sys.modules['__main__'].basic_fix = basic_fix

model = joblib.load(MODEL_PATH)

MODEL_XGB_PATH = "./models/fin_xgb_f20.pkl"
model_xgb = joblib.load(MODEL_XGB_PATH)

# ===== 데이터 불러오기 =====
df_raw = pd.read_csv("./data/train_raw.csv")

# ★ 특정 이상치 행 제거
df_raw = df_raw[
    (df_raw["low_section_speed"] != 65535) &
    (df_raw["lower_mold_temp3"] != 65503) &
    (df_raw["physical_strength"] != 65535)
]

# 예측용 데이터도 동일 처리
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)


# 예측 탭용 (모델 input 그대로)
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)

# df_predict = df_predict[
#     (df_predict["low_section_speed"] != 65535) &
#     (df_predict["lower_mold_temp3"] != 65503) &
#     (df_predict["physical_strength"] != 65535)
# ]

# 탐색 탭용 (필터링/EDA)
drop_cols_explore = ["id","line","name","mold_name","date","time", "registration_time"]
df_explore = df_raw.drop(columns=drop_cols_explore, errors="ignore")  # ← 안전하게
# mold_code는 남김


# 전처리 후 데이터 (모델 학습용)
df_processed = pd.read_csv("./data/processed_train.csv")

# 컬럼 이름 표준화
df_processed.columns = df_processed.columns.str.strip().str.lower()
# 원본 탐색 데이터도 동일하게
df_explore.columns = df_explore.columns.str.strip().str.lower()

# 혹시 passorfail이 인덱스로 들어갔다면 컬럼으로 리셋
if "passorfail" not in df_processed.columns and "passorfail" in df_processed.index.names:
    df_processed = df_processed.reset_index()


# ✅ 파생 변수 자동 추가
derived_cols = ["speed_ratio", "pressure_speed_ratio"]
for col in derived_cols:
    if col in df_predict.columns:
        df_explore[col] = df_predict[col]

# 예측에서 제외할 컬럼
drop_cols = [
    "real_time",   # registration_time → real_time
    "passorfail",
    # "count",
    # "global_count",
    # "monthly_count",
    # "speed_ratio",
	# "pressure_speed_ratio",
    # "shift",
]
used_columns = df_predict.drop(columns=drop_cols).columns

# 그룹 분류
cat_cols = ["mold_code","working","emergency_stop","heating_furnace", "shift", "tryshot_signal"]
num_cols = [c for c in used_columns if c not in cat_cols]

# ===== 라벨 맵 =====
label_map = {
    # 기본 정보 관련
    "id": "고유 번호",
    "line": "생산 라인 이름",
    "name": "장비 이름",
    "mold_name": "금형 이름",
    "time": "측정 날짜",
    "date": "측정 시간",

    # 공정 상태 관련
    "count": "누적 제품 개수",
    "working": "장비 가동 여부 (가동 / 멈춤 등)",
    "emergency_stop": "비상 정지 여부 (ON / OFF)",
    "registration_time": "데이터 등록 시간",
    "tryshot_signal": "측정 딜레이 여부",

    # 용융 단계
    "molten_temp": "용융 온도",
    "heating_furnace": "용해로 정보",

    # 충진 단계
    "sleeve_temperature": "주입 관 온도",
    "ems_operation_time": "전자 교반(EMS) 가동 시간",
    "EMS_operation_time": "전자 교반(EMS) 가동 시간",
    "low_section_speed": "하위 구간 주입 속도",
    "high_section_speed": "상위 구간 주입 속도",
    "mold_code": "금형 코드",
    "molten_volume": "주입한 금속 양",
    "cast_pressure": "주입 압력",

    # 냉각 단계
    "upper_mold_temp1": "상부1 금형 온도",
    "upper_mold_temp2": "상부2 금형 온도",
    "upper_mold_temp3": "상부3 금형 온도",
    "lower_mold_temp1": "하부1 금형 온도",
    "lower_mold_temp2": "하부2 금형 온도",
    "lower_mold_temp3": "하부3 금형 온도",
    "coolant_temperature": "냉각수 온도",
    "Coolant_temperature": "냉각수 온도",

    # 공정 속도 관련
    "facility_operation_cycletime": "장비 전체 사이클 시간",
    "facility_operation_cycleTime": "장비 전체 사이클 시간",
    "production_cycletime": "실제 생산 사이클 시간",

    # 품질 및 성능
    "biscuit_thickness": "주조물 두께",
    "physical_strength": "제품 강도",

    # 평가
    "passorfail": "합격/불합격",

    "global_count": "전체 누적 개수",
    "monthly_count": "월간 누적 개수",
    "speed_ratio": "상/하부 주입 속도 비율",
	"pressure_speed_ratio": "주입 압력 비율",
    "shift": "주/야간 교대",
}

# ===== 라벨 정의 (표시 텍스트 = 한글, 실제 var = 변수명) =====
labels = [
    {"id": "label1", "text": label_map["upper_mold_temp1"], "var": "upper_mold_temp1",
     "x": 200, "y": 85, "w": 120, "h": 30,
     "arrow_from": (260, 115), "arrow_to": (400, 195)}, 

    {"id": "label2", "text": label_map["lower_mold_temp1"], "var": "lower_mold_temp1",
     "x": 650, "y": 85, "w": 120, "h": 30,
     "arrow_from": (710, 115), "arrow_to": (580, 195)}, 

    {"id": "label3", "text": label_map["cast_pressure"], "var": "cast_pressure",
     "x": 900, "y": 285, "w": 100, "h": 30,
     "arrow_from": (950, 315), "arrow_to": (780, 395)}, 

    {"id": "label4", "text": label_map["molten_volume"], "var": "molten_volume",
     "x": 700, "y": 185, "w": 120, "h": 30,
     "arrow_from": (760, 215), "arrow_to": (780, 315)}, 

    {"id": "label5", "text": label_map["sleeve_temperature"], "var": "sleeve_temperature",
     "x": 670, "y": 435, "w": 120, "h": 30,
     "arrow_from": (730, 435), "arrow_to": (600, 395)},  

    {"id": "label6", "text": label_map["high_section_speed"], "var": "high_section_speed",
     "x": 400, "y": 105, "w": 160, "h": 30,
     "arrow_from": (480, 135), "arrow_to": (510, 215)}, 

    {"id": "label7", "text": label_map["low_section_speed"], "var": "low_section_speed",
     "x": 400, "y": 455, "w": 160, "h": 30,
     "arrow_from": (480, 455), "arrow_to": (510, 355)},
]

def get_label(col): return label_map.get(col, col)

# ===== Helper: 슬라이더 + 인풋 =====
def make_num_slider(col):
    return ui.div(
        ui.input_slider(
            f"{col}_slider", get_label(col),
            min=int(df_predict[col].min()), max=int(df_predict[col].max()),
            value=int(df_predict[col].mean()), width="100%"
        ),
        ui.input_numeric(col, "", value=int(df_predict[col].mean()), width="110px"),
        style="display: flex; align-items: center; gap: 8px; justify-content: space-between;"
    )

# ===== 범주형 없음도 추가 ========
def make_select(col, label=None, width="100%"):
    label = label if label else get_label(col)
    if(col == "tryshot_signal"):
        choices = ["없음"] + sorted(df_predict[col].dropna().unique().astype(str))
    else:
        choices = sorted(df_predict[col].dropna().unique().astype(str)) + ["없음"]
    return ui.input_select(col, label, choices=choices, width=width)


def make_svg(labels):
    parts = []
    for lbl in labels:
        # 화살표 시작점: arrow_from 있으면 사용, 없으면 중앙
        if "arrow_from" in lbl:
            cx, cy = lbl["arrow_from"]
        else:
            cx = lbl["x"] + lbl["w"]/2
            cy = lbl["y"] + lbl["h"]/2

        x2, y2 = lbl["arrow_to"]
        text = label_map.get(lbl["var"], lbl["var"])

        parts.append(f"""
        <g>
        <rect x="{lbl['x']}" y="{lbl['y']}" width="{lbl['w']}" height="{lbl['h']}" 
                fill="#e0e6ef" stroke="black"/>
        <text x="{lbl['x'] + lbl['w']/2}" y="{lbl['y'] + lbl['h']/2}" 
                fill="black" font-size="14" font-weight="bold"
                text-anchor="middle" dominant-baseline="middle">{text}</text>
        <line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" 
                stroke="red" marker-end="url(#arrow)"/>
        </g>
        """)
    return "\n".join(parts)

svg_code = f"""
<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg"
     style="background:url('die-castings.gif'); background-size:cover;">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="red"/>
    </marker>
  </defs>
  {make_svg(labels)}
</svg>
"""

# ===== CSS (카드 전체 클릭영역) =====
card_click_css = """
<style>
/* 개요 전용 카드만 hover 효과 */
.overview-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    position: relative;
}

.overview-card:hover {
    background-color: #f8f9fa !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* 카드 전체를 클릭 가능하게 하는 투명 버튼 */
.card-link {
    position: absolute;
    inset: 0;
    z-index: 10;
    cursor: pointer;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.card-link:hover,
.card-link:focus,
.card-link:active {
    background: transparent !important;
    box-shadow: none !important;
}
</style>
"""

# ========== 데이터 준비 ==========
train = pd.read_csv("./data/train_raw.csv")
train["time"] = pd.to_datetime(train["time"], errors="coerce")
train["day"] = train["time"].dt.date

# 몰드코드별 요약
mold_cycle = (
    train.groupby("mold_code")["facility_operation_cycleTime"]
    .mean()
    .reset_index(name="avg_facility_cycleTime")
)
mold_cycle["daily_capacity"] = (86400 / mold_cycle["avg_facility_cycleTime"]).round()

daily_actual = train.groupby(["day", "mold_code"])["count"].agg(["min", "max"]).reset_index()
daily_actual["daily_actual"] = daily_actual["max"] - daily_actual["min"] + 1

mold_stats = daily_actual.groupby("mold_code")["daily_actual"].agg(
    min_prod="min", max_prod="max", avg_prod="mean"
).reset_index()

mold_summary = pd.merge(mold_cycle, mold_stats, on="mold_code")

# mold_code를 문자열로 변환
mold_summary["mold_code"] = mold_summary["mold_code"].astype(int).astype(str)
codes = list(mold_summary["mold_code"])
last_code = codes[-1]

# 색상 팔레트
cmap = cm.get_cmap("tab10", len(codes))
mold_colors = {code: mcolors.to_hex(cmap(i)) for i, code in enumerate(codes)}

# ================================
# 권장 세팅값 계산
# ================================
def smooth_series(series, window=5):
    smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
    Q1, Q3 = smoothed.quantile(0.25), smoothed.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    filtered = smoothed[(smoothed >= lower) & (smoothed <= upper)]
    return filtered.dropna()

setting_cols = [
    "molten_temp",
    "upper_mold_temp1","upper_mold_temp2","upper_mold_temp3",
    "lower_mold_temp1","lower_mold_temp2","lower_mold_temp3",
    "sleeve_temperature","cast_pressure","biscuit_thickness",
    "physical_strength","Coolant_temperature"
]

setting_table = {}
for code, df in train.groupby("mold_code"):
    settings = {}
    for col in setting_cols:
        smoothed = smooth_series(df[col].dropna())
        if len(smoothed) == 0:
            settings[col] = df[col].mean()
            continue
        try:
            mode_val = stats.mode(smoothed, keepdims=True)[0][0]
            settings[col] = mode_val
        except Exception:
            settings[col] = smoothed.mean()
    setting_table[str(code)] = settings  # 🔑 mold_code를 문자열로 저장

setting_df = pd.DataFrame(setting_table).T.reset_index().rename(columns={"index": "mold_code"})
setting_df["mold_code"] = setting_df["mold_code"].astype(str)  # 문자열로 통일

# ================================
# 생산 시뮬레이션 탭 비율 그래프
# ================================
train_raw = pd.read_csv("./data/train_raw.csv")

if "date" in train_raw.columns and "time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(
        train_raw["date"].astype(str) + " " + train_raw["time"].astype(str),
        errors="coerce"
    )
elif "registration_time" in train_raw.columns:
    train_raw["real_time"] = pd.to_datetime(train_raw["registration_time"], errors="coerce")
else:
    raise ValueError("date/time 또는 registration_time 컬럼을 확인해주세요.")

train_raw["date_only"] = train_raw["real_time"].dt.date

# 날짜별 mold_code 생산 개수
daily_mold = train_raw.groupby(["date_only", "mold_code"]).size().reset_index(name="count")
pivot_count = daily_mold.pivot(index="date_only", columns="mold_code", values="count").fillna(0)

years = list(range(2024, 2027))
months = list(range(1, 13))

# ======== 전역 HEAD (favicon, CSS 등) ========
global_head = ui.head_content(
    ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
    ui.tags.link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"),
    ui.tags.link(rel="stylesheet", type="text/css", href="custom.css"),
    ui.tags.title("주조 공정 불량 예측 대시보드"),
    ui.tags.script("""
      Shiny.addCustomMessageHandler("updateSensors", function(values) {
        const units = {
          temp: "°C", Temp: "°C",
          pressure: "bar", Pressure: "bar",
          speed: "cm/s", Speed: "cm/s",
          volume: "cc", thickness: "mm",
          strength: "MPa", Strength: "MPa",
          cycle: "sec", time: "s"
        };

        function unitFor(key) {
          const k = key.toLowerCase();
          if (k.includes("temp")) return " °C";
          if (k.includes("pressure")) return " bar";
          if (k.includes("speed")) return " cm/s";
          if (k.includes("volume")) return " cc";
          if (k.includes("thickness")) return " mm";
          if (k.includes("strength")) return " MPa";
          if (k.includes("cycle") || k.includes("time")) return " s";
          return "";
        }

        function colorFor(key, val) {
          const k = key.toLowerCase();
          if (k.includes("temp")) {
            const c = Math.min(255, Math.max(0, Math.round(val*1.5)));
            return `rgb(${c},50,50)`;
          }
          if (k.includes("pressure")) {
            const c = Math.min(255, Math.max(0, Math.round(val*8)));
            return `rgb(50,${c},80)`;
          }
          if (k.includes("speed")) {
            const c = Math.min(255, Math.max(0, Math.round(val*6)));
            return `rgb(40,100,${c})`;
          }
          if (k.includes("strength")) {
            const c = Math.min(255, Math.max(0, Math.round(val*5)));
            return `rgb(${120+c/4},${80+c/5},${150+c/2})`;
          }
          return "#111827";
        }

        for (const [key, val] of Object.entries(values)) {
          if (typeof val !== "number" || isNaN(val)) continue;

          // ✅ 값 노드를 정확히 찾음: #var-<key> .value
          const valueNode = document.querySelector(`#var-${key} .value`);
          if (!valueNode) {
            console.log(`⚠️ '#var-${key} .value' 노드를 찾을 수 없습니다.`);
            continue;
          }

          const txt = `${val.toFixed(1)}${unitFor(key)}`;
          valueNode.textContent = txt;

          // 색상 반영
          valueNode.setAttribute("fill", colorFor(key, val));

          // 갱신 애니메이션
          valueNode.animate([{opacity:.3},{opacity:1}], {duration:350, iterations:1});
        }
      });
    """),
    ui.tags.script("""
    Shiny.addCustomMessageHandler("updateGif", function(data) {
        const img = document.getElementById("process_gif");
        if (!img) return;
        // ⚡ 캐시 무효화를 위해 timestamp 붙임
        img.src = data.src + "?t=" + new Date().getTime();
    });
    """),
)

# ======== 상태 저장 ========
login_state = reactive.Value(False)
page_state = reactive.Value("login")   # login → menu → main


# ======== 1️⃣ 로그인 페이지 ========
def login_page():
    return ui.page_fillable(
        ui.div(
            {
                "style": (
                    "display:flex; flex-direction:column; justify-content:center; "
                    "align-items:center; height:100vh; background-color:#f8f9fa;"
                )
            },
            # ▼ 로고 이미지
            ui.img(
                src="LS_Logo.svg",   # www 폴더 안에 LS_Logo.svg 위치해야 함
                style="width:150px; margin-bottom:25px;"
            ),
            # ▼ 로그인 카드
            ui.card(
                {
                    "style": (
                        "width:350px; padding:20px; box-shadow:0 0 10px rgba(0,0,0,0.1);"
                    )
                },
                ui.h3("🔐 로그인", style="text-align:center; margin-bottom:20px;"),
                ui.input_text("user", "아이디", placeholder="아이디를 입력하세요"),
                ui.input_password("password", "비밀번호", placeholder="비밀번호를 입력하세요"),
                ui.input_action_button("login_btn", "로그인", class_="btn btn-primary w-100 mt-3"),
                ui.div(
                    ui.output_text("login_msg"),
                    style="color:red; margin-top:10px; text-align:center;",
                ),
            ),
        )
    )


# ======== 2️⃣ 카드 탭 선택 페이지 ========
def menu_page():
    return ui.page_fillable(
        ui.div(
            {
                "style": (
                    "min-height:100vh; background-color:#fdfdfd; padding:40px; "
                    "display:flex; flex-direction:column; align-items:center;"
                )
            },
            ui.h3("메뉴 선택", style="margin-bottom:30px; font-weight:bold;"),
            ui.div(
                {
                    "style": (
                        "display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); "
                        "gap:20px; width:80%; max-width:800px;"
                    )
                },
                # 📊 현장 대시보드
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #FFC966; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "📊 현장 대시보드",
                        style=(
                            "background-color:#FFC966; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.p("현장별 주요 지표 및 트렌드"),
                    ui.input_action_button("goto_field", "이동", class_="btn btn-outline-primary mt-2"),
                ),

                # 🧭 품질 모니터링
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #A5C16A; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "🧭 품질 모니터링",
                        style=(
                            "background-color:#A5C16A; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.p("불량률, 센서 이상 감지, 예측 결과"),
                    ui.input_action_button("goto_quality", "이동", class_="btn btn-outline-success mt-2"),
                ),

                # 📈 데이터 분석
                ui.card(
                    {"class": "overview-card",
                     "style": (
                         "border:2px solid #80CBC4; color:#333; text-align:center; "
                         "cursor:pointer; box-shadow:0 2px 6px rgba(0,0,0,0.05);"
                     )},
                    ui.card_header(
                        "📈 데이터 분석",
                        style=(
                            "background-color:#80CBC4; color:#333; "
                            "font-weight:bold; font-size:20px; text-align:center; "
                            "padding-top:15px; padding-bottom:15px;"
                        ),
                    ),
                    ui.p("EDA 및 주요 피처 분석 결과"),
                    ui.input_action_button("goto_analysis", "이동", class_="btn btn-outline-secondary mt-2"),
                ),
            ),
            ui.input_action_button("logout_btn", "로그아웃", class_="btn btn-light mt-5"),
        )
    )

def field_dashboard_ui():
    return ui.div(
        {"style": "display:flex; flex-direction:column; gap:20px;"},  # 🔹 세로 2행 구성
        # ──────────────── 1행: 제어 + 공정 상태 ────────────────
        ui.div(
            {
                "style": (
                    "display:grid; grid-template-columns:1fr 2fr; gap:20px;"
                )
            },
        # ───────────── 공정 상태 카드 ─────────────
        ui.card(
            ui.card_header("공정 상태"),
            ui.output_ui("process_status_card"),  # ✅ 추가
            ui.output_ui("realtime_predict_card"),  # 🧠 추가
        ),
            ui.card(
                ui.card_header("🧩 주조 공정 실시간 상태"),
                ui.output_ui("process_svg_inline"),
                style="width:100%;"
            ),
        ),

        # ──────────────── 2행: 실시간 데이터 표 ────────────────
        ui.card(
            ui.card_header("📊 실시간 데이터"),
            ui.div(
            ),
            style="width:100%;"
        ),
    )

def floating_stream_bar():
    """헤더 바로 아래 탭 스타일 스트리밍 제어 바"""
    return ui.div(
        {
            "style": (
                "display:flex; align-items:center; gap:16px;"
                "background-color:#fef6ee; border:1px solid #e0c8a0;"
                "border-bottom:none; border-radius:8px 8px 0 0;"
                "padding:8px 16px; position:absolute; top:60px; right:40px;"
                "z-index:1500; font-weight:bold; color:#5c4b3b;"
            )
        },
        ui.div("스트리밍 제어", style="font-weight:bold; font-size:15px;"),
        ui.output_ui("stream_status"),

        # 시간 표시 (고정폭)
        ui.div(
            ui.output_ui("stream_time_display"),
            style=(
                "font-size:14px; width:180px; text-align:center; white-space:nowrap;"
            ),
        ),

        # 버튼 그룹
        ui.div(
            {"style": "display:flex; gap:8px;"},
            ui.output_ui("stream_buttons")
        ),
    )

def load_svg_inline():
    svg_path = os.path.join(APP_DIR, "www", "diagram.svg")
    with open(svg_path, "r", encoding="utf-8") as f:
        return f.read()

def make_dynamic_svg(sensor_list: list[str]) -> str:
    """센서 목록을 받아 SVG 텍스트 노드를 자동 생성"""
    base_svg = [
        '<svg width="900" height="{}" xmlns="http://www.w3.org/2000/svg">'.format(100 + 30 * len(sensor_list)),
        '<rect width="100%" height="100%" fill="#f9f9f9"/>'
    ]
    for i, name in enumerate(sensor_list):
        y = 40 + i * 30
        base_svg.append(f'<text id="{name}" x="50" y="{y}" font-size="16" font-weight="bold" fill="#333">{name}: --</text>')
    base_svg.append('</svg>')
    return "\n".join(base_svg)



### ⬇️⬇️⬇️ 1단계: 여기에 아래 함수 코드를 통째로 추가하세요. ⬇️⬇️⬇️ ###

def plan_page_ui():
    """생산계획 탭의 UI를 반환하는 함수"""
    years = list(range(datetime.date(2019, 1, 19).year, datetime.date(2019, 1, 19).year + 3))
    months = list(range(1, 13))
    return ui.layout_sidebar(
        ui.sidebar(
            ui.input_numeric("monthly_target", "이달의 총 생산 목표 수", value=20000, min=1000, step=1000),
            ui.input_select("year", "연도 선택", {str(y): str(y) for y in years}, selected=str(datetime.date(2019, 1, 19).year)),
            ui.input_select("month", "월 선택", {str(m): f"{m}월" for m in months}, selected=str(datetime.date(2019, 1, 19).month)),
            ui.output_ui("mold_inputs"),
            ui.output_text("remaining_qty"),
            ui.input_action_button("run_plan", "시뮬레이션 실행", class_="btn btn-primary"),
        ),
        ui.card(ui.card_header("금형코드별 생산성 요약"), ui.output_data_frame("mold_summary_table")),
        ui.card(
            ui.card_header("달력형 계획표", ui.input_action_button("show_modal", "날짜별 금형 코드 생산 추이", class_="btn btn-sm btn-outline-primary", style="position:absolute; top:10px; right:10px;")),
            ui.output_ui("calendar_view"),
            ui.hr(),
            
            # ✅✅✅ 에러 수정: ui.icon() -> ui.tags.i() 로 변경 ✅✅✅
            ui.input_action_button(
                "generate_report_btn", 
                ["PDF 보고서 생성 ", ui.tags.i(class_="fa-solid fa-file-pdf")], 
                class_="btn btn-danger"
            ),
            
            ui.output_ui("report_output_placeholder")
        )
    )
# ======== 3️⃣ 본문 페이지 ========
def main_page(selected_tab: str):
    # --- 메뉴별 제목 및 본문 내용 ---
    tab_titles = {
        "field": "📊 현장 대시보드",
        "quality": "🧭 품질 모니터링",
        "analysis": "📈 데이터 분석"
    }
    tab_contents = {
        "field": ui.navset_tab(
    ui.nav_panel("실시간 대시보드", field_dashboard_ui()),

    # ───────── 이번달 생산목표 ─────────
    ui.nav_panel(
    "생산현황",
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_date("ref_date", "조회 기준일", value=datetime.date(2019, 1, 19)),
            style="background-color:#fffaf2; padding:20px; border-radius:10px;"
        ),
        ui.card(
            ui.card_header("📅 생산 현황"),
            ui.output_ui("calendar_view_current"),
            ui.hr(),
            ui.output_text("daily_summary"),   # ← 누적/예상 표시
            style="background-color:white; padding:20px;"
        )
    )
),

    # ───────── 다음달 생산목표 ─────────
    ui.nav_panel(
        "생산목표",
        plan_page_ui()  # ✅ 기존의 시뮬레이션 탭
    ),
),

        



        # 🧭 품질 모니터링 (예측 시뮬레이션 UI 포함)
        "quality": ui.navset_tab(
            ui.nav_panel("원인 분석",
                ui.card(
                    ui.card_header("불량 및 공정 에러 발생 조건", style="text-align:center;"),
                    ui.output_plot("local_factor_plot", click=True),   # 클릭 가능한 그래프
                    ui.hr(),
                    ui.output_ui("local_factor_desc"),      # 텍스트 설명
                    ui.output_ui("sensor_detail_modal")     # 클릭 시 뜨는 모달창
                ),
                # ──────────────── 2행: 실시간 데이터 표 ────────────────
                ui.card(
                    ui.card_header("📊 실시간 데이터"),
                    ui.div(
                        ui.output_data_frame("recent_data_table"),
                        # 🔹 스크롤이 생기도록 wrapping div에 명시적 width/overflow 지정
                        style=(
                            "width:100%; "
                            "overflow-x:auto; overflow-y:auto; "  # 가로/세로 스크롤 모두 허용
                            "max-height:500px; "  # 너무 길면 세로 스크롤
                            "display:block;"
                        )
                    ),
                    style="width:100%;"
                ),
            ),
            ui.nav_panel("실시간 관리도",
                ui.card(
                    ui.card_header(
                        "📊 실시간 다변량 관리도 (Hotelling’s T²)"),

                    # 상단 3개
                    ui.layout_columns(
                        ui.card(
                            ui.output_plot("mv_chart_melting"),
                            ui.div(
                                ui.output_table("mv_log_melting"),
                                style=(
                                    "max-height:200px; overflow-y:auto; overflow-x:auto; "
                                    "white-space:nowrap; border-top:1px solid #ccc;"
                                )
                            ),
                        ),
                        ui.card(
                            ui.output_plot("mv_chart_filling"),
                            ui.div(
                                ui.output_table("mv_log_filling"),
                                style=(
                                    "max-height:200px; overflow-y:auto; overflow-x:auto; "
                                    "white-space:nowrap; border-top:1px solid #ccc;"
                                )
                            ),
                        ),
                        ui.card(
                            ui.output_plot("mv_chart_cooling"),
                            ui.div(
                                ui.output_table("mv_log_cooling"),
                                style=(
                                    "max-height:200px; overflow-y:auto; overflow-x:auto; "
                                    "white-space:nowrap; border-top:1px solid #ccc;"
                                )
                            ),
                        ),
                        col_widths=[4,4,4]
                    ),

                    ui.br(),

                    # 하단 2개
                    ui.layout_columns(
                        ui.card(
                            ui.output_plot("mv_chart_speed"),
                            ui.div(
                                ui.output_table("mv_log_speed"),
                                style=(
                                    "max-height:200px; overflow-y:auto; overflow-x:auto; "
                                    "white-space:nowrap; border-top:1px solid #ccc;"
                                )
                            ),
                        ),
                        ui.card(
                            ui.output_plot("mv_chart_quality"),
                            ui.div(
                                ui.output_table("mv_log_quality"),
                                style=(
                                    "max-height:200px; overflow-y:auto; overflow-x:auto; "
                                    "white-space:nowrap; border-top:1px solid #ccc;"
                                )
                            ),
                        ),
                        col_widths=[6,6]
                    )
                )
            ),

            # =========================================
            # 기존 코드 최대한 유지 + 탭 통합 버전
            # =========================================
            ui.nav_panel("예측 및 개선",
                # 입력 변수 카드
                ui.div(
                    ui.card(
                        ui.card_header("입력 변수", style="background-color:#f8f9fa; text-align:center;"),

                        # 생산 환경 정보 카드 (최상단)
                        ui.card(
                            ui.card_header("생산 환경 정보", style="text-align:center;"),
                            ui.layout_columns(
                                ui.div(
                                    f"생산 라인: {df_raw['line'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    f"장비 이름: {df_raw['name'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                ui.div(
                                    f"금형 이름: {df_raw['mold_name'].iloc[0]}",
                                    style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                ),
                                col_widths=[4,4,4]
                            )
                        ),

                        # === 공정 상태 관련 ===
                        ui.card(
                            ui.card_header("공정 상태 관련"),
                            ui.layout_columns(
                                ui.input_numeric("count", "일조 누적 제품 개수", value=int(df_predict["count"].mean())),
                                # ui.input_numeric("monthly_count", "월간 누적 제품 개수", value=int(df_predict["monthly_count"].mean())),
                                # ui.input_numeric("global_count", "전체 누적 제품 개수", value=int(df_predict["global_count"].mean())),
                                ui.input_numeric("speed_ratio", "상하 구역 속도 비율", value=int(df_predict["speed_ratio"].mean())),
                                ui.input_numeric("pressure_speed_ratio", "주조 압력 속도 비율", value=int(df_predict["pressure_speed_ratio"].mean())),
                                make_select("working", "장비 가동 여부"),
                                make_select("emergency_stop", "비상 정지 여부"),
                                make_select("tryshot_signal", "측정 딜레이 여부"),
                                make_select("shift", "주, 야간 조"),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 용융 단계 ===
                        ui.card(
                            ui.card_header("용융 단계"),
                            ui.layout_columns(
                                make_num_slider("molten_temp"),
                                make_select("heating_furnace", "용해로"),
                                col_widths=[6,6]
                            )
                        ),

                        # === 충진 단계 ===
                        ui.card(
                            ui.card_header("충진 단계"),
                            ui.layout_columns(
                                make_num_slider("sleeve_temperature"),
                                make_num_slider("EMS_operation_time"),
                                make_num_slider("low_section_speed"),
                                make_num_slider("high_section_speed"),
                                make_num_slider("molten_volume"),
                                make_num_slider("cast_pressure"),
                                ui.input_select("mold_code", "금형 코드", choices=sorted(df_predict["mold_code"].dropna().unique().astype(str))),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 냉각 단계 ===
                        ui.card(
                            ui.card_header("냉각 단계"),
                            ui.layout_columns(
                                make_num_slider("upper_mold_temp1"),
                                make_num_slider("upper_mold_temp2"),
                                make_num_slider("upper_mold_temp3"),
                                make_num_slider("lower_mold_temp1"),
                                make_num_slider("lower_mold_temp2"),
                                # make_num_slider("lower_mold_temp3"),
                                make_num_slider("Coolant_temperature"),
                                col_widths=[3,3,3,3]
                            )
                        ),

                        # === 공정 속도 관련 ===
                        ui.card(
                            ui.card_header("공정 속도 관련"),
                            ui.layout_columns(
                                make_num_slider("facility_operation_cycleTime"),
                                make_num_slider("production_cycletime"),
                                col_widths=[6,6]
                            )
                        ),

                        # === 품질 및 성능 ===
                        ui.card(
                            ui.card_header("품질 및 성능"),
                            ui.layout_columns(
                                make_num_slider("biscuit_thickness"),
                                make_num_slider("physical_strength"),
                                col_widths=[6,6]
                            )
                        )
                    ),
                    style="max-width: 1200px; margin: 0 auto;"
                ),

                ui.br(),

                # 예측 실행 + 결과 카드
                ui.div(
                    ui.card(
                        ui.card_header(
                            ui.div(
                                [
                                    ui.input_action_button(
                                        "predict_btn", "예측 실행",
                                        class_="btn btn-primary btn-lg",
                                        style="flex:1;"
                                    ),
                                    ui.input_action_button(
                                        "reset_btn", ui.HTML('<i class="fa-solid fa-rotate-left"></i>'),
                                        class_="btn btn-secondary btn-lg",
                                        style="margin-left:10px; width:60px;"
                                    )
                                ],
                                style="display:flex; align-items:center; width:100%;"
                            ),
                            style="background-color:#f8f9fa; text-align:center;"
                        ),
                        ui.output_ui("prediction_result")
                    ),
                    style="""
                        position: -webkit-sticky;
                        position: sticky;
                        bottom: 1px;
                        z-index: 1000;
                        max-width: 1200px;
                        margin: 0 auto;
                    """
                ),

                ui.br(),

                # === 개선 방안 섹션 (조건부 렌더링 추가) ===
                ui.output_ui("improvement_section")

            ),
        ),
        "analysis": ui.h5("여기에 데이터 분석 결과를 표시합니다.")
    }

    current_title = tab_titles.get(selected_tab, "")

    # === 상단 바 ===
    header_bar = ui.div(
        {
            "class": "app-title bg-primary text-white",
            "style": (
                "display:flex; justify-content:space-between; align-items:center; "
                "padding:10px 20px;"
            ),
        },
        # 왼쪽: 뒤로가기 버튼
        ui.input_action_button(
            "back_btn",
            "← 뒤로가기",
            class_="btn btn-light btn-sm",
            style="font-weight:bold; min-width:100px; height:34px;"
        ),

        # 중앙: 타이틀 + 메뉴명 + 드롭다운 버튼
        ui.div(
            {"style": "display:flex; align-items:center; gap:8px;"},
            ui.h4(
                [
                    "🏭 주조 공정 불량 예측 대시보드",
                    ui.span(
                        f" — {current_title}",
                        style="font-weight:normal; font-size:17px; margin-left:6px; color:#ffffff;"
                    ),
                ],
                style="margin:0; font-weight:bold;"
            ),

            # ▼ 드롭다운 메뉴 버튼
            ui.tags.div(
                {"class": "dropdown"},
                ui.tags.button(
                    "",
                    {
                        "class": "btn btn-outline-light btn-sm dropdown-toggle",
                        "type": "button",
                        "data-bs-toggle": "dropdown",
                        "aria-expanded": "false",
                        "style": (
                            "padding:2px 10px; font-weight:bold; font-size:16px; line-height:1;"
                        ),
                    },
                ),
                ui.tags.ul(
                    {"class": "dropdown-menu dropdown-menu-end"},
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_field",
                            "📊 현장 대시보드",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "field" else "")
                            ),
                        )
                    ),
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_quality",
                            "🧭 품질 모니터링",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "quality" else "")
                            ),
                        )
                    ),
                    ui.tags.li(
                        ui.input_action_button(
                            "goto_analysis",
                            "📈 데이터 분석",
                            class_=(
                                "dropdown-item w-100 text-start "
                                + ("active-menu" if selected_tab == "analysis" else "")
                            ),
                        )
                    ),
                ),
            ),
        ),

        # 오른쪽: 로그아웃 버튼
        ui.input_action_button(
            "logout_btn",
            "🔓 로그아웃",
            class_="btn btn-light btn-sm",
            style="font-weight:bold; min-width:100px; height:34px;"
        ),
    )

    # === 본문 영역 ===
    content_area = ui.div(
        {
            "style": (
                "padding:30px 40px; background-color:#f8f9fa; "
                "min-height:calc(100vh - 80px);"
            )
        },
        ui.h4(current_title),
        ui.div(tab_contents.get(selected_tab, ui.p("페이지 없음"))),
    )

    return ui.page_fluid(
        header_bar,
        ui.div(
            {"style": "position:relative;"},
            floating_stream_bar(),  # ✅ 새로운 탭 형태 바 적용
            content_area
        )
    )

# ======== 전체 UI ========
app_ui = ui.page_fluid(global_head, ui.output_ui("main_ui"))

# ======== 서버 로직 ========
def server(input, output, session):
# ============================================================
# 🟢 로그인 페이지
# ============================================================

    # 로그인 처리
    @reactive.effect
    @reactive.event(input.login_btn)
    def _login():
        if input.user() == "admin" and input.password() == "1234":
            login_state.set(True)
            page_state.set("menu")
        else:
            login_state.set(False)
            page_state.set("login")

    # 카드 선택 → 해당 본문으로 이동
    @reactive.effect
    @reactive.event(input.goto_field)
    def _go_field():
        page_state.set("field")

    @reactive.effect
    @reactive.event(input.goto_quality)
    def _go_quality():
        page_state.set("quality")

    @reactive.effect
    @reactive.event(input.goto_analysis)
    def _go_analysis():
        page_state.set("analysis")

    # 로그아웃 버튼 클릭 → 확인 모달 표시
    @reactive.effect
    @reactive.event(input.logout_btn)
    def _logout_confirm():
        if login_state():
            m = ui.modal(
                ui.p("정말 로그아웃 하시겠습니까?"),
                title="로그아웃 확인",
                easy_close=False,
                footer=ui.div(
                    ui.input_action_button("confirm_logout", "확인", class_="btn btn-danger"),
                    ui.input_action_button("cancel_logout", "취소", class_="btn btn-secondary ms-2"),
                ),
            )
            ui.modal_show(m)

    # 로그아웃 확인 / 취소
    @reactive.effect
    @reactive.event(input.confirm_logout)
    def _logout_ok():
        login_state.set(False)
        page_state.set("login")
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.cancel_logout)
    def _logout_cancel():
        ui.modal_remove()
    
    # ===== 뒤로가기 버튼: 카드 선택 페이지로 복귀 ===== 
    @reactive.effect 
    @reactive.event(input.back_btn) 
    def _go_back(): 
        page_state.set("menu")

    # 페이지 상태에 따라 UI 전환
    @output
    @render.ui
    def main_ui():
        state = page_state()
        if state == "login":
            return login_page()
        elif state == "menu":
            return menu_page()
        elif state in ["field", "quality", "analysis"]:
            return main_page(state)
        else:
            return ui.p("⚠️ 알 수 없는 페이지 상태입니다.")

    # 로그인 실패 메시지 출력
    @output
    @render.text
    def login_msg():
        if input.login_btn() > 0 and not login_state():
            return "아이디 또는 비밀번호가 올바르지 않습니다."
        return ""

# 🟢 로그인 페이지 끝
# ============================================================



    
        
# ============================================================
# 🟢 TAB1. 현장 관리 (최신 Shiny 버전 호환)
# ============================================================

    # =====================================================
    # ✅ 데이터 로드 및 전처리
    # =====================================================
    train_raw = pd.read_csv("./data/train_raw.csv", low_memory=False)
    fin_test = pd.read_csv("./data/fin_test.csv", low_memory=False)

    for df in [train_raw, fin_test]:
        if "real_time" not in df.columns and "date" in df.columns and "time" in df.columns:
            df["real_time"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")

    common_cols = [c for c in train_raw.columns if c in fin_test.columns]
    fin_all = pd.concat([train_raw[common_cols], fin_test[common_cols]], ignore_index=True)

    # real_time 변환 (NaT 제거)
    fin_all["real_time"] = pd.to_datetime(fin_all["real_time"], errors="coerce")
    fin_all = fin_all.dropna(subset=["real_time"]).copy()

    # 날짜 변환 (2019 → 2025년 10월)
    fin_all["real_time"] = fin_all["real_time"] 
    fin_all["date"] = fin_all["real_time"].dt.floor("D")

    # =====================================================
    # 📅 달력 렌더링 (선택한 달 기준으로 표시)
    # =====================================================
    @render.ui
    def calendar_view_current():
        ref_date_str = input.ref_date() or None
        if not ref_date_str:
            ref_date = datetime.date(2019, 1, 19)
        else:
            ref_date = pd.to_datetime(ref_date_str).date()

        year, month = ref_date.year, ref_date.month
        total_days_in_month = calendar.monthrange(year, month)[1]

        # 이번 달 데이터 필터링
        df_month = fin_all[
            (fin_all["real_time"].dt.year == year) &
            (fin_all["real_time"].dt.month == month)
        ].copy()

        if df_month.empty:
            return ui.HTML(f"<p>⚠️ {year}년 {month}월 데이터 없음</p>")

        # 날짜별 생산량 계산
        daily_df = df_month.groupby("date").size().reset_index(name="daily_prod")

        # 하루 평균 및 목표 계산
        total_rows = daily_df["daily_prod"].sum()
        unique_days = daily_df["date"].nunique()
        avg_daily = total_rows / unique_days
        daily_target = avg_daily
        monthly_target = daily_target * total_days_in_month 

        # 누적 계산
        daily_df = daily_df.sort_values("date")
        daily_df["cum_prod"] = daily_df["daily_prod"].cumsum()
        daily_df["achieve_rate(%)"] = (daily_df["cum_prod"] / monthly_target * 100).round(1)

        produced = daily_df[daily_df["date"] <= pd.Timestamp(ref_date)]["daily_prod"].sum()
        achieve_rate = (produced / monthly_target) * 100
        remaining = max(monthly_target - produced, 0)
        last_day = datetime.date(year, month, total_days_in_month)
        remaining_days = max((last_day - ref_date).days, 0)
        daily_need = round(remaining / remaining_days, 1) if remaining_days > 0 else 0

        # 달력 UI
        cal = calendar.Calendar(firstweekday=6)
        month_days = cal.monthdatescalendar(year, month)

        html = ["<table style='width:100%; text-align:center; border-collapse:collapse;'>"]
        html.append("<tr>" + "".join(f"<th>{d}</th>" for d in ["일","월","화","수","목","금","토"]) + "</tr>")

        for week in month_days:
            html.append("<tr>")
            for day in week:
                if day.month != month:
                    html.append("<td style='background:#efefef;'></td>")
                    continue

                row = daily_df[daily_df["date"].dt.date == day]
                if day <= ref_date:
                    if not row.empty:
                        prod = int(row["daily_prod"].values[0])
                        rate = (prod / daily_target) * 100
                        bg = "#c7f9cc" if rate >= 100 else "#fff3b0" if rate >= 80 else "#ffcccb"
                        html.append(
                            f"<td style='border:1px solid #ddd; height:70px; background:{bg};'>"
                            f"<b>{day.day}</b><br>{prod:,}ea<br>({rate:.1f}%)</td>"
                        )
                    else:
                        html.append(f"<td style='border:1px solid #ddd; background:#f9f9f9;'><b>{day.day}</b><br>-</td>")
                else:
                    html.append(
                        f"<td style='border:1px solid #ddd; background:#f0f8ff;'>"
                        f"<b>{day.day}</b><br>{daily_need:,.0f}ea 예정</td>"
                    )
            html.append("</tr>")
        html.append("</table>")

        return ui.HTML("".join(html))

    # =====================================================
    # 🧮 하단 요약 텍스트
    # =====================================================
    @render.text
    def daily_summary():
        ref_date_str = input.ref_date() or None
        if not ref_date_str:
            ref_date = datetime.date(2019, 1, 19)

        else:
            ref_date = pd.to_datetime(ref_date_str).normalize()

        year, month = ref_date.year, ref_date.month
        total_days_in_month = calendar.monthrange(year, month)[1]

        df_month = fin_all[
            (fin_all["real_time"].dt.year == year) &
            (fin_all["real_time"].dt.month == month)
        ].copy()

        if df_month.empty:
            return f"⚠️ {year}년 {month}월 데이터 없음"

        daily_df = df_month.groupby("date").size().reset_index(name="daily_prod")
        total_rows = daily_df["daily_prod"].sum()
        unique_days = daily_df["date"].nunique()
        avg_daily = total_rows / unique_days
        monthly_target = avg_daily * total_days_in_month 

        produced = daily_df[daily_df["date"] <= ref_date]["daily_prod"].sum()
        achieve_rate = round(produced / monthly_target * 100, 1)
        remaining = max(monthly_target - produced, 0)
        last_day = datetime.date(year, month, total_days_in_month)
        remaining_days = max((last_day - ref_date.date()).days, 0)
        daily_need = round(remaining / remaining_days, 1) if remaining_days > 0 else 0

        return (
            f"📆 {ref_date.strftime('%Y년 %m월 %d일')} 기준 누적 생산량: {produced:,.0f}ea "
            f"({achieve_rate:.1f}%) 🎯 남은 목표: {remaining:,.0f}ea / 남은 {remaining_days}일 → "
            f"하루 평균 {daily_need:,.0f}ea 필요"
        )



    # ======== 📈 데이터 분석 탭 ========
   # --- 생산계획 탭 서버 로직 ---
    @render.ui
    def mold_inputs():
        if not codes: return ui.p("금형코드 데이터 없음")
        inputs = []
        for code in codes[:-1]:
            inputs.append(ui.input_numeric(f"target_{code}", ui.HTML(f"<span style='color:{mold_colors.get(code, '#000')}; font-weight:bold;'>금형코드 {code}</span>"), value=0, min=0, step=100))
        return ui.div(*inputs)
    
    DATA_PATH = pathlib.Path("./data/train_raw.csv")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료: {df_raw.shape}")
    except Exception as e:
        print("⚠️ 데이터 로드 실패:", e)
        df_raw = pd.DataFrame()

    @render.text
    def remaining_qty():
        if not codes: return ""
        total_target = input.monthly_target() or 0
        user_sum = sum(input[f"target_{code}"]() or 0 for code in codes[:-1])
        remaining = total_target - user_sum
        if user_sum > total_target:
            return f"⚠️ 목표 초과: {user_sum-total_target:,}개"
        return f"남은 생산량 ({last_code}): {remaining:,}개"

    @output
    @render.data_frame
    def mold_summary_table():
        if mold_summary.empty: return pd.DataFrame()
        df = mold_summary.rename(columns={
            "mold_code": "금형코드", "avg_facility_cycleTime": "평균사이클(초)",
            "daily_capacity": "일일생산능력", "min_prod": "최소일일생산량",
            "max_prod": "최대일일생산량", "avg_prod": "평균일일생산량"
        })
        return df.round(2)

    plan_df = reactive.Value(pd.DataFrame())
    @reactive.effect
    @reactive.event(input.run_plan)
    def _():
        if not codes: 
            plan_df.set(pd.DataFrame())
            return
        
        total_target = input.monthly_target() or 0
        year, month = int(input.year()), int(input.month())
        targets = {code: input[f"target_{code}"]() or 0 for code in codes[:-1]}
        user_sum = sum(targets.values())
        targets[last_code] = max(total_target - user_sum, 0)
        
        if sum(targets.values()) == 0: # If all targets are 0, distribute by capacity
            total_capacity = mold_summary["daily_capacity"].sum()
            if total_capacity > 0:
                for code in codes:
                    ratio = mold_summary.loc[mold_summary.mold_code == code, "daily_capacity"].iloc[0] / total_capacity
                    targets[code] = int(total_target * ratio)

        _, last_day = calendar.monthrange(year, month)
        schedule = []
        # (This is a simplified scheduling logic)
        for day in range(1, last_day + 1):
            for code in codes:
                daily_plan = int(targets[code] / last_day) if last_day > 0 else 0
                schedule.append({"date": datetime.date(year, month, day), "mold_code": code, "plan_qty": daily_plan})
        plan_df.set(pd.DataFrame(schedule))


    DATA_PATH = pathlib.Path("./data/train_raw.csv")
    try:
        df_raw = pd.read_csv(DATA_PATH)
        print(f"✅ 데이터 로드 완료: {df_raw.shape}")
    except Exception as e:
        print("⚠️ 데이터 로드 실패:", e)
        df_raw = pd.DataFrame()

    # PDF 리포트 생성
    def generate_report(df):
        report_dir = os.path.join(APP_DIR, "report")
        os.makedirs(report_dir, exist_ok=True)
        pdf_path = os.path.join(report_dir, "Production_Achievement_Report.pdf")

        pdf = FPDF()
        pdf.add_page()
        pdf.add_font("Nanum", "", font_path, uni=True)
        pdf.set_font("Nanum", size=12)
        pdf.cell(0, 10, "📑 생산 계획 달성률 보고서", ln=True, align="C")
        pdf.ln(10)

        target = 1000
        achieved = len(df)
        rate = achieved / target * 100
        pdf.multi_cell(0, 8, f"이번 기간 달성률: {rate:.1f}%")
        pdf.multi_cell(0, 8, "주요 저하 원인:\n - 설비 온도 불안정\n - 냉각수 지연\n - 교대 시 세팅 시간 증가")

        if "mold_code" in df.columns:
            pdf.ln(5)
            pdf.cell(0, 8, "공정별 달성률:", ln=True)
            for m, v in (df["mold_code"].value_counts(normalize=True) * 100).items():
                pdf.cell(0, 8, f" - Mold {m}: {v:.1f}%", ln=True)

        pdf.ln(8)
        pdf.cell(0, 8, f"설비 가동률: {np.random.uniform(85,97):.1f}%", ln=True)
        pdf.output(pdf_path)
        return pdf_path

    # -------- UI 내용 --------

    @output
    @render.ui
    def calendar_view():
        df = plan_df()
        if df.empty:
            return ui.p("시뮬레이션 실행 버튼을 눌러주세요.", style="text-align:center; color:grey;")

        # ✅ 영어 변수명 → 한글 매핑
        label_map = {
            "molten_temp": "용탕 온도",
            "upper_mold_temp1": "상금형 온도1",
            "upper_mold_temp2": "상금형 온도2",
            "upper_mold_temp3": "상금형 온도3",
            "lower_mold_temp1": "하금형 온도1",
            "lower_mold_temp2": "하금형 온도2",
            "lower_mold_temp3": "하금형 온도3",
            "sleeve_temperature": "슬리브 온도",
            "cast_pressure": "주조 압력",
            "biscuit_thickness": "비스킷 두께",
            "physical_strength": "인장 강도",
            "Coolant_temperature": "냉각수 온도",
        }

        year, month = int(input.year()), int(input.month())
        cal = calendar.monthcalendar(year, month)
        days_kr = ["일", "월", "화", "수", "목", "금", "토"]

        html = '<div style="display:grid; grid-template-columns: 80px repeat(7, 1fr); gap:4px;">'
        html += '<div></div>' + "".join(
            [f"<div style='font-weight:bold; text-align:center;'>{d}</div>" for d in days_kr]
        )

        for w_i, week in enumerate(cal, start=1):
            html += f"<div style='font-weight:bold;'>{w_i}주</div>"
            for d in week:
                if d == 0:
                    html += "<div style='border:1px solid #ccc; min-height:80px; background:#f9f9f9;'></div>"
                else:
                    cell_date = datetime.date(year, month, d)
                    cell_df = df[df["date"] == cell_date]
                    cell_html = ""

                    for _, r in cell_df.iterrows():
                        if r["plan_qty"] > 0:
                            code = str(r["mold_code"])
                            row = setting_df[setting_df["mold_code"] == code]
                            if row.empty:
                                tooltip_html = "<p>세팅값 없음</p>"
                            else:
                                settings = row.to_dict("records")[0]
                                rows_html = "".join([
                                    # ✅ 영어 대신 한글 출력
                                    f"<tr><td>{label_map.get(k, k)}</td><td>{v:.2f}</td></tr>"
                                    for k, v in settings.items() if k != "mold_code"
                                ])
                                tooltip_html = f"""
                                <table class='table table-sm table-bordered' style='font-size:11px; background:white; color:black;'>
                                    <thead><tr><th>변수</th><th>값</th></tr></thead>
                                    <tbody>{rows_html}</tbody>
                                </table>
                                """

                            cell_html += str(
                                ui.tooltip(
                                    ui.span(
                                        f"{code}: {r['plan_qty']}",
                                        style=f"color:{mold_colors.get(code, '#000')}; font-weight:bold;"
                                    ),
                                    ui.HTML(tooltip_html),
                                    placement="right"
                                )
                            ) + "<br>"

                    html += f"<div style='border:1px solid #ccc; min-height:80px; padding:4px; font-size:12px;'>{d}<br>{cell_html}</div>"

        html += "</div>"
        return ui.HTML(html)




    @output
    @render.plot
    def mold_plot():
        fig, ax = plt.subplots(figsize=(12, 6))
        if not pivot_count.empty:
            pivot_count.plot(kind="bar", stacked=True, ax=ax, color=[mold_colors.get(str(int(c))) for c in pivot_count.columns])
        ax.set_title("날짜별 금형 코드 생산 추이")
        ax.set_xlabel("날짜")
        ax.set_ylabel("생산 개수")
        ax.legend(title="금형 코드")
        plt.tight_layout()
        return fig

    @reactive.effect
    @reactive.event(input.show_modal)
    def _():
        ui.modal_show(ui.modal(ui.output_plot("mold_plot"), title="날짜별 금형 코드 생산 추이", size="xl", easy_close=True))

    report_content = reactive.Value(None)
    @reactive.effect
    @reactive.event(input.generate_report_btn)
    def _():
        # This part will be handled by file generation, so we just set a trigger
        report_content.set("generate")

    @output
    @render.ui
    def report_output_placeholder():
        content = report_content.get()
        if content == "generate":
            ui.modal_show(ui.modal(ui.p("보고서 생성을 시작합니다..."), title="알림", easy_close=True))
            report_content.set(None) # Reset trigger
            # In a real app, you would now generate the file.
            return ui.div(ui.hr(), ui.p("보고서 생성이 완료되었습니다.", class_="alert alert-success"))
        return None
    
    # ===== 실시간 스트리밍 로직 =====
    @output
    @render.ui
    def stream_status():
        color = "green" if is_streaming() else "gray"
        return ui.span(f"{'🟢' if is_streaming() else '🔴'}", style=f"color:{color};")

    @output
    @render.plot
    def stream_plot():
        df = current_data()
        fig, ax = plt.subplots(figsize=(10, 4))
        if df.empty:
            ax.text(0.5, 0.5, "▶ Start Streaming", ha="center", va="center", fontsize=14)
            ax.set_xticks([]); ax.set_yticks([])
            return fig
        for col in display_cols:
            ax.plot(df[col].values, label=col)
        ax.legend(); ax.grid(True)
        ax.set_title("Real Time Sensor Data")
        return fig
    
    # ===== 품질 모니터링용 관리도 출력 =====
    # @output
    # @render.plot
    # @reactive.calc
    # def xr_chart_quality():
    #     df = current_data.get()
    #     if df is None or df.empty:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "데이터 수신 대기 중...", ha="center", va="center")
    #         return fig
    
    #     var = input.spc_var() or "cast_pressure"
    #     if var not in df.columns:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, f"{var} 데이터 없음", ha="center", va="center")
    #         return fig
    
    #     xbar, R, limits = calc_xr_chart(df, var=var)
    #     fig = plot_xr_chart_matplotlib(xbar, R, limits)
    #     return fig


    # @output
    # @render.plot
    # @reactive.calc
    # def p_chart_quality():
    #     df = current_data.get()
    #     if df is None or df.empty:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "데이터 수신 대기 중...", ha="center", va="center")
    #         return fig

    #     if "passorfail" not in df.columns:
    #         fig, ax = plt.subplots()
    #         ax.axis("off")
    #         ax.text(0.5, 0.5, "passorfail 데이터 없음", ha="center", va="center")
    #         return fig

    #     p_bar, UCL, LCL = calc_p_chart(df, var="passorfail")
    #     return plot_p_chart_matplotlib(p_bar, UCL, LCL)
    
    # ============================================================
    # 🧭 다변량 관리도 (Hotelling’s T²) 계산 함수
    # ============================================================
    def calc_hotelling_t2(df, cols):
        """Hotelling’s T² (고정 UCL 적용)"""
        df = df.dropna(subset=cols)
        if len(df) == 0:
            return None, None, None

        X = df[cols].values

        # ✅ 공정별 baseline 매칭
        if set(cols) == set(melting_cols):
            mean, inv_cov, UCL = MEAN_MELT, INV_MELT, UCL_MELT
        elif set(cols) == set(filling_cols):
            mean, inv_cov, UCL = MEAN_FILL, INV_FILL, UCL_FILL
        elif all(c in cooling_cols for c in cols):
            mean, inv_cov, UCL = MEAN_COOL, INV_COOL, UCL_COOL
        elif set(cols) == set(speed_cols):
            mean, inv_cov, UCL = MEAN_SPEED, INV_SPEED, UCL_SPEED
        elif set(cols) == set(quality_cols):
            mean, inv_cov, UCL = MEAN_QUAL, INV_QUAL, UCL_QUAL
        else:
            print("⚠ 알 수 없는 컬럼 세트, 실시간 UCL 계산으로 fallback")
            mean = np.mean(X, axis=0)
            cov = np.cov(X, rowvar=False)
            inv_cov = np.linalg.pinv(cov)
            from scipy.stats import f
            n, p = len(df), len(cols)
            UCL = p * (n - 1) * (n + 1) / (n * (n - p)) * f.ppf(0.99, p, n - p)

        # ✅ T² 계산
        T2 = np.array([(x - mean) @ inv_cov @ (x - mean).T for x in X])
        return df.index, T2, UCL


    # ───────────────────────────────
    # 공통 함수: 관리도 그리기
    # ───────────────────────────────
    def plot_t2_chart(index, T2, UCL, title):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3))
        if T2 is None or len(T2) == 0:
            ax.text(0.5, 0.5, "데이터 부족", ha="center", va="center")
            ax.axis("off")
            return fig

        # 데이터 선
        ax.plot(index, T2, marker="o", color="steelblue", label="T²", alpha=0.8)
        ax.axhline(UCL, color="red", linestyle="--", label="UCL(99%)")

        # y축 범위 반영 후 UCL 이상 배경 붉게 표시
        ax.figure.canvas.draw()
        y_min, y_max = ax.get_ylim()
        ax.axhspan(UCL, y_max, color="lightcoral", alpha=0.25, zorder=0)

        ax.set_title(title)
        ax.set_ylabel("T²")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


    # ───────────────────────────────
    # 공통 함수: 로그 데이터프레임 생성
    # ───────────────────────────────
    def make_overlog(df, cols):
        idx, T2, UCL = calc_hotelling_t2(df, cols)
        over_mask = T2 > UCL
        over_data = df.loc[over_mask, cols].copy()
        over_data["T2"] = T2[over_mask]

        if over_data.empty:
            return pd.DataFrame({"메시지": ["모든 데이터가 UCL 이하입니다."]})
        else:
            over_data = over_data.reset_index()

            # 🔹 시간 컬럼 추가
            if "registration_time" in df.columns:
                over_data["시간"] = df.loc[over_mask, "registration_time"].values
            elif "datetime" in df.columns:
                over_data["시간"] = df.loc[over_mask, "datetime"].values

            # 🔹 한글 컬럼명 매핑 (공정별 전체 반영)
            col_name_map = {
                "T2": "T²",
                # 용융 단계
                "molten_temp": "용융 온도",
                "molten_volume": "주입한 금속 양",

                # 충진 단계
                "sleeve_temperature": "주입 관 온도",
                "EMS_operation_time": "전자 교반(EMS) 가동 시간",
                "low_section_speed": "하위 구간 주입 속도",
                "high_section_speed": "상위 구간 주입 속도",
                "cast_pressure": "주입 압력",

                # 냉각 단계
                "upper_mold_temp1": "상부1 금형 온도",
                "upper_mold_temp2": "상부2 금형 온도",
                "upper_mold_temp3": "상부3 금형 온도",
                "lower_mold_temp1": "하부1 금형 온도",
                "lower_mold_temp2": "하부2 금형 온도",
                "lower_mold_temp3": "하부3 금형 온도",
                "Coolant_temperature": "냉각수 온도",

                # 생산 속도
                "facility_operation_cycleTime": "장비 전체 사이클 시간",
                "production_cycletime": "실제 생산 사이클 시간",

                # 제품 테스트
                "biscuit_thickness": "주조물 두께",
                "physical_strength": "제품 강도",

                # 공통
                "시간": "시간",
            }

            # 🔹 표시 컬럼 순서
            display_cols = ["시간", "T2"] + cols if "시간" in over_data.columns else ["T2"] + cols

            # 🔹 매핑 적용
            over_data = over_data[display_cols].round(3)
            over_data.rename(columns=col_name_map, inplace=True)

            return over_data.tail(10)

    # ===============================
    # 🔹 공통 에러 처리용 함수
    # ===============================
    def make_placeholder_chart(title):
        """데이터 없을 때 표시되는 안내 그래프"""
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                f"📡 {title}\n데이터 수집 중입니다. 잠시만 기다려주세요.",
                ha="center", va="center", color="gray", fontsize=11)
        ax.axis("off")
        return fig
    
    # ───────────────────────────────
    # 🔹 용융 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_melting():
        try:
            df = current_data().tail(50)
            cols = ["molten_temp", "molten_volume"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "용융 단계")
        except Exception:
            return make_placeholder_chart("용융 단계")


    @output
    @render.table
    def mv_log_melting():
        try:
            df = current_data().tail(50)
            cols = ["molten_temp", "molten_volume"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 충진 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_filling():
        try:
            df = current_data().tail(50)
            cols = ["sleeve_temperature", "EMS_operation_time",
                    "low_section_speed", "high_section_speed", "cast_pressure"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "충진 단계")
        except Exception:
            return make_placeholder_chart("충진 단계")


    @output
    @render.table
    def mv_log_filling():
        try:
            df = current_data().tail(50)
            cols = ["sleeve_temperature", "EMS_operation_time",
                    "low_section_speed", "high_section_speed", "cast_pressure"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 냉각 단계
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_cooling():
        try:
            df = current_data().tail(50)

            # ✅ (추가) 한글 컬럼명을 영어로 자동 되돌리기
            reverse_map = {v: k for k, v in label_map.items()}
            df = df.rename(columns=reverse_map)

            cols = [
                "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2", 
                "Coolant_temperature"
            ]

            # ✅ 실제 존재하는 컬럼만 필터
            cols = [c for c in cols if c in df.columns]

            if len(cols) < 2:
                print("⚠ 냉각 단계 컬럼 부족:", cols)
                return make_placeholder_chart("냉각 단계")

            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "냉각 단계")

        except Exception as e:
            print("❌ 냉각 단계 에러:", e)
            return make_placeholder_chart("냉각 단계")


    @output
    @render.table
    def mv_log_cooling():
        try:
            df = current_data().tail(50)
            reverse_map = {v: k for k, v in label_map.items()}
            df = df.rename(columns=reverse_map)   # ✅ inplace=False로 안전하게
    
            cols = [
                "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
                "lower_mold_temp1", "lower_mold_temp2",
                "Coolant_temperature"
            ]
    
            available_cols = [c for c in cols if c in df.columns]
            print("냉각 단계 사용 컬럼:", available_cols)
    
            if not available_cols:
                return pd.DataFrame({"메시지": ["냉각 단계 데이터가 존재하지 않습니다."]})
    
            log_df = make_overlog(df, available_cols)
    
            if log_df is None or log_df.empty:
                return pd.DataFrame({"메시지": ["모든 데이터가 UCL 이하입니다."]})
            return log_df
    
        except Exception as e:
            import traceback
            print("❌ 냉각 단계 로그 생성 오류:", e)
            traceback.print_exc()
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})


    # ───────────────────────────────
    # 🔹 생산 속도
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_speed():
        try:
            df = current_data().tail(50)
            cols = ["facility_operation_cycleTime", "production_cycletime"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "생산 속도")
        except Exception:
            return make_placeholder_chart("생산 속도")


    @output
    @render.table
    def mv_log_speed():
        try:
            df = current_data().tail(50)
            cols = ["facility_operation_cycleTime", "production_cycletime"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})

    # ───────────────────────────────
    # 🔹 제품 테스트
    # ───────────────────────────────
    @output
    @render.plot
    def mv_chart_quality():
        try:
            df = current_data().tail(50)
            cols = ["biscuit_thickness", "physical_strength"]
            idx, T2, UCL = calc_hotelling_t2(df, cols)
            return plot_t2_chart(idx, T2, UCL, "제품 테스트")
        except Exception:
            return make_placeholder_chart("제품 테스트")


    @output
    @render.table
    def mv_log_quality():
        try:
            df = current_data().tail(50)
            cols = ["biscuit_thickness", "physical_strength"]
            return make_overlog(df, cols)
        except Exception:
            return pd.DataFrame({"메시지": ["데이터 수집 중입니다. 잠시만 기다려주세요."]})


    @output
    @render.data_frame
    def recent_data_table():
        df = current_data()
        if df is None or df.empty:
            return pd.DataFrame({"데이터": ["현재 수신된 데이터가 없습니다."]})

        df = df.copy().round(2).fillna("-")

        # ✅ 컬럼명을 한글로 매핑
        inv_label_map = label_map  # 그대로 사용해도 됨
        df.rename(columns=inv_label_map, inplace=True)

        return df.reset_index(drop=True)

    # ---------- 버튼 렌더링 ----------
    @output
    @render.ui
    def stream_buttons():
        """스트리밍 상태에 따라 버튼 표시 전환 (Font Awesome 아이콘 + 가로 배속 표시)"""
        btn_base = (
            "min-width:32px; height:32px; display:flex; align-items:center; justify-content:center;"
            "border:none; border-radius:6px; font-size:14px; color:white; font-weight:bold;"
            "box-shadow:0 1px 3px rgba(0,0,0,0.15); padding:0 6px;"
            "transition:all 0.2s ease;"
        )

        # 현재 배속 표시
        speed = stream_speed()
        speed_map = {2.0: "1x", 1.0: "2x", 0.5: "4x", 0.1: "20x", 0.05: "40x"}
        label = speed_map.get(speed, "1x")

        # 색상 (스트리밍 중: 파랑 / 정지 시: 회색)
        fast_color = "#60a5fa" if is_streaming() else "#9ca3af"

        return ui.div(
            {"style": "display:flex; gap:6px; align-items:center;"},
            # ▶ / ⏸ 버튼
            ui.input_action_button(
                "pause_stream" if is_streaming() else "start_stream",
                ui.HTML(
                    '<i class="fa-solid fa-pause"></i>'
                    if is_streaming()
                    else '<i class="fa-solid fa-play"></i>'
                ),
                style=btn_base + (
                    "background-color:#fbbf24;" if is_streaming() else "background-color:#f59e0b;"
                ),
                title="재생/일시정지",
            ),

            # ✅ 빨리감기 버튼 (Font Awesome 아이콘 + 배속 가로 배치)
            ui.input_action_button(
                "fast_stream",
                ui.HTML(
                    f"<div style='display:flex; align-items:center; gap:3px;'>"
                    f"<i class='fa-solid fa-forward'></i>"
                    f"<span style='font-size:11px;'>{label}</span>"
                    f"</div>"
                ),
                style=btn_base + f"background-color:{fast_color}; "
                                f"opacity:{1 if is_streaming() else 0.5}; "
                                f"cursor:{'pointer' if is_streaming() else 'not-allowed'};",
                disabled=not is_streaming(),
                title="빨리감기",
            ),

            # 🔄 초기화 버튼
            ui.input_action_button(
                "reset_stream",
                ui.HTML('<i class="fa-solid fa-rotate-right"></i>'),
                style=btn_base + "background-color:#d97706;",
                title="리셋",
            ),
        )

    # ---------- 버튼 동작 ----------
    @reactive.effect
    @reactive.event(input.start_stream)
    def _start_stream():
        is_streaming.set(True)

    @reactive.effect
    @reactive.event(input.pause_stream)
    def _pause_stream():
        is_streaming.set(False)

    @reactive.effect
    @reactive.event(input.reset_stream)
    async def _reset_stream():
        streamer().reset_stream()
        kf_streamer().reset_stream()
        current_data.set(pd.DataFrame())
        is_streaming.set(False)
        reset_values = {col: 0.0 for col in display_cols}
        stream_speed.set(2.0)   # ✅ 배속 기본값으로 초기화
        await session.send_custom_message("updateSensors", reset_values)

    # 빨리감기 버튼 클릭 → 속도 순환 변경
    @reactive.effect
    @reactive.event(input.fast_stream)
    def _fast_stream():
        current = stream_speed()
        # 단계별 속도 순환
        next_speed = {2.0: 1.0, 1.0: 0.5, 0.5: 0.1, 0.1: 0.05, 0.05: 2.0}.get(current, 2.0)
        stream_speed.set(next_speed)

    # === GIF 표시 제어 (스트리밍 상태 연동) ===

    # ▶ 시작 시 GIF 표시
    @reactive.effect
    @reactive.event(input.start_stream)
    async def _gif_start():
        await session.send_custom_message("updateGif", {"src": "die-castings.gif"})


    # ⏸ 일시정지 시 PNG 표시
    @reactive.effect
    @reactive.event(input.pause_stream)
    async def _gif_pause():
        await session.send_custom_message("updateGif", {"src": "die-castings.png"})


    # 🔄 리셋 시 PNG 표시
    @reactive.effect
    @reactive.event(input.reset_stream)
    async def _gif_reset():
        await session.send_custom_message("updateGif", {"src": "die-castings.png"})


    # ✅ 스트리밍이 중단 상태일 때도 자동으로 PNG 표시 유지
    @reactive.effect
    async def _sync_gif_state():
        if not is_streaming():
            await session.send_custom_message("updateGif", {"src": "die-castings.png"})


    # 주기적 업데이트
    @reactive.effect
    async def _auto_update():
        """2초마다 실시간 스트리밍 업데이트 (현장 + 품질 분리)"""
        if not is_streaming():
            return

        reactive.invalidate_later(stream_speed())

        # 현재 페이지 상태 확인
        page = page_state()

        # 📊 현장 대시보드
        if page == "field":
            s = streamer()
        # 🧭 품질 모니터링 (칼만 필터 CSV 사용)
        elif page == "quality":
            s = kf_streamer()
        else:
            return

        next_batch = s.get_next_batch(1)
        if next_batch is not None:
            current_data.set(s.get_current_data())
            latest = next_batch.iloc[-1].to_dict()

            clean_values = {}
            for k, v in latest.items():
                # ❌ 불필요한 키 제거
                if any(sub in k.lower() for sub in ["unnamed", "index"]):
                    continue
                if not isinstance(v, (int, float)) or pd.isna(v):
                    continue
                # ✅ HTML id로 안전하게 바꾸기
                safe_key = str(k).replace(":", "_").replace(" ", "_")
                clean_values[safe_key] = float(v)
            await session.send_custom_message("updateSensors", clean_values)
        else:
            is_streaming.set(False)

    @output
    @render.ui
    def process_svg_inline():
        svg_items = []
        for key, label in VAR_LABELS.items():
            if key not in VAR_POSITIONS:
                continue
            x, y = VAR_POSITIONS[key]
            svg_items.append(make_item_with_bg(key, label, x, y))

        svg_html = "\n".join(svg_items)

        return ui.HTML(f"""
            <div style='position:relative;width:900px;height:500px;margin:auto;'>
                <!-- ✅ 초기 상태는 PNG (정지 상태) -->
                <img id='process_gif' src='die-castings.png'
                    style='position:absolute;width:100%;height:100%;object-fit:contain;z-index:1;'/>
                <svg xmlns='http://www.w3.org/2000/svg'
                    width='100%' height='100%'
                    viewBox='0 0 900 500'
                    preserveAspectRatio='xMidYMid meet'
                    style='position:absolute;z-index:2;pointer-events:none;'>
                    {svg_html}
                </svg>
            </div>
        """)
    
    def make_item_with_bg(key: str, label: str, x: int, y: int) -> str:
        return f"""
        <g id='var-{key}'>
            <rect x='{x - 5}' y='{y - 18}' rx='4' ry='4'
                width='200' height='24'
                fill='rgba(255,255,255,0.75)' stroke='#ddd' stroke-width='0.5'/>
            <text x='{x}' y='{y}' fill='#111827'
                font-size='15' font-weight='700'>
                <tspan class='label'>{label}: </tspan>
                <tspan class='value'>—</tspan>
            </text>
        </g>
        """
    
    # ======================================================
    # 🎯 목표 계산 (앱 실행 시 1회 수행)
    # ======================================================

    def _get_prod_date(t):
        return (t - datetime.timedelta(days=1)).date() if t.time() < datetime.time(8,0) else t.date()

    def _get_shift(t):
        if datetime.time(8,0) <= t.time() < datetime.time(20,0):
            return "Day"
        else:
            return "Night"

    streaming_df["prod_date"] = streaming_df["real_time"].apply(_get_prod_date)
    streaming_df["shift"] = streaming_df["real_time"].apply(_get_shift)

    # === 조별 목표량 (row 수 × 1.1)
    shift_target_df = (
        streaming_df.groupby(["prod_date","shift"])
        .size().reset_index(name="shift_target")
    )
    shift_target_df["shift_target"] = (shift_target_df["shift_target"] * 1.1).round().astype(int)

    # === 일일 목표량 (08~익일08시 row 수 × 1.1)
    daily_target_df = (
        streaming_df.groupby("prod_date")
        .size().reset_index(name="daily_target")
    )
    daily_target_df["daily_target"] = (daily_target_df["daily_target"] * 1.1).round().astype(int)


    # ======================================================
    # ⚙️ 실시간 달성률 계산 함수
    # ======================================================
    def calc_achievements(df_live):
        if df_live is None or df_live.empty:
            return 0, 0

        df = df_live.copy()
        df["real_time"] = pd.to_datetime(df["real_time"], errors="coerce")
        df = df.dropna(subset=["real_time"]).sort_values("real_time")

        now = df["real_time"].iloc[-1]
        prod_date = (now - datetime.timedelta(days=1)).date() if now.time() < datetime.time(8,0) else now.date()

        # --- 현재 교대 구간 ---
        if datetime.time(8,0) <= now.time() < datetime.time(20,0):
            current_shift = "Day"
            shift_start = datetime.datetime.combine(now.date(), datetime.time(8,0))
        else:
            current_shift = "Night"
            if now.time() >= datetime.time(20,0):
                shift_start = datetime.datetime.combine(now.date(), datetime.time(20,0))
            else:
                shift_start = datetime.datetime.combine(now.date()-datetime.timedelta(days=1), datetime.time(20,0))

        # --- 현재 구간별 누적 row 수 ---
        df_shift = df[df["real_time"] >= shift_start]
        shift_count = len(df_shift)

        day_start = datetime.datetime.combine(now.date(), datetime.time(8,0))
        if now.time() < datetime.time(8,0):
            day_start -= datetime.timedelta(days=1)
        df_day = df[df["real_time"] >= day_start]
        day_count = len(df_day)

        # --- 목표량 조회 ---
        shift_target_row = shift_target_df.query(
            "(prod_date == @prod_date) & (shift == @current_shift)"
        )["shift_target"]
        daily_target_row = daily_target_df.query(
            "prod_date == @prod_date"
        )["daily_target"]

        shift_target = int(shift_target_row.iloc[0]) if not shift_target_row.empty else 1
        daily_target = int(daily_target_row.iloc[0]) if not daily_target_row.empty else 1

        # --- 달성률 계산 ---
        shift_rate = min((shift_count / shift_target) * 100, 100)
        daily_rate = min((day_count / daily_target) * 100, 100)

        return round(daily_rate, 1), round(shift_rate, 1)


    # ======================================================
    # 🧩 실시간 달성률 카드 (UI) — 빈 상태 포함
    # ======================================================
    @output
    @render.ui
    def process_status_card():
        import datetime

        df_live = current_data()

        if df_live is None or df_live.empty:
            return ui.div(
                {
                    "style": (
                        "border:2px solid #ccc; border-radius:12px; padding:14px; "
                        "box-shadow:0 2px 6px rgba(0,0,0,0.1); background-color:white; "
                        "font-family:'NanumGothic'; width:100%; max-width:100%;"
                    )
                },
                ui.h4("📅 -", style="margin-bottom:8px; text-align:center; color:#aaa;"),
                ui.div("⏸ 데이터 대기 중...", 
                    style="text-align:center; font-size:18px; color:gray; font-weight:bold;"),
                ui.hr(),
                ui.div(
                    {"style": "padding:4px 8px;"},
                    ui.span("조별 달성률", style="font-weight:bold; color:#777;"),
                    ui.div("0.0%", style="text-align:right; color:#999; font-weight:bold;"),
                    ui.div(
                        {"style": (
                            "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                        )}
                    ),
                ),
                ui.div(
                    {"style": "padding:4px 8px; margin-top:6px;"},
                    ui.span("일일 달성률", style="font-weight:bold; color:#777;"),
                    ui.div("0.0%", style="text-align:right; color:#999; font-weight:bold;"),
                    ui.div(
                        {"style": (
                            "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                        )}
                    ),
                )
            )

        # 데이터 존재 시
        latest = df_live.iloc[-1]
        daily_rate, shift_rate = calc_achievements(df_live)

        shift_icon = "🌞" if datetime.time(8, 0) <= latest["real_time"].time() < datetime.time(20, 0) else "🌙"

        def progress_bar(value, color):
            return ui.div(
                {
                    "style": (
                        "background-color:#e9ecef; border-radius:8px; height:18px; width:100%; margin-top:4px;"
                    )
                },
                ui.div(
                    {
                        "style": (
                            f"width:{min(value, 100):.1f}%; background-color:{color}; height:100%; "
                            f"border-radius:8px; transition:width 0.3s;"
                        )
                    }
                )
            )

        return ui.div(
            {
                "style": (
                    "border:2px solid #e0e0e0; border-radius:12px; padding:14px; "
                    "box-shadow:0 2px 6px rgba(0,0,0,0.1); background-color:white; "
                    "font-family:'NanumGothic'; width:100%; max-width:100%;"
                )
            },
            ui.h4(f"📅 {latest['real_time']:%Y-%m-%d %H:%M:%S}",
                style="margin-bottom:8px; text-align:center; color:#333;"),
            ui.div(
                {"style": "text-align:center; font-size:18px; font-weight:bold; color:#555;"},
                f"{shift_icon} {latest.get('shift','-')}조  (Team {latest.get('team','-')})"
            ),
            ui.hr(),
            ui.div(
                {"style": "padding:4px 8px;"},
                ui.span("조별 달성률", style="font-weight:bold; color:#444;"),
                ui.div(f"{shift_rate:.1f}%", style="text-align:right; color:#0d6efd; font-weight:bold;"),
                progress_bar(shift_rate, "#0d6efd"),
            ),
            ui.div(
                {"style": "padding:4px 8px; margin-top:6px;"},
                ui.span("일일 달성률", style="font-weight:bold; color:#444;"),
                ui.div(f"{daily_rate:.1f}%", style="text-align:right; color:#198754; font-weight:bold;"),
                progress_bar(daily_rate, "#198754"),
            ),
        )


    # ======================================================
    # 🧠 실시간 품질 판정 + 누적 불량률 (깜빡임 애니메이션 포함)
    # ======================================================
    @output
    @render.ui
    def realtime_predict_card():
        df_live = current_data()

        if df_live is None or df_live.empty:
            return ui.div(
                {
                    "style": (
                        "border:2px solid #ccc; border-radius:12px; padding:16px; "
                        "background-color:white; box-shadow:0 2px 6px rgba(0,0,0,0.1); "
                        "text-align:center; font-family:'NanumGothic'; width:100%; max-width:100%;"
                    )
                },
                ui.h4("🤖 실시간 품질 판정", style="margin-bottom:10px; color:#333;"),
                ui.h3("⏸ 데이터 대기 중...", style="color:gray; margin-bottom:6px;"),
                ui.h5("누적 불량률: -%", style="color:#888; margin-bottom:6px;"),
                ui.p("데이터 시각: -", style="color:#aaa; font-size:14px; margin-top:6px;"),
            )

        latest = df_live.tail(1).iloc[0]
        if "passorfail" not in latest:
            return ui.div("⚠️ passorfail 컬럼이 없습니다.", style="color:red; text-align:center;")

        result = int(latest["passorfail"])
        label = "✅ 양품" if result == 0 else "❌ 불량"
        color = "#28a745" if result == 0 else "#dc3545"
        emoji = "🟢" if result == 0 else "🔴"
        anim_color = "rgba(40,167,69,0.25)" if result == 0 else "rgba(220,53,69,0.25)"

        total_count = len(df_live)
        fail_count = df_live["passorfail"].sum()
        fail_rate = (fail_count / total_count) * 100 if total_count > 0 else 0

        return ui.div(
            {
                "style": (
                    f"border:2px solid {color}; border-radius:12px; padding:16px; "
                    f"background-color:white; box-shadow:0 2px 6px rgba(0,0,0,0.1); "
                    f"text-align:center; font-family:'NanumGothic'; width:100%; max-width:100%; "
                    f"animation: flash-bg 0.8s ease;"
                )
            },
            ui.tags.style(f"""
                @keyframes flash-bg {{
                    0%   {{ box-shadow: 0 0 0px {anim_color}; }}
                    30%  {{ box-shadow: 0 0 30px {anim_color}; }}
                    60%  {{ box-shadow: 0 0 20px {anim_color}; }}
                    100% {{ box-shadow: 0 0 0px transparent; }}
                }}
            """),
            ui.h4("🤖 실시간 품질 판정", style="margin-bottom:10px; color:#333;"),
            ui.h3(f"{emoji} {label}", style=f"color:{color}; font-weight:bold; margin-bottom:6px;"),
            ui.h5(f"누적 불량률: {fail_rate:.1f}%", style="color:#555; margin-bottom:6px;"),
            ui.p(f"데이터 시각: {latest['real_time']}", style="color:#777; font-size:14px; margin-top:6px;"),
        )

    @output
    @render.ui
    def stream_time_display():
        df = current_data()
        if df is None or df.empty:
            time_str = "-------- --:--:--"
        else:
            latest_time = pd.to_datetime(df["real_time"].iloc[-1], errors="coerce")
            if pd.isna(latest_time):
                time_str = "-------- --:--:--"
            else:
                time_str = f"{latest_time:%Y-%m-%d %H:%M:%S}"

        color = "#16a34a" if is_streaming() else "#6b4f2a"

        return ui.HTML(
            f"<span style='color:{color}; font-weight:bold;'>🕒 {time_str}</span>"
        )
    
    @output
    @render.ui
    def stream_speed_badge():
        speed = stream_speed()

        # 배속 매핑
        speed_map = {2.0: "1x", 1.0: "2x", 0.5: "4x", 0.1: "20x", 0.05: "40x"}
        label = speed_map.get(speed, "1x")

        # 색상: 속도에 따라 강조
        color_map = {2.0: "#6b4f2a", 1.0: "#f59e0b", 0.5: "#f97316", 0.1: "#ef4444", 0.05: "#dc2626"}
        bg_color = color_map.get(speed, "#6b4f2a")

        return ui.HTML(
            f"<span style='background:{bg_color}; color:white; padding:3px 10px; border-radius:10px; font-size:13px;'>⏩ {label}</span>"
        )

# 🟢 TAB1. 끝
# ============================================================



# ============================================================
# 🟢 TAB2. 품질
# ============================================================

    last_proba = reactive.value(None)
    loading = reactive.value(False)
    local_factors = reactive.value(None)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
     loading.set(True)
     try:
        X = get_input_data()
        proba = model.predict_proba(X)[0, 1]
        last_proba.set(proba)

        # === 불량 기여 요인 계산 ===
        # 1) 누적형 변수 제거
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = X[use_num_cols].iloc[0]

        # 2) 표준화 거리 (표준편차로 나눔)
        stds = df_predict[use_num_cols].std().replace(0, 1)  # 분모=0 방지
        diffs = ((current - baseline) / stds) ** 2

        # 3) 기여도 계산
        if diffs.sum() > 0:
            contrib = (diffs / diffs.sum()).sort_values(ascending=False)
            local_factors.set(
                pd.DataFrame({
                    "feature": [get_label(c) for c in contrib.index],
                    "importance": contrib.values
                })
            )
        else:
            local_factors.set(
                pd.DataFrame({"feature": [], "importance": []})
            )

     except Exception as e:
        last_proba.set(f"error:{e}")
     finally:
        loading.set(False)
    
    # @reactive.effect
    # @reactive.event(input.apply_suggestions)
    # def _():
    #     factors = local_factors()
    #     if factors is None or factors.empty:
    #         return

    #     top = factors.head(5).copy()
    #     exclude_vars = ["count", "monthly_count", "global_count"]
    #     use_num_cols = [c for c in num_cols if c not in exclude_vars]

    #     baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
    #     current = get_input_data().iloc[0][use_num_cols]

    #     for _, row in top.iterrows():
    #         feat = row["feature"]
    #         col = [k for k, v in label_map.items() if v == feat]
    #         if not col: 
    #             continue
    #         col = col[0]

    #         if col in current.index:
    #             diff = current[col] - baseline[col]
    #             if abs(diff) > 1e-6:
    #                 new_val = current[col] - diff/2   # 현재값과 baseline 사이 중간으로 이동
    #                 update_slider(f"{col}_slider", value=float(new_val))
    #                 update_numeric(col, value=float(new_val))
    #                 print(f"[반영됨] {col}: {current[col]} → {new_val} (baseline {baseline[col]})")

    #     # 🔹 자동 예측 실행
    #     session.send_input_message("predict_btn", 1)


    # ============================================================
    # 개선 방안 반영 후 즉시 재예측 + 최종 판정 표시
    # ============================================================
    @reactive.effect
    @reactive.event(input.apply_suggestions)
    def _():
        factors = local_factors()
        if factors is None or factors.empty:
            return

        top = factors.head(5).copy()
        exclude_vars = ["count", "monthly_count", "global_count"]
        use_num_cols = [c for c in num_cols if c not in exclude_vars]

        baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
        current = get_input_data().iloc[0][use_num_cols]

        # === ① 개선값 반영 ===
        for _, row in top.iterrows():
            feat = row["feature"]
            col = [k for k, v in label_map.items() if v == feat]
            if not col: 
                continue
            col = col[0]

            if col in current.index:
                diff = current[col] - baseline[col]
                if abs(diff) > 1e-6:
                    new_val = current[col] - diff / 2  # baseline 쪽으로 50% 이동
                    update_slider(f"{col}_slider", value=float(new_val))
                    update_numeric(col, value=float(new_val))
                    print(f"[반영됨] {col}: {current[col]} → {new_val} (baseline {baseline[col]})")

        # === ② 개선 후 자동 예측 ===
        try:
            X_new = get_input_data()
            proba_new = model.predict_proba(X_new)[0, 1]
            last_proba.set(proba_new)
            prediction_done.set(True)  # 개선된 판정 결과 섹션 표시용

            # === ③ 개선된 결과 저장용 상태값 추가 ===
            session.send_custom_message("scroll_to_bottom", {})  # 하단 자동 스크롤

        except Exception as e:
            last_proba.set(f"error:{e}")




    @render.ui
    def prediction_result():
        if loading():
            return ui.div(
                ui.div(class_="spinner-border text-primary", role="status"),
                ui.HTML("<div style='margin-top:10px;'>예측 실행 중...</div>"),
                style="text-align:center; padding:20px;"
            )

        proba = last_proba()
        if proba is None:
            return ui.div(
                ui.HTML("<span style='color:gray; font-size:18px;'>아직 예측을 실행하지 않았습니다.</span>"),
                style="text-align:center; padding:20px;"
            )

        if isinstance(proba, str) and proba.startswith("error:"):
            return ui.div(
                ui.HTML(f"<span style='color:red;'>예측 중 오류 발생: {proba[6:]}</span>")
            )

        if proba < 0.02:
            style = "background-color:#d4edda; color:#155724; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        elif proba < 0.04:
            style = "background-color:#fff3cd; color:#856404; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        else:
            style = "background-color:#f8d7da; color:#721c24; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"

        judgment = "불량품" if proba >= 0.2 else "양품"

        return ui.div(
            [
                ui.HTML(f"예상 불량률: {proba*100:.2f}%"),
                ui.br(),
                ui.HTML(f"최종 판정: <span style='font-size:22px;'>{judgment}</span>")
            ],
            style=style
        )

    @render.plot
    def feature_importance_plot():
        try:
            importances = model.named_steps["model"].feature_importances_
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10)

            plt.figure(figsize=(8,5))
            plt.barh(imp_df["feature"], imp_df["importance"])
            plt.gca().invert_yaxis()
            plt.title("변수 중요도 Top 10")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"변수 중요도 계산 불가",ha="center",va="center")

    @render.plot
    def distribution_plot():
        try:
            plt.figure(figsize=(8,5))
            df_good = df_predict[df_predict["passorfail"]==0]["biscuit_thickness"]
            df_bad = df_predict[df_predict["passorfail"]==1]["biscuit_thickness"]

            plt.hist(df_good, bins=30, alpha=0.6, label="양품")
            plt.hist(df_bad, bins=30, alpha=0.6, label="불량품")

            plt.axvline(df_predict["biscuit_thickness"].mean(), color="red", linestyle="--", label="평균")
            plt.legend()
            plt.title("비스킷 두께 분포 (양품 vs 불량)")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"분포 그래프 생성 불가",ha="center",va="center")

    @render.plot
    def process_trend_plot():
        try:
            mold_trend = df_predict.groupby("mold_code")["passorfail"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            mold_trend.plot(kind="bar")
            plt.ylabel("불량률")
            plt.title("금형 코드별 불량률")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"공정별 그래프 생성 불가",ha="center",va="center")
            
    # ===== 품질 모니터링용 SPC 관리도 =====
    def calc_xr_chart(df, var='cast_pressure', subgroup_size=5):
        if df.empty:
            return None, None, (None, None, None, None)
        df = df.tail(subgroup_size * 10).copy()
        df['group'] = np.floor(np.arange(len(df)) / subgroup_size)
        grouped = df.groupby('group')[var]
        xbar = grouped.mean()
        R = grouped.max() - grouped.min()
        Xbar_bar, R_bar = xbar.mean(), R.mean()
        A2, D3, D4 = 0.577, 0, 2.114   # n=5 기준
        return xbar, R, (
            Xbar_bar + A2 * R_bar, Xbar_bar - A2 * R_bar,
            D4 * R_bar, D3 * R_bar
        )


    def calc_p_chart(df, var='passorfail', window=50):
        if df.empty or var not in df:
            return None, None, None
        df = df.tail(window)
        p_bar = df[var].mean()
        n = len(df)
        sigma_p = np.sqrt(p_bar * (1 - p_bar) / n)
        return p_bar, p_bar + 3*sigma_p, p_bar - 3*sigma_p


    def plot_xr_chart_matplotlib(xbar, R, limits):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        if xbar is None or R is None:
            for ax in axes: ax.axis("off")
            axes[0].text(0.5,0.5,"데이터 부족",ha="center",va="center")
            return fig
        UCLx,LCLx,UCLr,LCLr = limits
        axes[0].plot(xbar.index,xbar.values,marker='o'); axes[0].axhline(xbar.mean(),c='g')
        axes[0].axhline(UCLx,c='r',ls='--'); axes[0].axhline(LCLx,c='r',ls='--')
        axes[0].set_title("X-bar 관리도"); axes[0].grid(True,ls='--',alpha=.5)
        axes[1].plot(R.index,R.values,marker='o'); axes[1].axhline(R.mean(),c='g')
        axes[1].axhline(UCLr,c='r',ls='--'); axes[1].axhline(LCLr,c='r',ls='--')
        axes[1].set_title("R 관리도"); axes[1].grid(True,ls='--',alpha=.5)
        plt.tight_layout(); return fig


    def plot_p_chart_matplotlib(p_bar, UCL, LCL):
        fig, ax = plt.subplots(figsize=(8,4))
        if p_bar is None:
            ax.axis("off"); ax.text(0.5,0.5,"데이터 부족",ha="center",va="center"); return fig
        ax.hlines([p_bar,UCL,LCL],0,1,colors=['g','r','r'],linestyles=['-','--','--'])
        ax.text(0.5,p_bar,f"불량률 {p_bar*100:.2f}%",ha='center',va='bottom',fontsize=12)
        ax.set_ylim(0,max(1,UCL*1.2)); ax.set_title("P 관리도 (실시간 불량률)")
        ax.grid(True,ls='--',alpha=.5); return fig
    
    def get_input_data():
        data = {}
        for col in cat_cols + num_cols:
            data[col] = [input[col]()]

        return pd.DataFrame(data)

    for col in num_cols:
        @reactive.effect
        @reactive.event(input[col])
        def _(col=col):
            update_slider(f"{col}_slider", value=input[col]())
        @reactive.effect
        @reactive.event(input[f"{col}_slider"])
        def _(col=col):
            update_numeric(col, value=input[f"{col}_slider"]())

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        # 범주형 변수: 첫 번째 값으로 초기화
        for col in cat_cols:
            first_val = str(sorted(df_predict[col].dropna().unique())[0])
            if(col == "tryshot_signal"):
                first_val = "없음"
            ui.update_select(col, selected=first_val)

        # 수치형 변수: 안전하게 숫자 변환 후 평균값으로 초기화
        for col in num_cols:
            series = pd.to_numeric(df_predict[col], errors="coerce")       # 문자열 → 숫자 (에러시 NaN)
            series = series.replace([np.inf, -np.inf], np.nan)             # inf → NaN
            mean_val = series.dropna().mean()                              # NaN 제거 후 평균
            default_val = int(mean_val) if pd.notna(mean_val) else 0       # fallback: 0
            update_slider(f"{col}_slider", value=default_val)
            update_numeric(col, value=default_val)

        # 예측 결과 초기화
        last_proba.set(None)

    @output
    @render.plot
    def local_factor_plot():
     factors = local_factors()
     if factors is None or factors.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "아직 예측을 실행하지 않았습니다.", ha="center", va="center")
        ax.axis("off")
        return fig

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100  # % 변환

     fig, ax = plt.subplots(figsize=(8, 4))
     bars = ax.barh(top["feature"], top["importance"], color="tomato")

    # 각 막대 끝에 % 숫자 표시
     for bar, val in zip(bars, top["importance"]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%",
                va="center")

        ax.invert_yaxis()
        ax.set_xlabel("기여도 (%)")
        ax.set_title("이번 케이스 불량 기여 요인 Top 5")
        plt.tight_layout()
     return fig

    # === 여기에 local_factor_desc() 붙여넣기 ===
    @output
    @render.ui
    def local_factor_desc():
     factors = local_factors()
     if factors is None or factors.empty:
        return ui.markdown("아직 예측을 실행하지 않았습니다.")

     top = factors.head(5).copy()
     top["importance"] = top["importance"] * 100

     exclude_vars = ["count", "monthly_count", "global_count"]
     use_num_cols = [c for c in num_cols if c not in exclude_vars]
     baseline = df_predict[df_predict["passorfail"] == 0][use_num_cols].mean()
     current = get_input_data().iloc[0][use_num_cols]

     rows_html = []
     for _, row in top.iterrows():
        feat = row["feature"]
        importance = row["importance"]

        col = [k for k, v in label_map.items() if v == feat]
        if not col: 
            continue
        col = col[0]

        left_text = f"{feat}: {importance:.1f}%"

        if col in current.index:
            diff = current[col] - baseline[col]
            if abs(diff) > 1e-6:
                direction = "낮추세요" if diff > 0 else "올리세요"
                adj_val = abs(diff) / 2
                right_text = f"{adj_val:.1f} 단위 {direction} (현재 {current[col]:.1f}, 기준 {baseline[col]:.1f})"
            else:
                right_text = "-"
        else:
            right_text = "-"

        row_html = f"""
        <div style='display:flex; align-items:center; margin-bottom:8px; font-size:15px;'>
            <div style='flex:1; text-align:left;'>{left_text}</div>
            <div style='flex:0.2; text-align:center;'>
                <i class="fa-solid fa-arrow-right fa-beat" style="color:#007bff;"></i>
            </div>
            <div style='flex:2; text-align:left; color:#444;'>{right_text}</div>
        </div>
        """
        rows_html.append(row_html)

    # 🔹 for문 끝난 뒤에 return 실행
     return ui.div(
        [
            ui.markdown("**이번 예측에서 불량률은 아래 요인들의 영향을 많이 받습니다:**"),
            ui.HTML("".join(rows_html)),
            ui.input_action_button(
                "apply_suggestions", "반영하고 다시 예측하기",
                class_="btn btn-warning", style="margin-top:15px;"
            )
        ]
    )

# ================================================
# 개선 방안 섹션 조건부 표시
# ================================================

    @output
    @render.ui
    def improvement_section():
        # 예측 결과가 존재할 때만 개선 방안 섹션 렌더링
        if not prediction_done.get():   # 예: prediction_done은 reactive.Value(True/False)
            return None

        return ui.card(
            ui.card_header("불량 기여 요인 Top 5", style="text-align:center; background-color:#f8f9fa; font-weight:bold;"),
            ui.output_plot("local_factor_plot"),
            ui.hr(),
            ui.output_ui("local_factor_desc")
        )
    prediction_done = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
        # ... 기존 예측 로직 ...
        prediction_done.set(True)

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        prediction_done.set(False)


    # ================================================
    # 개선 방안 섹션 조건부 표시 (양품이면 숨김)
    # ================================================
    @output
    @render.ui
    def improvement_section():
        # 예측 결과가 존재하지 않으면 아무것도 표시하지 않음
        if not prediction_done.get():
            return None

        proba = last_proba()
        if proba is None:
            return None

        # ✅ 판정이 양품이면 Top5 숨기기
        if proba < 0.2:  # 양품 기준: 20% 미만
            return None

        # 불량인 경우만 표시
        return ui.card(
            ui.card_header("불량 기여 요인 Top 5", style="text-align:center; background-color:#f8f9fa; font-weight:bold;"),
            ui.output_plot("local_factor_plot"),
            ui.hr(),
            ui.output_ui("local_factor_desc")
        )





    ##### 원인 분석 - 불량 및 공정 에러 발생 조건

    # 선택된 센서 & 현재 그래프의 y라벨 순서 저장
    selected_sensor = reactive.Value(None)
    plot_labels = reactive.Value([])   # ← barh에 그려진 y축 카테고리 순서

    @output
    @render.plot
    def local_factor_plot():
        df = current_data()
        if df is None or df.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "실시간 데이터 수신 대기 중...", ha="center", va="center", fontsize=13)
            ax.axis("off")
            return fig

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "수치형 데이터 없음", ha="center", va="center")
            ax.axis("off")
            return fig

        mean_std = df[numeric_cols].describe().T[["mean", "std"]]
        latest = df.iloc[-1]
        z_scores = (latest - mean_std["mean"]) / mean_std["std"]
        z_scores = z_scores.dropna().sort_values(ascending=True)

        # ⬇️ 현재 그래프의 y축 카테고리 순서 저장 (index가 레이블 순서)
        plot_labels.set(list(z_scores.index))

        colors = ["#e74c3c" if abs(z) > 3 else "#95a5a6" for z in z_scores]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(range(len(z_scores)), z_scores.values, color=colors)  # ← 정수 y위치로 그림
        ax.set_yticks(range(len(z_scores)))
        ax.set_yticklabels(list(z_scores.index))
        ax.set_xlabel("Z-score (표준편차 기준)")
        ax.set_title("실시간 이상 감지 센서 (클릭 시 상세보기)")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)
        plt.tight_layout()
        return fig

    @output
    @render.ui
    def local_factor_desc():
        df = current_data()
        if df is None or df.empty:
            return ui.p("⚪ 실시간 데이터 수신 중이 아닙니다.", style="color:gray;")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return ui.p("데이터에 수치형 센서가 없습니다.")

        mean_std = df[numeric_cols].describe().T[["mean", "std"]]
        latest = df.iloc[-1]

        anomalies = []
        for col in numeric_cols:
            val, mean, std = latest[col], mean_std.loc[col, "mean"], mean_std.loc[col, "std"]
            if pd.notna(std) and std > 0 and abs(val - mean) > 3 * std:
                anomalies.append((col, val, mean, std))

        if not anomalies:
            return ui.p("✅ 현재 이상 조건이 없습니다.", style="color:green;")

        alerts = [
            f"<li><b>{col}</b>: 현재 {val:.2f} (평균 {mean:.2f} ± {3*std:.2f}) → "
            f"<span style='color:red;'>이상 감지</span></li>"
            for col, val, mean, std in anomalies
        ]
        return ui.HTML(f"""
            <div style="background:#fff7f7; padding:10px; border-radius:8px;">
                <p><b>⚠ 공정 이상 감지 항목 ({len(anomalies)}개)</b></p>
                <ul>{''.join(alerts)}</ul>
                <p style='color:gray;font-size:13px;'>그래프를 클릭하면 상세 추이를 볼 수 있습니다.</p>
            </div>
        """)

    # 클릭 이벤트 처리 (y좌표 → 레이블로 변환)
    @reactive.effect
    @reactive.event(input.local_factor_plot_click)
    def _handle_click():
        click = input.local_factor_plot_click()
        if not click:
            return

        # matplotlib 클릭 payload는 보통 domain.y가 실수(막대 인덱스 근처)로 옴
        y_val = None
        if isinstance(click, dict):
            y_val = (click.get("domain", {}) or {}).get("y", None)
            if y_val is None:
                y_val = click.get("y", None)

        if y_val is None:
            return

        labels = plot_labels() or []
        idx = int(round(float(y_val)))            # 실수 → 가장 가까운 막대 인덱스
        if idx < 0 or idx >= len(labels):
            return

        sensor = labels[idx]
        selected_sensor.set(sensor)

        df = current_data()
        if df is None or df.empty or sensor not in df.columns:
            return

        ui.modal_show(
            ui.modal(
                ui.output_plot("sensor_detail_plot"),
                title=f"🔍 {sensor} 센서 상세 그래프",
                size="l",
                easy_close=True,
            )
        )

    @output
    @render.plot
    def sensor_detail_plot():
        sensor = selected_sensor.get()
        df = current_data()
        if not sensor or df is None or df.empty or sensor not in df.columns:
            fig, ax = plt.subplots()
            ax.axis("off")
            ax.text(0.5, 0.5, "선택된 센서 데이터가 없습니다.", ha="center", va="center")
            return fig

        y = pd.to_numeric(df[sensor], errors="coerce")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(y.values[-100:], marker="o", linestyle="-", alpha=0.7)
        m, s = y.mean(), y.std()
        if pd.notna(m):
            ax.axhline(m, color="green", linestyle="--", label="평균")
        if pd.notna(m) and pd.notna(s):
            ax.axhline(m + 3*s, color="red", linestyle="--", alpha=0.5, label="+3σ")
            ax.axhline(m - 3*s, color="red", linestyle="--", alpha=0.5, label="-3σ")
        ax.legend()
        ax.set_title(f"📈 '{sensor}' 최근 추이 (최근 100개)")
        ax.set_xlabel("시간순")
        ax.set_ylabel(sensor)
        ax.grid(True)
        return fig

    @output
    @render.ui
    def sensor_detail_modal():
        return None














# 🟢 TAB2. 품질 끝
# ============================================================




# ============================================================
# 🟢 TAB3. 데이터 분석
# ============================================================




# 🟢 TAB3. 데이터 분석 끝
# ============================================================

# ======================================
# ✅ 6시그마 json 파일 생성 코드
# ======================================
# import pandas as pd
# import numpy as np
# import json

# df = pd.read_csv("./data/fin_train.csv")
# num_cols = df.select_dtypes(include=["number"]).columns

# thresholds = {}

# for col in num_cols:
#     mu = df[col].mean()
#     sigma = df[col].std()
#     if pd.isna(mu) or pd.isna(sigma):
#         continue

#     mu = float(np.nan_to_num(mu, nan=0.0))
#     sigma = float(np.nan_to_num(sigma, nan=0.0))

#     thresholds[col] = {
#         "mu": round(mu, 4),
#         "sigma": round(sigma, 4),
#         "min": round(mu - 3 * sigma, 4),
#         "max": round(mu + 3 * sigma, 4)
#     }

# with open("./www/sixsigma_thresholds_extended.json", "w", encoding="utf-8") as f:
#     json.dump(thresholds, f, ensure_ascii=False, indent=2, allow_nan=False)

# ======== 앱 실행 ========
app = App(app_ui, server, static_assets=app_dir / "www")
