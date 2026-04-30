# ==================================================
# Panel Dashboard App
# ==================================================

from pathlib import Path
import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas
from sklearn.ensemble import IsolationForest

pn.extension("tabulator", sizing_mode="stretch_width")

BASE_DIR = Path(__file__).resolve().parent
LOSS_RATIO_FILE = BASE_DIR / "loss_ratio.xlsx"
YEAR_FILE = BASE_DIR / "year2.xlsx"
IMAGE_FILE = BASE_DIR / "health.png"

df = pd.read_excel(LOSS_RATIO_FILE)
df2 = pd.read_excel(YEAR_FILE)

df["마감년월"] = pd.to_datetime(df["마감년월"]).dt.strftime("%Y-%m")
df2["year5"] = pd.to_datetime(df2["year5"]).dt.strftime("%Y-%m")

month_options = sorted(df["마감년월"].dropna().unique().tolist())

end_idx = len(month_options) - 1
start_idx = max(0, end_idx - 59)

default_start = month_options[start_idx]
default_end = month_options[end_idx]

coverages = [
    "장기보험 계","사망 계","생존 계",
    "의료비_상해","의료비_질병",
    "질병생존_일당","질병생존_3대진단",
]

# ==================================================
# 조회조건
# ==================================================

mode_radio = pn.widgets.RadioButtonGroup(
    name="조회 방식",
    options=["최근 N개월", "기간 지정"],
    value="기간 지정",
    button_type="success",
)

n_months_slider = pn.widgets.IntSlider(
    name="최근 개월 수", start=6, end=120, step=6, value=60
)

start_select = pn.widgets.Select(
    name="시작월", options=month_options, value=default_start
)

end_select = pn.widgets.Select(
    name="종료월", options=month_options, value=default_end
)

yaxis_loss_ratio = pn.widgets.RadioButtonGroup(
    name="손해율 기준",
    options=["당월손해율(%)", "누계손해율(%)"],
    value="당월손해율(%)",
    button_type="success",
)

yaxis_risk_premium_losses = pn.widgets.RadioButtonGroup(
    name="위험보험료/손해액 기준",
    options=["위험P(억원)", "손해액(억원)"],
    value="위험P(억원)",
    button_type="success",
)

# ==================================================
# 데이터 필터
# ==================================================

def get_filtered_df(mode, n_months, start_month, end_month):

    if mode == "최근 N개월":
        end_idx = month_options.index(end_month)
        start_idx = max(0, end_idx - n_months + 1)
        start_month = month_options[start_idx]

    return df[
        (df["마감년월"] >= start_month)
        & (df["마감년월"] <= end_month)
    ].copy(), start_month, end_month

# ==================================================
# AI (🔥 수정 완료 버전)
# ==================================================

def build_ai_df(filtered_df):

    df_ai = filtered_df.copy()

    df_ai["당월손해율(%)"] = pd.to_numeric(df_ai["당월손해율(%)"], errors="coerce")
    df_ai["누계손해율(%)"] = pd.to_numeric(df_ai["누계손해율(%)"], errors="coerce")

    df_ai = df_ai.sort_values(["담보분류", "마감년월"])

    df_ai["변화율"] = df_ai.groupby("담보분류")["당월손해율(%)"].pct_change()
    df_ai["편차"] = df_ai["당월손해율(%)"] - df_ai["누계손해율(%)"]

    df_ai["AI위험점수"] = 0.0
    df_ai["AI판정"] = "정상"

    for cov in df_ai["담보분류"].unique():

        temp = df_ai[df_ai["담보분류"] == cov]

        if len(temp) < 20:
            continue

        X = temp[["당월손해율(%)","변화율","편차"]].fillna(0)

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)

        score = -model.decision_function(X)

        # 🔥 핵심 수정
        df_ai.loc[temp.index, "AI위험점수"] = pd.Series(score, index=temp.index)

        df_ai.loc[temp.index, "AI판정"] = np.where(
            model.predict(X)==-1,"이상징후","정상"
        )

    return df_ai

# ==================================================
# 그래프 / 테이블
# ==================================================

@pn.depends(mode_radio, n_months_slider, start_select, end_select, yaxis_loss_ratio)
def loss_ratio_plot(mode, n, start, end, selected_y):

    temp, start, end = get_filtered_df(mode, n, start, end)

    temp = temp[temp["담보분류"].isin(coverages)]

    return temp.hvplot(
        x="마감년월",
        y=selected_y,
        by="담보분류",
        height=420,
        responsive=True,
    )

@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def table_view(mode, n, start, end):

    temp, _, _ = get_filtered_df(mode, n, start, end)

    return pn.widgets.Tabulator(temp, page_size=10, height=300)

@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def ai_view(mode, n, start, end):

    temp, _, _ = get_filtered_df(mode, n, start, end)
    ai_df = build_ai_df(temp)

    latest = ai_df[ai_df["마감년월"] == ai_df["마감년월"].max()]

    return pn.widgets.Tabulator(
        latest.sort_values("AI위험점수", ascending=False),
        page_size=10,
        height=300
    )

# ==================================================
# Layout
# ==================================================

template = pn.template.FastListTemplate(
    title="손해율 Dashboard",

    sidebar=[
        "## 조회조건",
        mode_radio,
        n_months_slider,
        start_select,
        end_select,
    ],

    main=[
        pn.Column(
            "## 손해율 추이",
            loss_ratio_plot,

            "## 데이터",
            table_view,

            "## AI 이상탐지",
            ai_view,
        )
    ],
)

template.servable()
