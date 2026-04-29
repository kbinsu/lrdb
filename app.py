from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas

from sklearn.ensemble import IsolationForest

pn.extension("tabulator", sizing_mode="stretch_width")

BASE_DIR = Path(__file__).resolve().parent
LOSS_RATIO_FILE = BASE_DIR / "loss_ratio.xlsx"
IMAGE_FILE = BASE_DIR / "health.png"

df = pd.read_excel(LOSS_RATIO_FILE)

df["마감년월"] = pd.to_datetime(df["마감년월"]).dt.strftime("%Y-%m")

month_options = sorted(df["마감년월"].dropna().unique().tolist())

end_idx = len(month_options) - 1
start_idx = max(0, end_idx - 59)

default_start = month_options[start_idx]
default_end = month_options[end_idx]

# ==================================================
# 조회 위젯
# ==================================================

mode_radio = pn.widgets.RadioButtonGroup(
    name="조회 방식",
    options=["최근 N개월", "기간 지정"],
    value="기간 지정"
)

n_months_slider = pn.widgets.IntSlider(name="최근 개월 수", start=6, end=120, value=60)

start_select = pn.widgets.Select(name="시작월", options=month_options, value=default_start)
end_select = pn.widgets.Select(name="종료월", options=month_options, value=default_end)

# ==================================================
# 데이터 필터
# ==================================================

def get_filtered_df():
    if mode_radio.value == "최근 N개월":
        n = n_months_slider.value
        return df.sort_values("마감년월").tail(n)

    return df[
        (df["마감년월"] >= start_select.value)
        & (df["마감년월"] <= end_select.value)
    ]

# ==================================================
# AI (에러 해결 버전)
# ==================================================

def build_ai_df(filtered_df):

    df_ai = filtered_df.copy()

    df_ai["당월손해율(%)"] = pd.to_numeric(df_ai["당월손해율(%)"], errors="coerce")
    df_ai["누계손해율(%)"] = pd.to_numeric(df_ai["누계손해율(%)"], errors="coerce")

    df_ai = df_ai.sort_values(["담보분류", "마감년월"])

    df_ai["변화율"] = df_ai.groupby("담보분류")["당월손해율(%)"].pct_change()
    df_ai["편차"] = df_ai["당월손해율(%)"] - df_ai["누계손해율(%)"]

    features = ["당월손해율(%)", "변화율", "편차"]

    # ⭐ 핵심: float로 초기화
    df_ai["AI위험점수"] = 0.0
    df_ai["AI판정"] = "정상"

    for cov in df_ai["담보분류"].dropna().unique():

        temp = df_ai[df_ai["담보분류"] == cov]

        if len(temp) < 20:
            continue

        X = temp[features].fillna(0)

        model = IsolationForest(contamination=0.1, random_state=42)

        model.fit(X)

        # ⭐ 핵심 수정 부분
        score = -model.decision_function(X)

        df_ai.loc[temp.index, "AI위험점수"] = pd.Series(score, index=temp.index)

        df_ai.loc[temp.index, "AI판정"] = np.where(
            model.predict(X) == -1,
            "이상징후",
            "정상"
        )

    return df_ai

# ==================================================
# AI 결과
# ==================================================

@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def ai_result():

    temp = get_filtered_df()

    ai_df = build_ai_df(temp)

    latest = ai_df[ai_df["마감년월"] == ai_df["마감년월"].max()]

    latest = latest.sort_values("AI위험점수", ascending=False)

    return pn.widgets.Tabulator(
        latest[
            [
                "담보분류",
                "당월손해율(%)",
                "누계손해율(%)",
                "변화율",
                "편차",
                "AI판정",
                "AI위험점수",
            ]
        ],
        page_size=10
    )

# ==================================================
# Layout
# ==================================================

template = pn.template.FastListTemplate(
    title="AI 손해율 이상탐지 Dashboard",

    sidebar=[
        "## 조회조건",
        mode_radio,
        n_months_slider,
        start_select,
        end_select,
    ],

    main=[
        pn.pane.Markdown("## AI 이상탐지 결과"),
        ai_result,
    ],
)

template.servable()
