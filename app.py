# ==================================================
# 0. 기본
# ==================================================
from pathlib import Path
import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas
from sklearn.ensemble import IsolationForest

pn.extension("tabulator", sizing_mode="stretch_width")

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_excel(BASE_DIR / "loss_ratio.xlsx")

df["마감년월"] = pd.to_datetime(df["마감년월"])
df = df.sort_values("마감년월")

month_list = df["마감년월"].dt.strftime("%Y-%m").unique().tolist()

# ==================================================
# 1. 조회조건
# ==================================================
mode_radio = pn.widgets.RadioButtonGroup(
    options=["최근 N개월", "기간 지정"], value="기간 지정"
)

n_slider = pn.widgets.IntSlider(name="최근 개월", start=6, end=120, value=60)

start_select = pn.widgets.Select(name="시작월", options=month_list, value=month_list[-60])
end_select = pn.widgets.Select(name="종료월", options=month_list, value=month_list[-1])

# ==================================================
# 2. 데이터 필터
# ==================================================
def get_df(mode, n, start, end):

    if mode == "최근 N개월":
        return df.tail(n)

    return df[
        (df["마감년월"].dt.strftime("%Y-%m") >= start) &
        (df["마감년월"].dt.strftime("%Y-%m") <= end)
    ]

# ==================================================
# 3. 그래프
# ==================================================
@pn.depends(mode_radio, n_slider, start_select, end_select)
def line_plot(mode, n, start, end):

    temp = get_df(mode, n, start, end)

    return temp.hvplot.line(
        x="마감년월",
        y="당월손해율(%)",
        by="담보분류",
        height=400,
        title="당월 손해율 추이"
    )

# ==================================================
# 4. 테이블
# ==================================================
@pn.depends(mode_radio, n_slider, start_select, end_select)
def table_view(mode, n, start, end):

    temp = get_df(mode, n, start, end)

    return pn.widgets.Tabulator(temp, page_size=10)

# ==================================================
# 5. AI (핵심)
# ==================================================
def build_ai(temp):

    df_ai = temp.copy()

    df_ai["당월손해율(%)"] = pd.to_numeric(df_ai["당월손해율(%)"], errors="coerce")
    df_ai["누계손해율(%)"] = pd.to_numeric(df_ai["누계손해율(%)"], errors="coerce")

    df_ai["변화율"] = df_ai.groupby("담보분류")["당월손해율(%)"].pct_change()
    df_ai["편차"] = df_ai["당월손해율(%)"] - df_ai["누계손해율(%)"]

    df_ai["AI위험점수"] = 0.0
    df_ai["AI판정"] = "정상"

    for g in df_ai["담보분류"].unique():

        temp_g = df_ai[df_ai["담보분류"] == g]

        if len(temp_g) < 20:
            continue

        X = temp_g[["당월손해율(%)", "변화율", "편차"]].fillna(0)

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(X)

        score = -model.decision_function(X)

        df_ai.loc[temp_g.index, "AI위험점수"] = pd.Series(score, index=temp_g.index)

        df_ai.loc[temp_g.index, "AI판정"] = np.where(
            model.predict(X) == -1,
            "이상징후",
            "정상"
        )

    return df_ai

# ==================================================
# 6. AI 결과
# ==================================================
@pn.depends(mode_radio, n_slider, start_select, end_select)
def ai_view(mode, n, start, end):

    temp = get_df(mode, n, start, end)
    ai_df = build_ai(temp)

    latest = ai_df[ai_df["마감년월"] == ai_df["마감년월"].max()]
    latest = latest.sort_values("AI위험점수", ascending=False)

    return pn.widgets.Tabulator(
        latest[[
            "담보분류",
            "당월손해율(%)",
            "누계손해율(%)",
            "변화율",
            "편차",
            "AI판정",
            "AI위험점수"
        ]],
        page_size=10
    )

# ==================================================
# 7. 레이아웃
# ==================================================
template = pn.template.FastListTemplate(
    title="AI 손해율 이상탐지 Dashboard",

    sidebar=[
        "## 조회조건",
        mode_radio,
        n_slider,
        start_select,
        end_select,
    ],

    main=[
        pn.pane.Markdown("## 손해율 추이"),
        line_plot,

        pn.pane.Markdown("## 원본 데이터"),
        table_view,

        pn.pane.Markdown("## AI 이상탐지 결과"),
        ai_view,
    ],
)

template.servable()
