# ==================================================
# Panel Dashboard App
# Render Start Command:
# panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin="*"
# ==================================================

from pathlib import Path

import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas

from sklearn.ensemble import IsolationForest

pn.extension(
    "tabulator",
    sizing_mode="stretch_width",
    raw_css=[
"""
.bk-tabs-header .bk-tab {
    background-color: #e0e0e0;
    color: #666;
}

.bk-tabs-header .bk-tab.bk-active {
    background-color: #88d8b0 !important;
    color: black !important;
    font-weight: bold;
    border-bottom: 3px solid #2e7d32;
}

.bk-tabs-header .bk-tab:hover {
    background-color: #c8f0dc;
}
"""
    ]
)

pn.config.raw_css.append("""
/* 비활성 탭 (연하게) */
.bk-tabs-header .bk-tab:not(.bk-active) {
    background-color: #f5f5f5 !important;
    color: #999 !important;
    font-weight: normal !important;
}

/* 활성 탭 (진하게) */
.bk-tabs-header .bk-tab.bk-active {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-bottom: 2px solid #4CAF50 !important;
}
""")

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
    "장기보험 계",
    "사망 계",
    "생존 계",
    "의료비_상해",
    "의료비_질병",
    "질병생존_일당",
    "질병생존_3대진단",
]

mode_radio = pn.widgets.RadioButtonGroup(
    name="조회 방식",
    options=["최근 N개월", "기간 지정"],
    value="기간 지정",
    button_type="success",
)

n_months_slider = pn.widgets.IntSlider(
    name="최근 개월 수",
    start=6,
    end=120,
    step=6,
    value=60,
)

start_select = pn.widgets.Select(
    name="시작월",
    options=month_options,
    value=default_start,
)

end_select = pn.widgets.Select(
    name="종료월",
    options=month_options,
    value=default_end,
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

def get_period(mode, n_months, start_month, end_month):
    if mode == "최근 N개월":
        end_idx = month_options.index(end_month)
        start_idx = max(0, end_idx - n_months + 1)
        start_month = month_options[start_idx]

    if start_month > end_month:
        start_month, end_month = end_month, start_month

    return start_month, end_month


def get_filtered_df(mode, n_months, start_month, end_month):
    start_month, end_month = get_period(
        mode,
        n_months,
        start_month,
        end_month,
    )

    temp = df[
        (df["마감년월"] >= start_month)
        & (df["마감년월"] <= end_month)
    ].copy()

    return temp, start_month, end_month


def build_ai_df(filtered_df):
    df_ai = filtered_df.copy()

    df_ai["당월손해율(%)"] = pd.to_numeric(df_ai["당월손해율(%)"], errors="coerce")
    df_ai["누계손해율(%)"] = pd.to_numeric(df_ai["누계손해율(%)"], errors="coerce")

    df_ai = df_ai.sort_values(["담보분류", "마감년월"])

    df_ai["변화율"] = df_ai.groupby("담보분류")["당월손해율(%)"].pct_change()
    df_ai["편차"] = df_ai["당월손해율(%)"] - df_ai["누계손해율(%)"]

    features = ["당월손해율(%)", "변화율", "편차"]

    df_ai["AI위험점수"] = 0.0
    df_ai["AI판정"] = "정상"
    df_ai["AI설명"] = "정상 범위"

    for cov in df_ai["담보분류"].dropna().unique():
        temp = df_ai[df_ai["담보분류"] == cov].copy()

        if len(temp) < 20:
            df_ai.loc[temp.index, "AI설명"] = "데이터 수 부족으로 AI 판단 제외"
            continue

        X = temp[features].replace([np.inf, -np.inf], np.nan).fillna(0)

        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
        )

        pred = model.fit_predict(X)
        score = -model.decision_function(X)

        df_ai.loc[temp.index, "AI위험점수"] = pd.Series(score, index=temp.index)
        df_ai.loc[temp.index, "AI판정"] = np.where(pred == -1, "이상징후", "정상")

    df_ai["변화율"] = df_ai["변화율"].replace([np.inf, -np.inf], np.nan)

    def make_reason(row):
        reasons = []

        if row["AI판정"] != "이상징후":
            return "해당 담보의 최근 패턴 내에서 정상 범위로 판단"

        if pd.notna(row["변화율"]) and abs(row["변화율"]) >= 0.5:
            reasons.append("전월 대비 당월손해율 변화율이 큼")

        if pd.notna(row["편차"]) and abs(row["편차"]) >= 30:
            reasons.append("당월손해율과 누계손해율 간 편차가 큼")

        if pd.notna(row["당월손해율(%)"]) and row["당월손해율(%)"] >= 100:
            reasons.append("당월손해율이 100% 이상")

        if len(reasons) == 0:
            reasons.append("동일 담보의 과거 패턴 대비 비정상 조합으로 탐지")

        return ", ".join(reasons)

    df_ai["AI설명"] = df_ai.apply(make_reason, axis=1)

    return df_ai


@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def period_summary(mode, n_months, start_month, end_month):
    start_month, end_month = get_period(
        mode,
        n_months,
        start_month,
        end_month,
    )

    return pn.pane.Markdown(
        f"""
### 현재 조회조건

**조회방식:** {mode}

**조회기간:** {start_month} ~ {end_month}
"""
    )


@pn.depends(mode_radio)
def period_controls(mode):
    if mode == "최근 N개월":
        return pn.Column(
            n_months_slider,
            end_select,
        )

    return pn.Column(
        start_select,
        end_select,
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select, yaxis_loss_ratio)
def loss_ratio_plot(mode, n_months, start_month, end_month, selected_y):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    temp = temp[temp["담보분류"].isin(coverages)]

    temp = (
        temp.groupby(["담보분류", "마감년월"])[selected_y]
        .mean()
        .reset_index()
        .sort_values("마감년월")
        .reset_index(drop=True)
    )

    return temp.hvplot(
        x="마감년월",
        y=selected_y,
        by="담보분류",
        line_width=2,
        height=420,
        responsive=True,
        title=f"[ A 원수 손해율 추이 : 주요담보 ] {start_month} ~ {end_month}",
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select, yaxis_loss_ratio)
def loss_ratio_table(mode, n_months, start_month, end_month, selected_y):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    temp = temp[temp["담보분류"].isin(coverages)]

    temp = (
        temp.groupby(["담보분류", "마감년월"])[selected_y]
        .mean()
        .reset_index()
        .sort_values(["담보분류", "마감년월"])
        .reset_index(drop=True)
    )

    return pn.widgets.Tabulator(
        temp,
        pagination="remote",
        page_size=10,
        sizing_mode="stretch_width",
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def scatter_plot(mode, n_months, start_month, end_month):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    temp = temp[
        (temp["마감년월"] == end_month)
        & (~temp["담보분류"].isin(coverages))
    ]

    temp = (
        temp.groupby(["담보분류", "마감년월", "위험P(억원)"])["당월손해율(%)"]
        .mean()
        .reset_index()
        .sort_values("마감년월")
        .reset_index(drop=True)
    )

    return temp.hvplot(
        x="위험P(억원)",
        y="당월손해율(%)",
        by="담보분류",
        size=80,
        kind="scatter",
        alpha=0.7,
        legend=True,
        height=500,
        responsive=True,
        title=f"[ B 당월 위험보험료 VS 손해율 : 그 외 담보 ] {end_month}",
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select, yaxis_risk_premium_losses)
def bar_plot(mode, n_months, start_month, end_month, selected_y):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    temp = temp[
        (temp["마감년월"] == end_month)
        & (temp["담보분류"].isin(coverages))
    ]

    temp = (
        temp.groupby(["마감년월", "담보분류"])[selected_y]
        .sum()
        .reset_index()
        .sort_values("담보분류")
        .reset_index(drop=True)
    )

    return temp.hvplot(
        kind="bar",
        x="담보분류",
        y=selected_y,
        height=500,
        responsive=True,
        title=f"[ C 당월 위험보험료/손해액 비교 : 주요담보 ] {end_month}",
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def ai_summary(mode, n_months, start_month, end_month):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    ai_df = build_ai_df(temp)
    result = ai_df[ai_df["마감년월"] == end_month].copy()

    if len(result) == 0:
        return pn.pane.Markdown("## AI 이상탐지 요약\n\n조회 결과가 없습니다.")

    result = result.sort_values("AI위험점수", ascending=False)
    top = result.iloc[0]

    return pn.pane.Markdown(
        f"""
## AI 이상탐지 요약

**분석 기준:** 담보별 과거 패턴 기반 비지도학습

**조회기간:** {start_month} ~ {end_month}

**평가월:** {end_month}

**AI 최상위 위험 담보:** {top["담보분류"]}

**AI 판정:** {top["AI판정"]}

**당월손해율:** {round(top["당월손해율(%)"], 2)}%

**누계손해율:** {round(top["누계손해율(%)"], 2)}%

**전월 대비 변화율:** {round(top["변화율"], 4) if pd.notna(top["변화율"]) else "계산불가"}

**당월-누계 편차:** {round(top["편차"], 2) if pd.notna(top["편차"]) else "계산불가"}

**AI 위험점수:** {round(top["AI위험점수"], 4)}

### AI 설명

{top["AI설명"]}

### 추천 액션

AI 이상징후 담보는 당월 손해율 급등 여부, 고액 사고 발생 여부, 위험보험료 감소로 인한 분모 효과, 보장구조 변경 여부를 우선 점검하는 것이 좋습니다.
"""
    )


@pn.depends(mode_radio, n_months_slider, start_select, end_select)
def ai_risk_table(mode, n_months, start_month, end_month):
    temp, start_month, end_month = get_filtered_df(
        mode,
        n_months,
        start_month,
        end_month,
    )

    ai_df = build_ai_df(temp)
    result = ai_df[ai_df["마감년월"] == end_month].copy()

    result = result.sort_values("AI위험점수", ascending=False)

    result["AI판정표시"] = result["AI판정"].map(
        lambda x: "🔴 이상징후" if x == "이상징후" else "⚪ 정상"
    )

    result = result[
        [
            "마감년월",
            "담보분류",
            "당월손해율(%)",
            "누계손해율(%)",
            "변화율",
            "편차",
            "AI판정표시",
            "AI위험점수",
            "AI설명",
        ]
    ].head(20)

    return pn.widgets.Tabulator(
        result,
        pagination="remote",
        page_size=10,
        sizing_mode="stretch_width",
        stylesheets=[
            """
            .tabulator-row .tabulator-cell[title="이상징후"] {
            background-color: #ffcccc !important;
            color: #b00020 !important;
            font-weight: bold !important;
            }
            """
        ],
    )

image_pane = pn.pane.PNG(
    str(IMAGE_FILE),
    sizing_mode="scale_width",
)

template = pn.template.FastListTemplate(
    title="담보분류별 원수손해율 Dashboard",
    sidebar=[
        pn.pane.Markdown("# 담보별 당월 및 누계 손해율 변화"),
        pn.pane.Markdown(
            "#### 2006.1월에서 2022.3월까지 주요 담보별 당월 및 누계 원수손해율 변화를 조회합니다."
        ),
        pn.pane.Markdown("#### 1. A 원수 손해율 추이 : 7개 담보군"),
        pn.pane.Markdown("#### 2. 원수 손해율 리스트 : 7개 담보군"),
        pn.pane.Markdown("#### 3. B 당월 위험보험료 VS 손해율 : 그 외 담보"),
        pn.pane.Markdown("#### 4. C 당월 위험보험료/손해액 비교 : 주요담보"),
        pn.pane.Markdown("#### 5. AI 이상탐지 : 담보별 과거 패턴 기준"),
        image_pane,
        pn.pane.Markdown("## 조회 조건"),
        mode_radio,
        period_controls,
        period_summary,
    ],
    main=[
        pn.Tabs(
            ("📊 대시보드", pn.Column(
                pn.pane.Markdown("## 📊 대시보드"),
                pn.Row(
                    pn.Column(yaxis_loss_ratio, loss_ratio_plot, margin=(0, 25)),
                    loss_ratio_table,
                ),
                pn.Row(
                    pn.Column(scatter_plot, margin=(0, 25)),
                    pn.Column(yaxis_risk_premium_losses, bar_plot),
                ),
            )),
            ("🤖 AI 이상탐지", pn.Column(
                pn.pane.Markdown("## 🤖 AI 이상탐지"),
                pn.Row(
                    pn.Column(ai_summary, margin=(0, 25)),
                    ai_risk_table,
                ),
            )),
        )
    ],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

template.servable()
