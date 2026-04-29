# ==================================================
# Panel Dashboard App
# Render Start Command:
# panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin="*"
# ==================================================

from pathlib import Path

import pandas as pd
import panel as pn
import hvplot.pandas
from sklearn.ensemble import IsolationForest

# ==================================================
# 1) 기본 설정
# ==================================================

pn.extension("tabulator", sizing_mode="stretch_width")

BASE_DIR = Path(__file__).resolve().parent
LOSS_RATIO_FILE = BASE_DIR / "loss_ratio.xlsx"
YEAR_FILE = BASE_DIR / "year2.xlsx"
IMAGE_FILE = BASE_DIR / "health.png"

# ==================================================
# 2) 데이터 로드
# ==================================================

df = pd.read_excel(LOSS_RATIO_FILE)
df2 = pd.read_excel(YEAR_FILE)

df["마감년월"] = pd.to_datetime(df["마감년월"]).dt.strftime("%Y-%m")
df2["year5"] = pd.to_datetime(df2["year5"]).dt.strftime("%Y-%m")

month_options = sorted(df["마감년월"].dropna().unique().tolist())

coverages = [
    "장기보험 계",
    "사망 계",
    "생존 계",
    "의료비_상해",
    "의료비_질병",
    "질병생존_일당",
    "질병생존_3대진단",
]

# ==================================================
# AI 기능: 손해율 이상탐지 모델
# ==================================================

ai_features = [
    "당월손해율(%)",
    "누계손해율(%)",
    "위험P(억원)",
    "손해액(억원)",
]

ai_df = df.copy()

for col in ai_features:
    ai_df[col] = pd.to_numeric(ai_df[col], errors="coerce")

ai_train = ai_df[ai_features].fillna(ai_df[ai_features].median())

ai_model = IsolationForest(
    n_estimators=200,
    contamination=0.05,
    random_state=42
)

ai_model.fit(ai_train)

ai_df["AI이상여부"] = ai_model.predict(ai_train)
ai_df["AI위험점수"] = -ai_model.decision_function(ai_train)
ai_df["AI판정"] = np.where(ai_df["AI이상여부"] == -1, "이상징후", "정상")

# ==================================================
# 3) 위젯
# ==================================================

discrete_slider = pn.widgets.DiscreteSlider(
    name="마감년월",
    options=month_options,
    value=month_options[-1],
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
# 4) 반응형 함수
# ==================================================

@pn.depends(discrete_slider, yaxis_loss_ratio)
def loss_ratio_plot(selected_month, selected_y):
    temp = df[
        (df["마감년월"] <= selected_month)
        & (df["담보분류"].isin(coverages))
    ]

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
        title="[ A 원수 손해율 추이 : 주요담보 ]",
    )


@pn.depends(discrete_slider, yaxis_loss_ratio)
def loss_ratio_table(selected_month, selected_y):
    temp = df[
        (df["마감년월"] <= selected_month)
        & (df["담보분류"].isin(coverages))
    ]

    temp = (
        temp.groupby(["담보분류", "마감년월"])[selected_y]
        .mean()
        .reset_index()
        .sort_values("마감년월")
        .reset_index(drop=True)
    )

    return pn.widgets.Tabulator(
        temp,
        pagination="remote",
        page_size=10,
        sizing_mode="stretch_width",
    )


@pn.depends(discrete_slider)
def scatter_plot(selected_month):
    temp = df[
        (df["마감년월"] == selected_month)
        & (~df["담보분류"].isin(coverages))
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
        title="[ B 당월 위험보험료 VS 손해율 : 그 외 담보 ]",
    )


@pn.depends(discrete_slider, yaxis_risk_premium_losses)
def bar_plot(selected_month, selected_y):
    temp = df[
        (df["마감년월"] == selected_month)
        & (df["담보분류"].isin(coverages))
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
        title="[ C 당월 위험보험료/손해액 비교 : 주요담보 ]",
    )

# ==================================================
# AI 분석 결과
# ==================================================

@pn.depends(discrete_slider)
def ai_risk_table(selected_month):
    temp = ai_df[ai_df["마감년월"] == selected_month].copy()

    temp = temp.sort_values(
        ["AI위험점수", "당월손해율(%)"],
        ascending=False
    )

    result = temp[
        [
            "마감년월",
            "담보분류",
            "당월손해율(%)",
            "누계손해율(%)",
            "위험P(억원)",
            "손해액(억원)",
            "AI판정",
            "AI위험점수",
        ]
    ].head(10)

    return pn.widgets.Tabulator(
        result,
        pagination="remote",
        page_size=10,
        sizing_mode="stretch_width",
    )


@pn.depends(discrete_slider)
def ai_explain_text(selected_month):
    temp = ai_df[ai_df["마감년월"] == selected_month].copy()

    temp = temp.sort_values(
        ["AI위험점수", "당월손해율(%)"],
        ascending=False
    )

    top = temp.iloc[0]

    coverage = top["담보분류"]
    monthly_lr = round(top["당월손해율(%)"], 1)
    cum_lr = round(top["누계손해율(%)"], 1)
    risk_score = round(top["AI위험점수"], 4)
    ai_judge = top["AI판정"]

    if ai_judge == "이상징후":
        action = "해당 담보의 손해율 급등 원인을 점검하고, 신계약 인수 기준 또는 보장 조건 재검토가 필요합니다."
    else:
        action = "현재는 뚜렷한 이상징후가 낮으나, 위험점수 상위 담보는 지속 모니터링이 필요합니다."

    text = f"""
## AI 이상탐지 요약

**선택 월:** {selected_month}

**AI가 가장 위험하게 본 담보:** {coverage}

**AI 판정:** {ai_judge}

**당월손해율:** {monthly_lr}%

**누계손해율:** {cum_lr}%

**AI 위험점수:** {risk_score}

### AI 해석

해당 담보는 선택 월 기준으로 손해율, 위험보험료, 손해액 패턴을 종합했을 때 상대적으로 위험도가 높게 탐지되었습니다.

### 추천 액션

{action}
"""
    return pn.pane.Markdown(text)


# ==================================================
# 5) 레이아웃
# ==================================================

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
        image_pane,
        pn.pane.Markdown("## 조회 기간 설정"),
        pn.pane.Markdown("#### 2006.1월 ~ 2022.3월"),
        discrete_slider,
    ],
    main=[
        pn.Row(
            pn.Column(yaxis_loss_ratio, loss_ratio_plot, margin=(0, 25)),
            loss_ratio_table,
        ),
        pn.Row(
            pn.Column(scatter_plot, margin=(0, 25)),
            pn.Column(yaxis_risk_premium_losses, bar_plot),
        ),
        pn.Row(
            pn.Column(ai_explain_text, margin=(0, 25)),
            ai_risk_table,
        ),
    ],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

template.servable()
