# ==================================================
# Panel Dashboard App
# - 원수손해율 대시보드 배포용 app.py
# - 실행: panel serve app.py --show
# - Render Start Command:
#   panel serve app.py --address 0.0.0.0 --port $PORT --allow-websocket-origin="*"
# ==================================================

from pathlib import Path

import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401


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

# 문자열 기준 비교가 가능하도록 yyyy-mm 형태 유지
df["마감년월"] = df["마감년월"].astype(str)
df2["year5"] = df2["year5"].astype(str)

idf = df.interactive()


# ==================================================
# 3) 위젯
# ==================================================

discrete_slider = pn.widgets.DiscreteSlider(
    name="마감년월",
    options=list(df2["year5"]),
    value=list(df2["year5"])[-1],
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
# 4) 주요 담보 원수손해율 추이
# ==================================================

coverages = [
    "장기보험 계",
    "사망 계",
    "생존 계",
    "의료비_상해",
    "의료비_질병",
    "질병생존_일당",
    "질병생존_3대진단",
]

loss_ratio_pipeline = (
    idf[
        (idf["마감년월"] <= discrete_slider)
        & (idf["담보분류"].isin(coverages))
    ]
    .groupby(["담보분류", "마감년월"])[yaxis_loss_ratio]
    .mean()
    .to_frame()
    .reset_index()
    .sort_values(by="마감년월")
    .reset_index(drop=True)
)

loss_ratio_plot = loss_ratio_pipeline.hvplot(
    x="마감년월",
    by="담보분류",
    y=yaxis_loss_ratio,
    line_width=2,
    height=420,
    responsive=True,
    title="[ A 원수 손해율 추이 : 주요담보 ]",
)

loss_ratio_table = loss_ratio_pipeline.pipe(
    pn.widgets.Tabulator,
    pagination="remote",
    page_size=10,
    sizing_mode="stretch_width",
)


# ==================================================
# 5) 그 외 담보 위험보험료 VS 손해율
# ==================================================

risk_premium_vs_loss_ratio_scatterplot_pipeline = (
    idf[
        (idf["마감년월"] == discrete_slider)
        & (~idf["담보분류"].isin(coverages))
    ]
    .groupby(["담보분류", "마감년월", "위험P(억원)"])["당월손해율(%)"]
    .mean()
    .to_frame()
    .reset_index()
    .sort_values(by="마감년월")
    .reset_index(drop=True)
)

risk_premium_vs_loss_ratio_scatterplot = (
    risk_premium_vs_loss_ratio_scatterplot_pipeline.hvplot(
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
)


# ==================================================
# 6) 주요 담보 위험보험료/손해액 비교
# ==================================================

risk_premium_losses_bar_pipeline = (
    idf[
        (idf["마감년월"] == discrete_slider)
        & (idf["담보분류"].isin(coverages))
    ]
    .groupby(["마감년월", "담보분류"])[yaxis_risk_premium_losses]
    .sum()
    .to_frame()
    .reset_index()
    .sort_values(by="마감년월")
    .reset_index(drop=True)
)

risk_premium_losses_bar_plot = risk_premium_losses_bar_pipeline.hvplot(
    kind="bar",
    x="담보분류",
    y=yaxis_risk_premium_losses,
    height=500,
    responsive=True,
    title="[ C 당월 위험보험료/손해액 비교 : 주요담보 ]",
)


# ==================================================
# 7) 레이아웃
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
            pn.Column(yaxis_loss_ratio, loss_ratio_plot.panel(), margin=(0, 25)),
            loss_ratio_table.panel(width=500),
        ),
        pn.Row(
            pn.Column(risk_premium_vs_loss_ratio_scatterplot.panel(), margin=(0, 25)),
            pn.Column(yaxis_risk_premium_losses, risk_premium_losses_bar_plot.panel()),
        ),
    ],
    accent_base_color="#88d8b0",
    header_background="#88d8b0",
)

template.servable()