import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import json

PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"

with open(f"{REPORTS_DIR}/eda_summary.json") as f:
    eda = json.load(f)
with open(f"{REPORTS_DIR}/revenue_analysis.json") as f:
    rev = json.load(f)
with open(f"{REPORTS_DIR}/model_results.json") as f:
    model_results = json.load(f)

kpi     = eda["kpi_snapshot"]
rev_kpi = rev["revenue_kpis"]

BLUE   = "#0052CC"
CYAN   = "#06B6D4"
RED    = "#EF4444"
ORANGE = "#F97316"
GREEN  = "#22C55E"
BG     = "#0F172A"
CARD   = "#1E293B"
TEXT   = "#F1F5F9"
MUTED  = "#94A3B8"

def kpi_card(title, value, color):
    return html.Div(style={"backgroundColor":CARD,"borderRadius":"12px","padding":"20px","borderLeft":f"4px solid {color}"}, children=[
        html.P(title, style={"color":MUTED,"fontSize":"12px","margin":"0","textTransform":"uppercase","letterSpacing":"1px"}),
        html.H2(value, style={"color":TEXT,"fontSize":"28px","fontWeight":"700","margin":"8px 0 0 0"}),
    ])

def chart_card(title, chart):
    return html.Div(style={"backgroundColor":CARD,"borderRadius":"12px","padding":"20px"}, children=[
        html.H3(title, style={"color":TEXT,"fontSize":"14px","fontWeight":"600","margin":"0 0 16px 0","textTransform":"uppercase"}),
        chart,
    ])

def dark_layout():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, sans-serif", size=11),
        margin=dict(l=40, r=20, t=20, b=40), height=280,
        xaxis=dict(gridcolor="#334155", linecolor="#334155"),
        yaxis=dict(gridcolor="#334155", linecolor="#334155"),
    )

app = dash.Dash(__name__, title="HealthPulse Analytics")
app.layout = html.Div(style={"backgroundColor":BG,"minHeight":"100vh","fontFamily":"Inter, sans-serif","color":TEXT,"padding":"24px"}, children=[
    html.Div(style={"marginBottom":"32px"}, children=[
        html.H1("HealthPulse Analytics", style={"color":BLUE,"fontSize":"28px","fontWeight":"700","margin":"0"}),
        html.P("Patient & Revenue Intelligence Platform | 2021-2024", style={"color":MUTED,"margin":"4px 0 0 0","fontSize":"14px"}),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"repeat(4,1fr)","gap":"16px","marginBottom":"24px"}, children=[
        kpi_card("Total Claims",      f"{kpi['total_claims']:,}",                BLUE),
        kpi_card("Revenue Collected", f"${rev_kpi['total_collected']/1e6:.2f}M", GREEN),
        kpi_card("Revenue Gap",       f"${rev_kpi['total_gap']/1e6:.2f}M",       RED),
        kpi_card("30-day Readmit",    f"{kpi['readmit_rate_pct']}%",             ORANGE),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"2fr 1fr","gap":"16px","marginBottom":"24px"}, children=[
        chart_card("Monthly Revenue Trend", dcc.Graph(id="monthly-chart", config={"displayModeBar":False})),
        chart_card("Revenue Gap by Payer",  dcc.Graph(id="payer-chart",   config={"displayModeBar":False})),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"24px"}, children=[
        chart_card("30-Day Readmission by Department", dcc.Graph(id="readmit-chart", config={"displayModeBar":False})),
        chart_card("Readmission Lift by Risk Factor",  dcc.Graph(id="risk-chart",    config={"displayModeBar":False})),
    ]),
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"16px","marginBottom":"24px"}, children=[
        chart_card("Claim Denial Rate by Department", dcc.Graph(id="denial-chart", config={"displayModeBar":False})),
        chart_card("ML Model Performance (AUC-ROC)",  dcc.Graph(id="model-chart",  config={"displayModeBar":False})),
    ]),
    html.Div(style={"textAlign":"center","color":MUTED,"fontSize":"12px","marginTop":"16px"}, children=[
        html.P("HealthPulse Analytics | Built by Saatwika Matta | github.com/saatwika-matta")
    ]),
])

@app.callback(Output("monthly-chart","figure"), Input("monthly-chart","id"))
def monthly_chart(_):
    data = pd.DataFrame(rev["monthly_trend"])
    data["month"] = data["month"].astype(str)
    data["reimbursed_amt"] = pd.to_numeric(data["reimbursed_amt"], errors="coerce").fillna(0)
    data["charge_amount"]  = pd.to_numeric(data["charge_amount"],  errors="coerce").fillna(0)
    data["rolling_avg_3m"] = pd.to_numeric(data["rolling_avg_3m"], errors="coerce").fillna(0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["month"], y=data["charge_amount"], name="Charged", line=dict(color=BLUE, width=2), fill="tozeroy", fillcolor="rgba(0,82,204,0.1)"))
    fig.add_trace(go.Scatter(x=data["month"], y=data["reimbursed_amt"], name="Collected", line=dict(color=GREEN, width=2)))
    fig.add_trace(go.Scatter(x=data["month"], y=data["rolling_avg_3m"], name="3M Avg", line=dict(color=CYAN, width=1.5, dash="dot")))
    fig.update_layout(**dark_layout(), legend=dict(bgcolor="rgba(0,0,0,0)"), xaxis_tickangle=45)
    return fig

@app.callback(Output("payer-chart","figure"), Input("payer-chart","id"))
def payer_chart(_):
    data = pd.DataFrame(rev["revenue_by_payer"]).sort_values("total_gap")
    data["total_gap"] = pd.to_numeric(data["total_gap"], errors="coerce").fillna(0)
    fig = go.Figure(go.Bar(x=data["total_gap"], y=data["payer_type"], orientation="h",
        marker=dict(color=data["total_gap"], colorscale=[[0,BLUE],[0.5,CYAN],[1,RED]]),
        text=[f"${v/1e6:.2f}M" for v in data["total_gap"]], textposition="outside", textfont=dict(color=TEXT)))
    fig.update_layout(**dark_layout())
    return fig

@app.callback(Output("readmit-chart","figure"), Input("readmit-chart","id"))
def readmit_chart(_):
    data = pd.DataFrame(eda["readmit_by_dept"]).sort_values("readmit_rate")
    data["readmit_rate"] = pd.to_numeric(data["readmit_rate"], errors="coerce").fillna(0)
    fig = go.Figure(go.Bar(x=data["readmit_rate"], y=data["department"], orientation="h",
        marker=dict(color=data["readmit_rate"], colorscale=[[0,GREEN],[0.5,ORANGE],[1,RED]]),
        text=[f"{v:.1f}%" for v in data["readmit_rate"]], textposition="outside", textfont=dict(color=TEXT)))
    fig.update_layout(**dark_layout())
    return fig

@app.callback(Output("risk-chart","figure"), Input("risk-chart","id"))
def risk_chart(_):
    data    = eda["risk_factor_readmit"]
    cats    = ["Diabetes","Hypertension","Smoker","High Risk"]
    keys    = ["has_diabetes","has_hypertension","is_smoker","is_high_risk"]
    without = [float(data[k]["without"]) for k in keys]
    with_   = [float(data[k]["with"])    for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Without", x=cats, y=without, marker_color=BLUE))
    fig.add_trace(go.Bar(name="With",    x=cats, y=with_,   marker_color=ORANGE))
    fig.update_layout(**dark_layout(), barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)"), yaxis_title="Readmit %")
    return fig

@app.callback(Output("denial-chart","figure"), Input("denial-chart","id"))
def denial_chart(_):
    data = pd.DataFrame(rev["denial_by_dept"]).sort_values("denial_rate_pct")
    data["denial_rate_pct"] = pd.to_numeric(data["denial_rate_pct"], errors="coerce").fillna(0)
    fig = go.Figure(go.Bar(x=data["denial_rate_pct"], y=data["department"], orientation="h",
        marker=dict(color=data["denial_rate_pct"], colorscale=[[0,CYAN],[0.5,ORANGE],[1,RED]]),
        text=[f"{v:.1f}%" for v in data["denial_rate_pct"]], textposition="outside", textfont=dict(color=TEXT)))
    fig.update_layout(**dark_layout())
    return fig

@app.callback(Output("model-chart","figure"), Input("model-chart","id"))
def model_chart(_):
    names = list(model_results.keys())
    aucs  = [float(model_results[n]["auc_roc"]) for n in names]
    f1s   = [float(model_results[n]["f1"])      for n in names]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="AUC-ROC",  x=names, y=aucs, marker_color=BLUE, text=[f"{v:.4f}" for v in aucs], textposition="outside", textfont=dict(color=TEXT)))
    fig.add_trace(go.Bar(name="F1 Score", x=names, y=f1s,  marker_color=CYAN, text=[f"{v:.4f}" for v in f1s],  textposition="outside", textfont=dict(color=TEXT)))
    fig.update_layout(**dark_layout(), barmode="group", legend=dict(bgcolor="rgba(0,0,0,0)"), yaxis_range=[0,0.75])
    return fig

if __name__ == "__main__":
    print("Starting HealthPulse Dashboard...")
    print("Open your browser at: http://127.0.0.1:8050")
    app.run(debug=True)
