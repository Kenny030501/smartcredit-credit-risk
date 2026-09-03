"""Interactive SmartCredit dashboard for probability calibration and credit policy."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from smartcredit.modeling import build_risk_deciles
from smartcredit.policy import (
    build_rejection_capacity_table,
    build_three_band_table,
    build_threshold_policy_table,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PATH = (
    ROOT / "data" / "processed" / "calibration_policy" / "validation_calibration_comparison.parquet"
)
METRICS_PATH = ROOT / "reports" / "modeling" / "calibration_policy_metrics.json"
REPORT_PATH = ROOT / "reports" / "modeling" / "calibration_policy_report.md"

PROBABILITY_COLUMNS = {
    "原始调优XGBoost概率（当前选择）": "RAW_PROBABILITY",
    "Platt校准概率": "PLATT_PROBABILITY",
    "Isotonic校准概率": "ISOTONIC_PROBABILITY",
}
METHOD_NAMES = {
    "raw": "原始概率",
    "platt": "Platt",
    "isotonic": "Isotonic",
}


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> tuple[pd.DataFrame, dict[str, object]]:
    """Load validated holdout predictions and calibration metrics once per session."""
    if not VALIDATION_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError(
            "缺少校准策略产物，请先运行 scripts/calibrate_and_simulate_policy.py"
        )
    predictions = pd.read_parquet(VALIDATION_PATH).sort_values("SK_ID_CURR")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    required = {
        "SK_ID_CURR",
        "TARGET",
        "RAW_PROBABILITY",
        "PLATT_PROBABILITY",
        "ISOTONIC_PROBABILITY",
    }
    if not required.issubset(predictions.columns):
        raise ValueError(f"验证预测缺少字段：{sorted(required - set(predictions.columns))}")
    return predictions, metrics


def percentage(value: float) -> str:
    """Format a ratio as a percentage with two decimal places."""
    return "—" if pd.isna(value) else f"{value:.2%}"


def strategy_row(
    target: pd.Series,
    probability: np.ndarray,
    mode: str,
    control_value: float,
    performing_margin: float,
    loss_given_default: float,
) -> pd.Series:
    """Compute one policy scenario from the active sidebar controls."""
    if mode == "按拒绝比例":
        table = build_rejection_capacity_table(
            target,
            probability,
            [control_value],
            performing_margin,
            loss_given_default,
        )
    else:
        table = build_threshold_policy_table(
            target,
            probability,
            [control_value],
            performing_margin,
            loss_given_default,
        )
    return table.iloc[0]


def rejected_mask(probability: np.ndarray, mode: str, row: pd.Series) -> np.ndarray:
    """Reconstruct applicant-level approval decisions for the active scenario."""
    if mode == "按概率阈值":
        return probability >= float(row["threshold"])
    rejection_count = int(row["rejection_count"])
    if rejection_count == 0:
        return np.zeros(len(probability), dtype=bool)
    rank = pd.Series(probability).rank(method="first", ascending=False).to_numpy()
    return rank <= rejection_count


def build_tradeoff_chart(capacity: pd.DataFrame, selected_rejection_rate: float) -> go.Figure:
    """Plot approved bad rate and bad-sample capture rate together."""
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=capacity["rejection_rate"],
            y=capacity["approved_bad_rate"],
            name="审批客群坏样本率",
            mode="lines+markers",
            line={"color": "#4477AA", "width": 3},
            marker={"symbol": "circle", "size": 7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=capacity["rejection_rate"],
            y=capacity["bad_capture_rate"],
            name="坏样本捕获率",
            mode="lines+markers",
            line={"color": "#AA3377", "width": 3, "dash": "dash"},
            marker={"symbol": "square", "size": 7},
        )
    )
    figure.add_vline(
        x=selected_rejection_rate,
        line={"color": "#222222", "dash": "dot"},
        annotation_text="当前策略",
    )
    figure.update_layout(
        title="拒绝比例与风险结果",
        xaxis_title="拒绝率",
        yaxis_title="比例",
        xaxis={"tickformat": ".0%"},
        yaxis={"tickformat": ".0%"},
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


def build_value_chart(capacity: pd.DataFrame, selected_rejection_rate: float) -> go.Figure:
    """Plot normalized value per application under the current economic assumptions."""
    figure = go.Figure(
        go.Scatter(
            x=capacity["rejection_rate"],
            y=capacity["normalized_value_per_applicant"],
            name="单位申请价值",
            mode="lines+markers",
            line={"color": "#EE7733", "width": 3, "dash": "dot"},
            marker={"symbol": "triangle-up", "size": 8},
        )
    )
    figure.add_vline(
        x=selected_rejection_rate,
        line={"color": "#222222", "dash": "dot"},
        annotation_text="当前策略",
    )
    figure.update_layout(
        title="假设条件下的单位申请价值",
        xaxis_title="拒绝率",
        yaxis_title="归一化价值",
        xaxis={"tickformat": ".0%"},
        hovermode="x unified",
    )
    return figure


def build_decile_chart(deciles: pd.DataFrame) -> go.Figure:
    """Plot and label the observed bad rate for each risk decile."""
    figure = go.Figure(
        go.Bar(
            x=deciles["risk_decile"].astype(str),
            y=deciles["observed_bad_rate"],
            marker={"color": "#4477AA", "line": {"color": "#222222", "width": 0.8}},
            text=[f"{value:.1%}" for value in deciles["observed_bad_rate"]],
            textposition="outside",
            name="真实坏样本率",
        )
    )
    figure.update_layout(
        title="风险十分位真实坏样本率",
        xaxis_title="风险组（1=预测风险最高）",
        yaxis_title="真实坏样本率",
        yaxis={"tickformat": ".0%"},
        showlegend=False,
    )
    return figure


def build_calibration_chart(
    target: pd.Series,
    predictions: pd.DataFrame,
) -> go.Figure:
    """Compare three probability estimates with observed rates by risk decile."""
    styles = {
        "RAW_PROBABILITY": ("原始概率", "#4477AA", "solid", "circle"),
        "PLATT_PROBABILITY": ("Platt", "#EE7733", "dash", "square"),
        "ISOTONIC_PROBABILITY": ("Isotonic", "#AA3377", "dot", "triangle-up"),
    }
    figure = go.Figure()
    for column, (label, color, dash, marker) in styles.items():
        deciles = build_risk_deciles(target, predictions[column].to_numpy())
        figure.add_trace(
            go.Scatter(
                x=deciles["average_predicted_probability"],
                y=deciles["observed_bad_rate"],
                name=label,
                mode="lines+markers",
                line={"color": color, "width": 3, "dash": dash},
                marker={"symbol": marker, "size": 8},
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 0.35],
            y=[0, 0.35],
            name="理想校准线",
            mode="lines",
            line={"color": "#666666", "dash": "dashdot"},
        )
    )
    figure.update_layout(
        title="概率校准比较",
        xaxis_title="平均预测概率",
        yaxis_title="真实坏样本率",
        xaxis={"tickformat": ".0%", "range": [0, 0.35]},
        yaxis={"tickformat": ".0%", "range": [0, 0.35]},
        legend={"orientation": "h", "y": 1.12},
    )
    return figure


st.set_page_config(page_title="SmartCredit策略看板", page_icon="🏦", layout="wide")
st.title("🏦 SmartCredit｜授信策略模拟看板")
st.caption("基于Home Credit验证集的教学型风险策略模拟；不构成真实授信建议。")

try:
    data, manifest = load_dashboard_data()
except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
    st.error(str(error))
    st.stop()

target = data["TARGET"].astype("int8")
st.sidebar.header("策略设置")
probability_label = st.sidebar.selectbox("概率来源", list(PROBABILITY_COLUMNS))
probability_column = PROBABILITY_COLUMNS[probability_label]
probability = data[probability_column].to_numpy()
mode = st.sidebar.radio("决策方式", ["按拒绝比例", "按概率阈值"])

if mode == "按拒绝比例":
    control_value = st.sidebar.slider("拒绝风险最高人群", 0, 50, 15, 1) / 100
else:
    control_value = st.sidebar.slider("拒绝概率阈值", 0.01, 0.50, 0.15, 0.01)

st.sidebar.subheader("经济情景假设")
performing_margin = st.sidebar.slider("正常贷款净贡献率", 0.01, 0.20, 0.08, 0.01)
loss_given_default = st.sidebar.slider("坏样本损失率（LGD）", 0.10, 0.90, 0.50, 0.05)
st.sidebar.caption("假设每笔EAD=1；拒绝后收入、损失和审核成本均为0。")

current = strategy_row(
    target,
    probability,
    mode,
    control_value,
    performing_margin,
    loss_given_default,
)
current_threshold = float(current["threshold"])
threshold_text = "∞" if math.isinf(current_threshold) else f"{current_threshold:.2%}"

columns = st.columns(6)
columns[0].metric("审批率", percentage(float(current["approval_rate"])))
columns[1].metric("拒绝率", percentage(float(current["rejection_rate"])))
columns[2].metric("对应阈值", threshold_text)
columns[3].metric("审批客群坏样本率", percentage(float(current["approved_bad_rate"])))
columns[4].metric("坏样本捕获率", percentage(float(current["bad_capture_rate"])))
columns[5].metric("误拒正常客户", f"{int(current['good_rejection_count']):,}")

st.info(
    f"当前单位申请价值为 **{float(current['normalized_value_per_applicant']):.4f}**。"
    "该数字只在侧边栏假设下有效，不能解释为实际利润率。"
)

strategy_tab, segmentation_tab, calibration_tab, audit_tab = st.tabs(
    ["策略权衡", "风险分层", "概率校准", "数据与边界"]
)

with strategy_tab:
    capacity = build_rejection_capacity_table(
        target,
        probability,
        [value / 100 for value in range(51)],
        performing_margin,
        loss_given_default,
    )
    chart_columns = st.columns(2)
    chart_columns[0].plotly_chart(
        build_tradeoff_chart(capacity, float(current["rejection_rate"])),
        use_container_width=True,
    )
    chart_columns[1].plotly_chart(
        build_value_chart(capacity, float(current["rejection_rate"])),
        use_container_width=True,
    )

    st.subheader("固定拒绝容量情景")
    display_capacity = build_rejection_capacity_table(
        target,
        probability,
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        performing_margin,
        loss_given_default,
    )[
        [
            "rejection_rate",
            "threshold",
            "approval_rate",
            "approved_bad_rate",
            "bad_capture_rate",
            "good_rejection_count",
            "normalized_value_per_applicant",
        ]
    ].copy()
    display_capacity.columns = [
        "拒绝率",
        "概率阈值",
        "审批率",
        "审批客群坏样本率",
        "坏样本捕获率",
        "误拒正常客户",
        "单位申请价值",
    ]
    for column in ["拒绝率", "概率阈值", "审批率", "审批客群坏样本率", "坏样本捕获率"]:
        display_capacity[column] = display_capacity[column].map(
            lambda value: "∞" if math.isinf(value) else f"{value:.2%}"
        )
    st.dataframe(display_capacity, hide_index=True, use_container_width=True)

with segmentation_tab:
    deciles = build_risk_deciles(target, probability)
    st.plotly_chart(build_decile_chart(deciles), use_container_width=True)
    st.subheader("三段式风险策略示例")
    band_table = build_three_band_table(target, probability)
    display_bands = band_table[
        [
            "risk_band",
            "applicant_count",
            "population_share",
            "observed_bad_rate",
            "average_predicted_probability",
        ]
    ].copy()
    display_bands.columns = ["风险带", "人数", "占比", "真实坏样本率", "平均预测概率"]
    for column in ["占比", "真实坏样本率", "平均预测概率"]:
        display_bands[column] = display_bands[column].map(lambda value: f"{value:.2%}")
    st.dataframe(display_bands, hide_index=True, use_container_width=True)
    st.caption("示例定义：最高风险10%为拒绝候选，随后20%人工审核，其余70%自动批准。")

with calibration_tab:
    selected_method = manifest["selected_method"]
    st.success(
        f"开发集交叉校准最终选择：{METHOD_NAMES[selected_method]}。"
        "没有满足实质改善条件时，优先保留原始概率。"
    )
    st.plotly_chart(build_calibration_chart(target, data), use_container_width=True)
    calibration_rows = []
    for method, values in manifest["calibration_metrics"].items():
        calibration_rows.append(
            {
                "方法": METHOD_NAMES[method],
                "ROC-AUC": values["roc_auc"],
                "Brier": values["brier_score"],
                "Log Loss": values["log_loss"],
                "十分位校准误差": values["decile_calibration_error"],
            }
        )
    st.dataframe(pd.DataFrame(calibration_rows), hide_index=True, use_container_width=True)

with audit_tab:
    st.warning(
        "TARGET是比赛定义的还款困难，不等同于监管违约。数据缺少真实利率、期限、资金成本、"
        "EAD、LGD、回收率和运营成本；页面中的经济结果只是情景模拟。"
    )
    st.markdown(
        f"- 验证申请人：**{len(data):,}**\n"
        f"- 真实坏样本率：**{target.mean():.2%}**\n"
        f"- 当前概率字段：`{probability_column}`\n"
        f"- 本地报告：`{REPORT_PATH.relative_to(ROOT)}`"
    )
    mask = rejected_mask(probability, mode, current)
    decisions = pd.DataFrame(
        {
            "SK_ID_CURR": data["SK_ID_CURR"],
            "TARGET": target,
            "PREDICTED_PROBABILITY": probability,
            "DECISION": np.where(mask, "拒绝候选", "批准候选"),
        }
    )
    st.download_button(
        "下载当前验证集策略结果",
        decisions.to_csv(index=False).encode("utf-8-sig"),
        file_name="smartcredit_validation_policy.csv",
        mime="text/csv",
    )
    st.dataframe(decisions.head(100), hide_index=True, use_container_width=True)
