
import re
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="크록스 주차별 광고 리포트",
    layout="wide"
)

# =========================
# 기본 설정
# =========================
DEFAULT_NUMERIC_COLS = [
    "노출수", "클릭수", "광고비", "총 주문수(1일)",
    "총 판매수량(1일)", "총 전환매출액(1일)", "잔여수"
]

METRIC_LABELS = {
    "노출수": "노출수",
    "클릭수": "클릭수",
    "광고비": "광고비",
    "총 주문수(1일)": "주문수",
    "총 판매수량(1일)": "판매수량",
    "총 전환매출액(1일)": "전환매출액",
    "CTR": "CTR",
    "CPC": "CPC",
    "CVR": "CVR",
    "ROAS": "ROAS",
}


# =========================
# 유틸 함수
# =========================
def safe_div(numerator, denominator):
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0)
    return np.where(denominator == 0, 0, numerator / denominator)


def format_number(value, metric=None):
    if pd.isna(value):
        return "-"
    if metric in ["CTR", "CVR", "ROAS", "증감률"]:
        return f"{value:,.1f}%"
    if metric == "CPC":
        return f"{value:,.0f}원"
    return f"{value:,.0f}"


def short_campaign_name(name):
    """화면 표시용 캠페인명: 두 번째 언더바(_) 전까지만 표시합니다.
    예: CR_AI스마트광고_바야밴드클로그 -> CR_AI스마트광고
    집계는 원본 캠페인명으로 유지하고, 화면 제목/선택값만 이 함수로 표시합니다.
    """
    if pd.isna(name):
        return "UNKNOWN"
    parts = str(name).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(name)


def extract_size(product_name):
    """
    광고집행 상품명 안의 옵션 문자열에서 사이즈 추출
    예시:
    - M10W12(280)
    - M4W6(230)
    - 270
    - M12(300)
    """
    if pd.isna(product_name):
        return "사이즈없음"

    text = str(product_name)

    # 1) 괄호 안 3자리 사이즈 우선 추출: M10W12(280)
    match = re.search(r"\((\d{3})\)", text)
    if match:
        return match.group(1)

    # 2) 콤마로 나뉜 옵션 중 220~320 범위의 3자리 숫자 추출
    parts = [p.strip() for p in text.split(",")]
    for part in parts:
        if re.fullmatch(r"\d{3}", part):
            size = int(part)
            if 100 <= size <= 400:
                return str(size)

    # 3) 전체 텍스트에서 220~320 범위 숫자 추출
    nums = re.findall(r"\b(\d{3})\b", text)
    for num in nums:
        size = int(num)
        if 100 <= size <= 400:
            return str(size)

    return "사이즈없음"




def extract_product_display_name(product_name):
    """
    광고집행 상품명에서 보고서용 대표 상품명을 추출합니다.
    예: 크록스 본사 클레오 2,에스프레소 / 브론즈,에스프레소 / 브론즈,W8(250),W8(250)
        -> 클레오2, 에스프레소
    """
    if pd.isna(product_name):
        return "상품명없음"

    text = str(product_name).strip()
    if not text:
        return "상품명없음"

    # 슬래시 뒤는 색상/사이즈 옵션이 반복되는 경우가 많아 첫 구간만 사용
    first_part = re.split(r"\s*/\s*", text)[0].strip()

    # 앞쪽 브랜드/판매처 prefix 제거
    first_part = re.sub(r"^\s*크록스\s*본사\s*", "", first_part)
    first_part = re.sub(r"^\s*크록스\s*", "", first_part)
    first_part = re.sub(r"^\s*본사\s*", "", first_part)

    parts = [x.strip() for x in first_part.split(",") if str(x).strip()]

    if len(parts) >= 2:
        product = parts[0]
        option = parts[1]
        # 클레오 2 -> 클레오2 처럼 모델명 숫자 간격만 정리
        product = re.sub(r"([가-힣A-Za-z])\s+(\d)", r"\1\2", product)
        option = re.sub(r"([가-힣A-Za-z])\s+(\d)", r"\1\2", option)
        return f"{product}, {option}"

    product = parts[0] if parts else first_part
    product = re.sub(r"([가-힣A-Za-z])\s+(\d)", r"\1\2", product)
    return product.strip() or "상품명없음"


def representative_product_by_color(base_df, color_list, metric="총 판매수량(1일)"):
    """컬러별로 판매량이 가장 큰 상품표시명을 대표 상품명으로 사용합니다."""
    result = {}
    for color in list(color_list):
        target = base_df[base_df["color"] == color].copy()
        if target.empty or "상품표시명" not in target.columns:
            result[color] = ""
            continue
        rep = (
            target.groupby("상품표시명", dropna=False)[metric]
            .sum()
            .sort_values(ascending=False)
        )
        result[color] = str(rep.index[0]) if len(rep) else ""
    return result


def color_label(color_name, product_name=""):
    color_name = str(color_name)
    product_name = str(product_name).strip()
    if product_name and product_name != "상품명없음":
        return f"{color_name} ({product_name})"
    return color_name


def normalize_df(df, week_label):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required_cols = [
        "날짜", "캠페인명", "광고집행 상품명", "광고집행 옵션ID",
        "노출수", "클릭수", "광고비", "총 주문수(1일)",
        "총 판매수량(1일)", "총 전환매출액(1일)", "color", "잔여수"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"필수 컬럼이 없습니다: {missing}")
        st.stop()

    for col in DEFAULT_NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str), format="%Y%m%d", errors="coerce")
    df["주차"] = week_label
    df["캠페인명_축약"] = df["캠페인명"].apply(short_campaign_name)
    df["color"] = df["color"].astype(str).str.strip()
    df["상품사이즈"] = df["광고집행 상품명"].apply(extract_size)
    df["상품표시명"] = df["광고집행 상품명"].apply(extract_product_display_name)
    df["옵션키"] = df["광고집행 옵션ID"].astype(str) + "_" + df["color"].astype(str) + "_" + df["상품사이즈"].astype(str)

    return df


def add_kpis(summary):
    summary = summary.copy()
    summary["CTR"] = safe_div(summary["클릭수"], summary["노출수"]) * 100
    summary["CPC"] = safe_div(summary["광고비"], summary["클릭수"])
    summary["CPA"] = safe_div(summary["광고비"], summary["총 주문수(1일)"])
    summary["CVR"] = safe_div(summary["총 주문수(1일)"], summary["클릭수"]) * 100
    summary["ROAS"] = safe_div(summary["총 전환매출액(1일)"], summary["광고비"]) * 100
    return summary


def aggregate(df, group_cols):
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg({
            "노출수": "sum",
            "클릭수": "sum",
            "광고비": "sum",
            "총 주문수(1일)": "sum",
            "총 판매수량(1일)": "sum",
            "총 전환매출액(1일)": "sum",
        })
        .reset_index()
    )
    return add_kpis(summary)


def compare_weeks(summary, key_cols, metric_cols):
    prev = summary[summary["주차"] == "전전주"].copy()
    curr = summary[summary["주차"] == "전주"].copy()

    # 전체값 비교처럼 key_cols가 없는 경우 pandas merge(on=[])에서 IndexError가 발생합니다.
    # 이때는 전전주/전주를 각각 1행 요약으로 만든 뒤 가로 결합합니다.
    if not key_cols:
        prev_row = prev[metric_cols].sum(numeric_only=True).to_frame().T
        curr_row = curr[metric_cols].sum(numeric_only=True).to_frame().T

        # 비율 지표는 단순 sum이 아니라 이미 aggregate에서 계산된 1행 값을 사용합니다.
        for rate_col in ["CTR", "CPC", "CVR", "ROAS"]:
            if rate_col in metric_cols:
                prev_row[rate_col] = prev[rate_col].iloc[0] if len(prev) else 0
                curr_row[rate_col] = curr[rate_col].iloc[0] if len(curr) else 0

        prev_row = prev_row.rename(columns={c: f"{c}_전전주" for c in metric_cols})
        curr_row = curr_row.rename(columns={c: f"{c}_전주" for c in metric_cols})
        merged = pd.concat([prev_row.reset_index(drop=True), curr_row.reset_index(drop=True)], axis=1).fillna(0)
    else:
        prev = prev[key_cols + metric_cols].rename(columns={c: f"{c}_전전주" for c in metric_cols})
        curr = curr[key_cols + metric_cols].rename(columns={c: f"{c}_전주" for c in metric_cols})
        merged = pd.merge(prev, curr, on=key_cols, how="outer").fillna(0)

    for c in metric_cols:
        merged[f"{c}_증감"] = merged[f"{c}_전주"] - merged[f"{c}_전전주"]
        merged[f"{c}_증감률"] = np.where(
            merged[f"{c}_전전주"] == 0,
            0,
            merged[f"{c}_증감"] / merged[f"{c}_전전주"] * 100
        )

    return merged


def style_compare_table(df):
    """캠페인별 비교표 스타일: 증감/증감률은 증가 빨강, 하락 파랑."""
    number_fmt = {}
    for col in df.columns:
        if col in ["캠페인명", "캠페인명_축약"]:
            continue
        if "증감률" in col or any(x in col for x in ["CTR", "CVR", "ROAS"]):
            number_fmt[col] = "{:,.2f}%"
        elif any(x in col for x in ["노출수", "클릭수", "광고비", "주문수", "판매수량", "전환매출액", "CPC", "CPA"]):
            number_fmt[col] = "{:,.0f}"

    def color_delta(value):
        try:
            v = float(value)
            if v > 0:
                return "color:red; font-weight:700;"
            if v < 0:
                return "color:blue; font-weight:700;"
        except Exception:
            pass
        return "color:black;"

    delta_cols = [c for c in df.columns if "_증감" in c or "_증감률" in c]

    return (
        df.style
        .format(number_fmt)
        .map(color_delta, subset=delta_cols)
        .set_properties(**{
            "text-align": "center",
            "border": "1px solid #d0d0d0",
            "font-size": "13px",
        })
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#bfbfbf"),
                ("color", "black"),
                ("font-weight", "700"),
                ("border", "1px solid #555"),
                ("text-align", "center"),
                ("white-space", "nowrap"),
            ]},
            {"selector": "td", "props": [("white-space", "nowrap")]},
        ])
    )

def format_period_label(df):
    start = pd.to_datetime(df["날짜"]).min()
    end = pd.to_datetime(df["날짜"]).max()
    if pd.isna(start) or pd.isna(end):
        return "-"
    return f"{start.month}.{start.day}~{end.month}.{end.day}"


def make_total_visual_table(total_summary, prev_df, curr_df):
    """이미지 예시처럼 전체 지표를 행=기간/증감, 열=지표 구조로 만듭니다."""
    columns = [
        ("노출수", "노출수"),
        ("클릭수", "클릭수"),
        ("클릭률(%)", "CTR"),
        ("CPC(원)", "CPC"),
        ("총비용(원)", "광고비"),
        ("총전환수", "총 주문수(1일)"),
        ("총전환매출(원)", "총 전환매출액(1일)"),
        ("CPA(원)", "CPA"),
        ("전환율", "CVR"),
        ("ROAS", "ROAS"),
    ]

    def get_week_row(summary, week):
        target = summary[summary["주차"] == week]
        if len(target) > 0:
            return target.iloc[0]
        return pd.Series({col: 0 for _, col in columns})

    prev = get_week_row(total_summary, "전전주")
    curr = get_week_row(total_summary, "전주")

    prev_label = format_period_label(prev_df)
    curr_label = format_period_label(curr_df)

    row_curr = {"집행기간": curr_label}
    row_diff = {"집행기간": "증감분"}
    row_rate = {"집행기간": "증감률"}
    row_prev = {"집행기간": prev_label}

    for label, col in columns:
        prev_val = float(prev[col]) if col in prev.index and pd.notna(prev[col]) else 0
        curr_val = float(curr[col]) if col in curr.index and pd.notna(curr[col]) else 0
        diff = curr_val - prev_val
        rate = 0 if prev_val == 0 else diff / prev_val * 100
        row_curr[label] = curr_val
        row_diff[label] = diff
        row_rate[label] = rate
        row_prev[label] = prev_val

    return pd.DataFrame([row_curr, row_diff, row_rate, row_prev])


def make_campaign_total_visual_table(campaign_name, campaign_summary, prev_df, curr_df):
    """캠페인 1개 기준 전전주 vs 전주 전체값 비교표를 생성합니다."""
    target_summary = campaign_summary[campaign_summary["캠페인명"] == campaign_name].copy()
    target_prev = prev_df[prev_df["캠페인명"] == campaign_name].copy()
    target_curr = curr_df[curr_df["캠페인명"] == campaign_name].copy()
    table = make_total_visual_table(target_summary, target_prev, target_curr)
    table.insert(0, "캠페인명", campaign_name)
    return table


def make_filtered_visual_table(base_df, filter_col, filter_value):
    """특정 컬러/사이즈 등 1개 값 기준으로 전체값 비교표 형태를 생성합니다."""
    target = base_df[base_df[filter_col] == filter_value].copy()
    target_summary = aggregate(target, ["주차"])
    target_prev = target[target["주차"] == "전전주"].copy()
    target_curr = target[target["주차"] == "전주"].copy()
    return make_total_visual_table(target_summary, target_prev, target_curr)


def make_color_compare_table(campaign_df, color_name):
    table = make_filtered_visual_table(campaign_df, "color", color_name)
    table.insert(0, "color", color_name)
    return table


def style_total_visual_table(df):
    """전체값 비교표 스타일: 이미지 형태 + 증감분/증감률 색상."""
    percent_cols = ["클릭률(%)", "전환율", "ROAS"]

    def fmt_value(value, col, row_type):
        if col == "집행기간":
            return value
        try:
            v = float(value)
        except Exception:
            return value
        if row_type == "증감률":
            return f"{v:,.2f}%"
        if col in percent_cols:
            return f"{v:,.2f}%"
        return f"{v:,.0f}"

    def to_number(value):
        try:
            return float(str(value).replace(",", "").replace("%", ""))
        except Exception:
            return 0

    # pandas 2.3+/Python 3.14 환경에서는 숫자 dtype 컬럼에 문자열 포맷값을 바로 넣으면 TypeError가 발생할 수 있어 object로 변환합니다.
    formatted = df.copy().astype(object)
    for idx in formatted.index:
        row_type = formatted.loc[idx, "집행기간"]
        for col in formatted.columns:
            formatted.loc[idx, col] = fmt_value(formatted.loc[idx, col], col, row_type)

    def apply_color(row):
        styles = []
        row_type = row.get("집행기간", "")
        for col, value in row.items():
            base = "border:1px solid #999; text-align:center; white-space:nowrap;"

            if col == "집행기간":
                base += " font-weight:700; background-color:#eeeeee;"
            # 증감률 행 배경은 흰색으로 유지합니다.

            if row_type in ["증감분", "증감률"] and col != "집행기간":
                v = to_number(value)
                if v > 0:
                    base += " color:red; font-weight:700;"
                elif v < 0:
                    base += " color:blue; font-weight:700;"
                else:
                    base += " color:black;"
            styles.append(base)
        return styles

    return (
        formatted.style
        .apply(apply_color, axis=1)
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#bfbfbf"),
                ("color", "black"),
                ("font-weight", "700"),
                ("border", "1px solid #555"),
                ("text-align", "center"),
                ("white-space", "nowrap"),
            ]},
            {"selector": "td", "props": [("font-size", "13px")]},
        ])
    )



def make_daily_performance_table(base_df):
    """전주 데이터 기준 날짜별 성과표를 생성합니다."""
    if base_df.empty:
        return pd.DataFrame()

    daily = aggregate(base_df, ["날짜"]).sort_values("날짜").copy()

    # 합계 행은 비율 지표를 단순 합산하지 않고 전체 합계 기준으로 재계산
    total_raw = {
        "날짜": "총합계",
        "노출수": base_df["노출수"].sum(),
        "클릭수": base_df["클릭수"].sum(),
        "광고비": base_df["광고비"].sum(),
        "총 주문수(1일)": base_df["총 주문수(1일)"].sum(),
        "총 판매수량(1일)": base_df["총 판매수량(1일)"].sum(),
        "총 전환매출액(1일)": base_df["총 전환매출액(1일)"].sum(),
    }
    total = add_kpis(pd.DataFrame([total_raw]))

    result = pd.concat([daily, total], ignore_index=True)

    result["날짜"] = result["날짜"].apply(
        lambda x: x if str(x) == "총합계" else pd.to_datetime(x).strftime("%Y%m%d")
    )

    result = result[[
        "날짜", "광고비", "노출수", "클릭수", "CPC", "CTR",
        "총 주문수(1일)", "CVR", "총 전환매출액(1일)", "ROAS"
    ]]

    result = result.rename(columns={
        "날짜": "행 레이블",
        "광고비": "합계 : 광고비",
        "노출수": "합계 : 노출수",
        "클릭수": "합계 : 클릭수",
        "CPC": "합계 : CPC",
        "CTR": "합계 : CTR",
        "총 주문수(1일)": "합계 : 총 주문수(1일)",
        "CVR": "합계 : CVR",
        "총 전환매출액(1일)": "합계 : 총 전환매출",
        "ROAS": "합계 : ROAS",
    })
    return result


def style_daily_performance_table(df):
    """날짜별 성과표 스타일."""
    if df.empty:
        return df

    fmt = {}
    for col in df.columns:
        if col == "행 레이블":
            continue
        if col in ["합계 : CTR", "합계 : CVR", "합계 : ROAS"]:
            fmt[col] = "{:,.2f}%"
        elif col == "합계 : CPC":
            fmt[col] = "{:,.0f}"
        else:
            fmt[col] = "{:,.0f}"

    def row_style(row):
        is_total = str(row.get("행 레이블", "")) == "총합계"
        styles = []
        for _ in row:
            base = "border:1px solid #999; text-align:right; white-space:nowrap;"
            if is_total:
                base += " background-color:#eeeeee; font-weight:700;"
            styles.append(base)
        return styles

    return (
        df.style
        .format(fmt)
        .apply(row_style, axis=1)
        .set_properties(subset=["행 레이블"], **{"text-align": "left", "font-weight": "700"})
        .set_table_styles([
            {"selector": "th", "props": [
                ("background-color", "#4472C4"),
                ("color", "white"),
                ("font-weight", "700"),
                ("border", "1px solid #555"),
                ("text-align", "center"),
                ("white-space", "nowrap"),
            ]},
            {"selector": "td", "props": [("font-size", "13px")]},
        ])
    )

def make_download(df, filename):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    st.download_button(
        label=f"{filename} 다운로드",
        data=buffer.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )



# =========================
# 신규고객확보캠페인 전용 함수
# =========================
def to_numeric_clean(series):
    """쉼표, %, - 문자가 섞인 숫자 컬럼을 안전하게 숫자로 변환합니다."""
    return (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("-", "0", regex=False)
        .str.strip()
        .replace("", "0")
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
    )


def short_new_customer_campaign_name(name):
    """신규고객확보 캠페인명 표시용 정리."""
    if pd.isna(name):
        return "UNKNOWN"
    text = str(name).strip()
    # 슬래시 뒤 변경 이력 제거
    text = text.split("/")[0].strip()
    # 날짜 토큰 제거
    text = re.sub(r"_?\d{8}", "", text)
    text = re.sub(r"[-_]\d{4}", "", text)
    text = text.replace("신규고객확보캠페인", "신규고객확보")
    return text.strip(" _-") or str(name)


def normalize_new_customer_df(raw_df, prev_period_df, curr_period_df):
    """신규고객확보 raw를 전전주/전주 비교용 공통 KPI 구조로 변환합니다."""
    df_new = raw_df.copy()
    df_new.columns = [str(c).strip() for c in df_new.columns]

    required_cols = [
        "날짜", "캠페인 이름", "노출수", "클릭수", "집행 광고비",
        "신규 구매 고객 수", "첫구매를 통한 광고 전환 매출"
    ]
    missing = [c for c in required_cols if c not in df_new.columns]
    if missing:
        st.error(f"신규고객확보 raw 필수 컬럼이 없습니다: {missing}")
        st.stop()

    df_new["날짜"] = pd.to_datetime(df_new["날짜"].astype(str), format="%Y%m%d", errors="coerce")
    df_new["캠페인명"] = df_new["캠페인 이름"].astype(str).str.strip()
    df_new["캠페인명_축약"] = df_new["캠페인명"].apply(short_new_customer_campaign_name)

    numeric_map = {
        "노출수": "노출수",
        "클릭수": "클릭수",
        "집행 광고비": "광고비",
        "신규 구매 고객 수": "총 주문수(1일)",
        "첫구매를 통한 광고 전환 매출": "총 전환매출액(1일)",
    }
    for src_col, dst_col in numeric_map.items():
        df_new[dst_col] = to_numeric_clean(df_new[src_col])

    # 신규고객확보 raw에는 판매수량 개념이 없으므로 신규 구매 고객 수와 동일하게 둡니다.
    df_new["총 판매수량(1일)"] = df_new["총 주문수(1일)"]

    prev_start = pd.to_datetime(prev_period_df["날짜"]).min()
    prev_end = pd.to_datetime(prev_period_df["날짜"]).max()
    curr_start = pd.to_datetime(curr_period_df["날짜"]).min()
    curr_end = pd.to_datetime(curr_period_df["날짜"]).max()

    df_new["주차"] = np.where(
        df_new["날짜"].between(prev_start, prev_end),
        "전전주",
        np.where(df_new["날짜"].between(curr_start, curr_end), "전주", "제외")
    )
    df_new = df_new[df_new["주차"].isin(["전전주", "전주"])].copy()

    return df_new


def make_new_customer_visual_table(summary, prev_base_df, curr_base_df):
    """신규고객확보용 전체값 비교표."""
    table = make_total_visual_table(summary, prev_base_df, curr_base_df)
    rename_map = {
        "총전환수": "신규구매고객수",
        "총전환매출(원)": "첫구매 전환매출(원)",
        "CPA(원)": "신규고객당 비용(원)",
        "전환율": "클릭 대비 전환율",
    }
    return table.rename(columns=rename_map)


def make_new_customer_daily_table(base_df):
    """신규고객확보 전주 날짜별 성과표."""
    daily = make_daily_performance_table(base_df)
    if daily.empty:
        return daily
    return daily.rename(columns={
        "합계 : 총 주문수(1일)": "합계 : 신규 구매 고객 수",
        "합계 : 총 전환매출": "합계 : 첫구매 전환매출",
    })


def render_new_customer_dashboard(new_customer_df):
    """신규고객확보캠페인 탭 화면."""
    st.subheader("신규고객확보캠페인")
    st.caption("신규고객확보 raw는 기존 전전주/전주 raw의 날짜 기간과 동일하게 필터링하여 비교합니다.")

    if new_customer_df.empty:
        st.warning("전전주/전주 기간에 해당하는 신규고객확보 데이터가 없습니다.")
        return

    new_summary = aggregate(new_customer_df, ["주차", "캠페인명_축약"])
    new_compare = compare_weeks(new_summary, ["캠페인명_축약"], metric_cols)
    new_compare = new_compare.sort_values("총 전환매출액(1일)_전주", ascending=False).reset_index(drop=True)
    new_campaign_order = new_compare["캠페인명_축약"].tolist()

    st.markdown("#### 캠페인별 전전주 vs 전주 비교")
    new_all_tables = []
    for campaign_name in new_campaign_order:
        target_df = new_customer_df[new_customer_df["캠페인명_축약"] == campaign_name].copy()
        target_summary = aggregate(target_df, ["주차"])
        target_prev = target_df[target_df["주차"] == "전전주"].copy()
        target_curr = target_df[target_df["주차"] == "전주"].copy()
        table = make_new_customer_visual_table(target_summary, target_prev, target_curr)
        table.insert(0, "캠페인명", clean_tab_name(campaign_name))
        new_all_tables.append(table)

        st.markdown(f"##### {clean_tab_name(campaign_name)}")
        display_table = table.drop(columns=[c for c in ["캠페인명", "상품명"] if c in table.columns])
        st.dataframe(style_total_visual_table(display_table), use_container_width=True, hide_index=True)

        st.markdown("###### 전주 날짜별 데이터")
        daily_table = make_new_customer_daily_table(target_curr)
        if daily_table.empty:
            st.warning("전주 날짜별 데이터가 없습니다.")
        else:
            st.dataframe(style_daily_performance_table(daily_table), use_container_width=True, hide_index=True)
            make_download(daily_table, f"신규고객확보_{clean_tab_name(campaign_name)}_전주_날짜별성과.xlsx")

    if new_all_tables:
        new_all = pd.concat(new_all_tables, ignore_index=True)
        make_download(new_all, "신규고객확보_캠페인별_전전주vs전주비교.xlsx")

    with st.expander("신규고객확보 원본 비교 데이터 보기"):
        display_compare = new_compare.copy()
        display_compare["캠페인명"] = display_compare["캠페인명_축약"].apply(clean_tab_name)
        display_compare = display_compare.drop(columns=["캠페인명_축약"])
        cols = ["캠페인명"] + [c for c in display_compare.columns if c != "캠페인명"]
        st.dataframe(style_compare_table(display_compare[cols]), use_container_width=True, hide_index=True)

# =========================
# 사이드바 업로드
# =========================
st.title("크록스 주차별 광고 리포트")
st.caption("전전주 raw와 전주 raw 엑셀을 업로드하면 전체/캠페인/컬러/사이즈/재고 현황을 자동 비교합니다.")

with st.sidebar:
    st.header("파일 업로드")
    prev_file = st.file_uploader("전전주 raw 엑셀", type=["xlsx", "xls"])
    curr_file = st.file_uploader("전주 raw 엑셀", type=["xlsx", "xls"])
    new_customer_file = st.file_uploader("신규고객확보캠페인 raw 엑셀", type=["xlsx", "xls"])

    st.divider()
    top_n_color = st.number_input("컬러 Top N", min_value=3, max_value=20, value=5, step=1)
    top_n_size = st.number_input("사이즈 Top N", min_value=1, max_value=10, value=2, step=1)

    st.caption("기본 기준: 판매수량. 필요 시 아래에서 그래프 지표를 변경할 수 있습니다.")


if not prev_file or not curr_file:
    st.info("왼쪽 사이드바에서 전전주/전주 엑셀 파일을 업로드해주세요.")
    st.stop()


# =========================
# 데이터 로드
# =========================
prev_raw = pd.read_excel(prev_file)
curr_raw = pd.read_excel(curr_file)

prev_df = normalize_df(prev_raw, "전전주")
curr_df = normalize_df(curr_raw, "전주")
df = pd.concat([prev_df, curr_df], ignore_index=True)

new_customer_df = pd.DataFrame()
if new_customer_file:
    new_customer_raw = pd.read_excel(new_customer_file)
    new_customer_df = normalize_new_customer_df(new_customer_raw, prev_df, curr_df)

date_min = df["날짜"].min()
date_max = df["날짜"].max()

st.subheader("데이터 기간")
col1, col2, col3 = st.columns(3)
col1.metric("전전주 기간", f"{prev_df['날짜'].min().date()} ~ {prev_df['날짜'].max().date()}")
col2.metric("전주 기간", f"{curr_df['날짜'].min().date()} ~ {curr_df['날짜'].max().date()}")
col3.metric("전체 row 수", f"{len(df):,}")



# =========================
# 탭 렌더링 함수
# =========================
def clean_tab_name(display_name):
    """탭 제목용: CR_ 같은 앞 prefix 제거"""
    text = str(display_name).strip()
    text = re.sub(r"^(CR|cr|Crocs|크록스)_", "", text)
    text = re.sub(r"^크록스\s*", "", text)
    return text or display_name


def make_tab_label(name, used):
    base = clean_tab_name(name)
    # Streamlit 탭 제목이 너무 길면 가독성이 떨어져 18자 기준으로 줄임
    label = base[:18]
    if label not in used:
        used[label] = 1
        return label
    used[label] += 1
    return f"{label} {used[label]}"


def render_campaign_dashboard(campaign_display_name, campaign_df):
    """축약 캠페인명 기준 1개 탭 화면"""
    st.subheader(f"{clean_tab_name(campaign_display_name)}")

    campaign_prev = campaign_df[campaign_df["주차"] == "전전주"].copy()
    campaign_curr = campaign_df[campaign_df["주차"] == "전주"].copy()
    campaign_summary_tab = aggregate(campaign_df, ["주차"])

    st.markdown("#### 전전주 vs 전주 전체값 비교")
    campaign_total_table = make_total_visual_table(campaign_summary_tab, campaign_prev, campaign_curr)
    st.dataframe(style_total_visual_table(campaign_total_table), use_container_width=True, hide_index=True)

    st.markdown("#### 전주 날짜별 성과")
    campaign_daily_table = make_daily_performance_table(campaign_curr)
    if campaign_daily_table.empty:
        st.warning("전주 날짜별 데이터가 없습니다.")
    else:
        st.dataframe(style_daily_performance_table(campaign_daily_table), use_container_width=True, hide_index=True)
        make_download(campaign_daily_table, f"{clean_tab_name(campaign_display_name)}_전주_날짜별성과.xlsx")

    campaign_total_compare = compare_weeks(campaign_summary_tab, [], metric_cols)
    kpi_cols = st.columns(5)
    for idx, metric in enumerate(["광고비", "총 주문수(1일)", "총 판매수량(1일)", "총 전환매출액(1일)", "ROAS"]):
        curr_value = float(campaign_total_compare[f"{metric}_전주"].iloc[0]) if len(campaign_total_compare) else 0
        delta_value = float(campaign_total_compare[f"{metric}_증감"].iloc[0]) if len(campaign_total_compare) else 0
        delta_pct = float(campaign_total_compare[f"{metric}_증감률"].iloc[0]) if len(campaign_total_compare) else 0
        kpi_cols[idx].metric(
            METRIC_LABELS.get(metric, metric),
            format_number(curr_value, metric),
            f"{format_number(delta_value, metric)} / {delta_pct:,.1f}%"
        )

    st.markdown("#### 상품 컬러 Top 비교")
    color_metric = st.selectbox(
        "컬러 Top 비교 기준",
        ["총 판매수량(1일)", "총 주문수(1일)", "총 전환매출액(1일)", "광고비", "ROAS"],
        index=0,
        key=f"color_metric_{campaign_display_name}"
    )

    color_summary = aggregate(campaign_df, ["주차", "color"])
    top_colors = (
        color_summary.groupby("color")[color_metric]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n_color))
        .index
    )

    if len(top_colors) == 0:
        st.warning("컬러 데이터가 없습니다.")
        return

    color_rep_products = representative_product_by_color(campaign_df, top_colors, metric="총 판매수량(1일)")
    color_top = color_summary[color_summary["color"].isin(top_colors)].copy()
    color_top["컬러_상품명"] = color_top["color"].apply(lambda x: color_label(x, color_rep_products.get(x, "")))

    fig_color = px.bar(
        color_top,
        x="컬러_상품명",
        y=color_metric,
        color="주차",
        barmode="group",
        title=f"{clean_tab_name(campaign_display_name)} 컬러 Top {top_n_color} {METRIC_LABELS.get(color_metric, color_metric)} 비교"
    )
    st.plotly_chart(fig_color, use_container_width=True)

    st.markdown("##### 컬러별 전전주 vs 전주 비교표")
    color_visual_tables = []
    for color_name in list(top_colors):
        display_color_name = color_label(color_name, color_rep_products.get(color_name, ""))
        st.markdown(f"###### {display_color_name}")
        color_table = make_color_compare_table(campaign_df, color_name)
        color_visual_tables.append(color_table)
        st.dataframe(
            style_total_visual_table(color_table.drop(columns=["color"])),
            use_container_width=True,
            hide_index=True
        )

        with st.expander(f"{display_color_name} 전주 일자별 데이터 열기"):
            color_daily_df = campaign_df[(campaign_df["color"] == color_name) & (campaign_df["주차"] == "전주")].copy()
            color_daily_table = make_daily_performance_table(color_daily_df)
            if color_daily_table.empty:
                st.warning("전주 일자별 데이터가 없습니다.")
            else:
                st.dataframe(style_daily_performance_table(color_daily_table), use_container_width=True, hide_index=True)

    if color_visual_tables:
        color_visual_all = pd.concat(color_visual_tables, ignore_index=True)
        make_download(color_visual_all, f"{clean_tab_name(campaign_display_name)}_컬러Top비교표.xlsx")

    st.markdown("#### 상품 컬러별 상품 사이즈 Top 비교")
    top_color_options = list(top_colors)
    selected_color = st.selectbox(
        "컬러 선택 - 위 컬러 Top 목록 기준",
        top_color_options,
        format_func=lambda x: color_label(x, color_rep_products.get(x, "")),
        key=f"selected_color_{campaign_display_name}"
    )

    size_metric = st.selectbox(
        "사이즈 Top 비교 기준",
        ["총 판매수량(1일)", "총 주문수(1일)", "총 전환매출액(1일)", "광고비"],
        index=0,
        key=f"size_metric_{campaign_display_name}"
    )

    color_df = campaign_df[campaign_df["color"] == selected_color].copy()
    size_summary = aggregate(color_df, ["주차", "상품사이즈"])
    top_sizes = (
        size_summary.groupby("상품사이즈")[size_metric]
        .sum()
        .sort_values(ascending=False)
        .head(int(top_n_size))
        .index
    )
    size_top = size_summary[size_summary["상품사이즈"].isin(top_sizes)].copy()

    fig_size = px.bar(
        size_top,
        x="상품사이즈",
        y=size_metric,
        color="주차",
        barmode="group",
        title=f"{clean_tab_name(campaign_display_name)} / {color_label(selected_color, color_rep_products.get(selected_color, ''))} 사이즈 Top {top_n_size} 비교"
    )
    st.plotly_chart(fig_size, use_container_width=True)
    st.dataframe(size_top.sort_values(["주차", size_metric], ascending=[True, False]), use_container_width=True, hide_index=True)

    st.markdown("#### 인기 상품 컬러의 사이즈별 재고현황 요약")
    st.caption("인기 컬러는 전주 판매수량 기준 Top 컬러로 산정하고, 재고는 전주 raw의 옵션ID 기준 최신/최대 잔여수로 요약합니다.")

    popular_color_count = st.slider(
        "재고 요약 대상 인기 컬러 수",
        min_value=3,
        max_value=20,
        value=5,
        key=f"popular_color_count_{campaign_display_name}"
    )

    curr_campaign_df = campaign_curr.copy()
    popular_colors = (
        curr_campaign_df.groupby("color")["총 판매수량(1일)"]
        .sum()
        .sort_values(ascending=False)
        .head(popular_color_count)
        .index
    )

    stock_base = curr_campaign_df[curr_campaign_df["color"].isin(popular_colors)].copy()
    if stock_base.empty:
        st.warning("재고 요약 데이터가 없습니다.")
        return

    stock_option = (
        stock_base.sort_values("날짜")
        .groupby(["color", "상품사이즈", "광고집행 옵션ID"], dropna=False)
        .agg({
            "잔여수": "max",
            "총 판매수량(1일)": "sum",
            "총 전환매출액(1일)": "sum",
        })
        .reset_index()
    )

    stock_summary = (
        stock_option.groupby(["color", "상품사이즈"], dropna=False)
        .agg({
            "잔여수": "sum",
            "총 판매수량(1일)": "sum",
            "총 전환매출액(1일)": "sum",
            "광고집행 옵션ID": "nunique",
        })
        .reset_index()
        .rename(columns={"광고집행 옵션ID": "옵션수"})
    )

    stock_summary["재고상태"] = pd.cut(
        stock_summary["잔여수"],
        bins=[-1, 0, 10, 30, 999999999],
        labels=["품절", "부족", "주의", "여유"]
    )

    # 컬러별 대표 상품명(전체 상품명) 생성
    stock_rep_products = (
        curr_campaign_df[curr_campaign_df["color"].isin(popular_colors)]
        .groupby("color", dropna=False)["광고집행 상품명"]
        .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
        .to_dict()
    )

    # 컬러_상품명에는 전체 상품명만 표시
    stock_summary["컬러_상품명"] = stock_summary["color"].map(stock_rep_products)
    stock_summary = stock_summary.sort_values(["총 판매수량(1일)", "잔여수"], ascending=[False, True])
    stock_summary = stock_summary.drop(columns=["color"])
    st.dataframe(stock_summary, use_container_width=True, hide_index=True)
    make_download(stock_summary, f"{clean_tab_name(campaign_display_name)}_인기컬러_사이즈별_재고현황.xlsx")

    fig_stock = px.bar(
        stock_summary,
        x="상품사이즈",
        y="잔여수",
        color="컬러_상품명",
        barmode="group",
        title=f"{clean_tab_name(campaign_display_name)} 인기 컬러 사이즈별 재고현황"
    )
    st.plotly_chart(fig_stock, use_container_width=True)


# =========================
# 탭 구성
# =========================
st.header("리포트 탭")

metric_cols = ["노출수", "클릭수", "광고비", "총 주문수(1일)", "총 판매수량(1일)", "총 전환매출액(1일)", "CTR", "CPC", "CPA", "CVR", "ROAS"]

total_summary = aggregate(df, ["주차"])
total_compare = compare_weeks(total_summary, [], metric_cols)

# 축약 캠페인명 기준으로 탭 생성
campaign_tab_summary = aggregate(df, ["주차", "캠페인명_축약"])
campaign_tab_compare = compare_weeks(campaign_tab_summary, ["캠페인명_축약"], metric_cols)
campaign_tab_compare = campaign_tab_compare.sort_values("총 전환매출액(1일)_전주", ascending=False).reset_index(drop=True)
campaign_display_order = campaign_tab_compare["캠페인명_축약"].tolist()

used_tab_names = {}
tab_labels = ["전전주vs전주 비교", "신규고객확보캠페인"] + [make_tab_label(name, used_tab_names) for name in campaign_display_order]
tabs = st.tabs(tab_labels)

with tabs[0]:
    st.subheader("전전주 vs 전주 전체값 비교")

    col1, col2, col3 = st.columns(3)
    col1.metric("전전주 기간", f"{prev_df['날짜'].min().date()} ~ {prev_df['날짜'].max().date()}")
    col2.metric("전주 기간", f"{curr_df['날짜'].min().date()} ~ {curr_df['날짜'].max().date()}")
    col3.metric("전체 row 수", f"{len(df):,}")

    total_visual_table = make_total_visual_table(total_summary, prev_df, curr_df)
    st.dataframe(style_total_visual_table(total_visual_table), use_container_width=True, hide_index=True)

    st.markdown("#### 전체 전주 날짜별 성과")
    total_daily_table = make_daily_performance_table(curr_df)
    st.dataframe(style_daily_performance_table(total_daily_table), use_container_width=True, hide_index=True)
    make_download(total_daily_table, "전체_전주_날짜별성과.xlsx")

    kpi_cols = st.columns(5)
    for idx, metric in enumerate(["광고비", "총 주문수(1일)", "총 판매수량(1일)", "총 전환매출액(1일)", "ROAS"]):
        curr_value = float(total_compare[f"{metric}_전주"].iloc[0])
        delta_value = float(total_compare[f"{metric}_증감"].iloc[0])
        delta_pct = float(total_compare[f"{metric}_증감률"].iloc[0])
        kpi_cols[idx].metric(
            METRIC_LABELS.get(metric, metric),
            format_number(curr_value, metric),
            f"{format_number(delta_value, metric)} / {delta_pct:,.1f}%"
        )

    chart_metric = st.selectbox(
        "전체 비교 그래프 지표",
        ["광고비", "총 전환매출액(1일)", "총 판매수량(1일)", "총 주문수(1일)", "ROAS", "CTR", "CPC", "CVR"],
        index=1,
        key="total_chart_metric"
    )

    fig_total = px.bar(
        total_summary,
        x="주차",
        y=chart_metric,
        text=chart_metric,
        title=f"전체 {METRIC_LABELS.get(chart_metric, chart_metric)} 비교"
    )
    fig_total.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    st.plotly_chart(fig_total, use_container_width=True)

    st.markdown("#### 캠페인별 전전주 vs 전주 전체값 비교")
    st.caption("탭과 표는 캠페인명을 두 번째 언더바(_) 전까지만 표시한 기준입니다.")

    all_campaign_tables = []
    for campaign_display in campaign_display_order:
        target_df = df[df["캠페인명_축약"] == campaign_display].copy()
        target_summary = aggregate(target_df, ["주차"])
        target_prev = target_df[target_df["주차"] == "전전주"].copy()
        target_curr = target_df[target_df["주차"] == "전주"].copy()
        table = make_total_visual_table(target_summary, target_prev, target_curr)
        table.insert(0, "캠페인명", clean_tab_name(campaign_display))
        all_campaign_tables.append(table)

        st.markdown(f"##### {clean_tab_name(campaign_display)}")
        display_table = table.drop(columns=[c for c in ["캠페인명", "상품명"] if c in table.columns])
        st.dataframe(style_total_visual_table(display_table), use_container_width=True, hide_index=True)

    if all_campaign_tables:
        campaign_visual_all = pd.concat(all_campaign_tables, ignore_index=True)
        make_download(campaign_visual_all, "캠페인별_전체값비교표.xlsx")

    with st.expander("캠페인별 원본 비교 데이터 보기"):
        display_compare = campaign_tab_compare.copy()
        display_compare["캠페인명"] = display_compare["캠페인명_축약"].apply(clean_tab_name)
        display_compare = display_compare.drop(columns=["캠페인명_축약"])
        cols = ["캠페인명"] + [c for c in display_compare.columns if c != "캠페인명"]
        st.dataframe(style_compare_table(display_compare[cols]), use_container_width=True, hide_index=True)

    campaign_chart_metric = st.selectbox(
        "캠페인 비교 그래프 지표",
        ["총 전환매출액(1일)", "광고비", "총 판매수량(1일)", "총 주문수(1일)", "ROAS"],
        index=0,
        key="campaign_chart_metric"
    )

    campaign_summary_chart = campaign_tab_summary.copy()
    campaign_summary_chart["캠페인명_표시"] = campaign_summary_chart["캠페인명_축약"].apply(clean_tab_name)
    fig_campaign = px.bar(
        campaign_summary_chart,
        x="캠페인명_표시",
        y=campaign_chart_metric,
        color="주차",
        barmode="group",
        title=f"캠페인별 {METRIC_LABELS.get(campaign_chart_metric, campaign_chart_metric)} 비교"
    )
    fig_campaign.update_layout(xaxis_tickangle=-35)
    st.plotly_chart(fig_campaign, use_container_width=True)

with tabs[1]:
    if new_customer_file:
        render_new_customer_dashboard(new_customer_df)
    else:
        st.info("왼쪽 사이드바에서 신규고객확보캠페인 raw 엑셀을 업로드하면 캠페인별 비교표와 날짜별 데이터가 표시됩니다.")

for tab, campaign_display in zip(tabs[2:], campaign_display_order):
    with tab:
        campaign_df_tab = df[df["캠페인명_축약"] == campaign_display].copy()
        render_campaign_dashboard(campaign_display, campaign_df_tab)

with st.expander("원본 데이터 보기"):
    st.dataframe(df, use_container_width=True, hide_index=True)
