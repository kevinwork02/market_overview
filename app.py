"""
Target Market Overview
Household penetration analysis by DMA and Experian audience segment.

Data model:
  experian_location      → luid, dma (DMA code), zipcode, region
  dma_codes_v3           → dma_code, dma_name (friendly name lookup)
  experian_consumerview  → recd_luid, 526 BOOLEAN segment flags
  experian_consumerview2 → recd_luid, demographic/spend cols (no boolean flags currently)
  experian_marketing_attributes (gold) → recd_luid, 2 BOOLEAN flags + demographics

Join spine: experian_location.luid = experian_consumerview.recd_luid (both STRING)
"""

import os
import re
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from databricks import sql

# ── Configuration ─────────────────────────────────────────────────────────────
def _cfg(env_key: str, secret_key: str | None = None) -> str:
    # Try lowercase, then exact case, then env var — handles any Streamlit secrets casing
    for key in [secret_key or env_key.lower(), env_key]:
        try:
            val = st.secrets.get(key, "")
            if val:
                return val
        except Exception:
            pass
    return os.environ.get(env_key, "")

SERVER_HOSTNAME = _cfg("DATABRICKS_SERVER_HOSTNAME")
HTTP_PATH       = _cfg("DATABRICKS_HTTP_PATH")
TOKEN           = _cfg("DATABRICKS_TOKEN")

# ── Palette (matches iSpot app) ───────────────────────────────────────────────
NAVY       = "#1B2A4A"
CYAN       = "#00BCD4"
LIGHT_CYAN = "#80DEEA"
LIME       = "#C5E063"
DARK_BG    = "#0d1f3a"
BORDER     = "#2a3d5e"
AMBER      = "#FFB74D"

# ── DB helpers ─────────────────────────────────────────────────────────────────
def _conn():
    if not SERVER_HOSTNAME:
        raise ValueError("DATABRICKS_SERVER_HOSTNAME secret is missing or empty.")
    if not HTTP_PATH:
        raise ValueError("DATABRICKS_HTTP_PATH secret is missing or empty.")
    if not TOKEN:
        raise ValueError("DATABRICKS_TOKEN secret is missing or empty.")
    return sql.connect(
        server_hostname=SERVER_HOSTNAME.strip(),
        http_path=HTTP_PATH.strip(),
        access_token=TOKEN.strip(),
    )


def run_query(q: str) -> pd.DataFrame:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute(q)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


# ── Data loaders (all cached) ──────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_dma_list() -> pd.DataFrame:
    """All DMAs with HH counts and national rank derived from experian_location."""
    df = run_query("""
        SELECT
            d.dma_code,
            d.dma_name,
            COUNT(DISTINCT l.luid) AS hh_count
        FROM locality_dev.silver.experian_location  l
        JOIN locality_dev.default.dma_codes_v3       d
          ON CAST(d.dma_code AS STRING) = l.dma
        WHERE l.dma IS NOT NULL
        GROUP BY d.dma_code, d.dma_name
        ORDER BY hh_count DESC
    """)
    df["hh_count"]  = df["hh_count"].astype(int)
    df["us_hh_rank"] = (
        df["hh_count"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_audience_columns() -> pd.DataFrame:
    """
    Deduplicated BOOLEAN audience columns from all three Experian tables.
    Priority for dedup: experian_consumerview > experian_marketing_attributes > experian_consumerview2.
    Note: experian_consumerview2 has 0 boolean cols currently (all numeric/string).
    """
    q = """
    SELECT column_name,
           'locality_dev.silver.experian_consumerview' AS source_table,
           1 AS priority
    FROM locality_dev.information_schema.columns
    WHERE table_schema = 'silver'
      AND table_name   = 'experian_consumerview'
      AND data_type    = 'BOOLEAN'

    UNION ALL
    SELECT column_name,
           'locality_dev.gold.experian_marketing_attributes' AS source_table,
           2 AS priority
    FROM locality_dev.information_schema.columns
    WHERE table_schema = 'gold'
      AND table_name   = 'experian_marketing_attributes'
      AND data_type    = 'BOOLEAN'

    UNION ALL
    SELECT column_name,
           'locality_dev.silver.experian_consumerview2' AS source_table,
           3 AS priority
    FROM locality_dev.information_schema.columns
    WHERE table_schema = 'silver'
      AND table_name   = 'experian_consumerview2'
      AND data_type    = 'BOOLEAN'
    """
    df = run_query(q)
    # Keep only highest-priority row per column name (dedup)
    df = (
        df.sort_values("priority")
        .drop_duplicates(subset="column_name", keep="first")
        .reset_index(drop=True)
    )
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data(
    dma_codes: tuple,
    audience_col: str,
    source_table: str,
) -> pd.DataFrame:
    """HH counts and audience penetration per DMA for the selected markets."""
    # Sanitize column name (prevent injection)
    if not re.match(r"^[a-zA-Z0-9_]+$", audience_col):
        raise ValueError(f"Invalid column name: {audience_col}")

    dma_filter = ", ".join(f"'{c}'" for c in dma_codes)

    # experian_consumerview / experian_marketing_attributes: recd_luid is STRING
    # experian_consumerview2 (if ever has boolean cols): recd_luid is BIGINT → needs cast
    if "experian_consumerview2" in source_table:
        join_clause = f"LEFT JOIN {source_table} cv ON CAST(l.luid AS BIGINT) = cv.recd_luid"
    else:
        join_clause = f"LEFT JOIN {source_table} cv ON l.luid = cv.recd_luid"

    # For marketing_attributes, restrict to live records only
    extra_where = ""
    if "experian_marketing_attributes" in source_table:
        extra_where = "AND (cv.reliability_code BETWEEN 1 AND 4 OR cv.reliability_code IS NULL)"

    q = f"""
    SELECT
        d.dma_name,
        l.dma                                                                       AS dma_code,
        COUNT(DISTINCT l.luid)                                                      AS tv_households,
        COUNT(DISTINCT CASE WHEN cv.`{audience_col}` = TRUE THEN l.luid END)        AS audience_hhs
    FROM locality_dev.silver.experian_location  l
    JOIN locality_dev.default.dma_codes_v3       d  ON CAST(d.dma_code AS STRING) = l.dma
    {join_clause}
    WHERE l.dma IN ({dma_filter})
    {extra_where}
    GROUP BY d.dma_name, l.dma
    ORDER BY tv_households DESC
    """
    df = run_query(q)
    df["tv_households"] = df["tv_households"].astype(int)
    df["audience_hhs"]  = df["audience_hhs"].astype(int)
    return df


# ── Boolean audience query ───────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_market_data_bool(
    dma_codes: tuple,
    include_all: tuple,
    include_any: tuple,
    exclude: tuple,
    aud_table_map: tuple,   # ((col, table), ...)
) -> pd.DataFrame:
    """
    HH penetration with boolean AND / OR / NOT audience logic.
    include_all  → HH must match EVERY segment in this group (AND)
    include_any  → HH must match AT LEAST ONE segment in this group (OR)
    exclude      → HH must NOT match any segment in this group (AND NOT)
    """
    all_segs = list(include_all) + list(include_any) + list(exclude)
    for col in all_segs:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    table_of = dict(aud_table_map)

    def _alias(col: str) -> str:
        return "ma" if "experian_marketing_attributes" in table_of.get(col, "") else "cv"

    # Determine which tables are actually needed
    used_tables = {table_of.get(s, "") for s in all_segs}
    needs_cv = any("experian_consumerview" in t and "marketing" not in t for t in used_tables)
    needs_ma = any("experian_marketing_attributes" in t for t in used_tables)

    joins = []
    if needs_cv:
        joins.append("LEFT JOIN locality_dev.silver.experian_consumerview cv ON l.luid = cv.recd_luid")
    if needs_ma:
        joins.append(
            "LEFT JOIN locality_dev.gold.experian_marketing_attributes ma "
            "ON l.luid = ma.recd_luid "
            "AND (ma.reliability_code BETWEEN 1 AND 4 OR ma.reliability_code IS NULL)"
        )

    # Build the boolean CASE WHEN filter
    filter_parts = []
    for seg in include_all:
        filter_parts.append(f"{_alias(seg)}.`{seg}` = TRUE")
    if include_any:
        or_clauses = [f"{_alias(seg)}.`{seg}` = TRUE" for seg in include_any]
        filter_parts.append(f"({' OR '.join(or_clauses)})")
    for seg in exclude:
        filter_parts.append(f"({_alias(seg)}.`{seg}` IS NULL OR {_alias(seg)}.`{seg}` != TRUE)")

    filter_expr = " AND ".join(filter_parts) if filter_parts else "TRUE"
    dma_filter  = ", ".join(f"'{c}'" for c in dma_codes)
    join_block  = "\n    ".join(joins)

    q = f"""
    SELECT
        d.dma_name,
        l.dma                                                                     AS dma_code,
        COUNT(DISTINCT l.luid)                                                    AS tv_households,
        COUNT(DISTINCT CASE WHEN {filter_expr} THEN l.luid END)                   AS audience_hhs
    FROM locality_dev.silver.experian_location  l
    JOIN locality_dev.default.dma_codes_v3       d  ON CAST(d.dma_code AS STRING) = l.dma
    {join_block}
    WHERE l.dma IN ({dma_filter})
    GROUP BY d.dma_name, l.dma
    ORDER BY tv_households DESC
    """
    df = run_query(q)
    df["tv_households"] = df["tv_households"].astype(int)
    df["audience_hhs"]  = df["audience_hhs"].astype(int)
    return df


# ── Inscape political loaders ───────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def load_inscape_segments() -> pd.DataFrame:
    """Inscape political taxonomy segments with short labels."""
    return run_query("""
        SELECT
            segment_id,
            regexp_replace(name, '^.*> Political > ', '') AS short_name
        FROM inscape.inscape_locality_prod.inscape_standardpoliticaltaxonomy_metadata
        ORDER BY segment_id
    """)


@st.cache_data(ttl=1800, show_spinner=False)
def load_inscape_market_data(dma_codes: tuple, seg_id: str) -> pd.DataFrame:
    """Single Inscape political segment — panel HH counts per DMA."""
    if not re.match(r"^[A-Z0-9_]+$", seg_id):
        raise ValueError(f"Invalid segment ID: {seg_id}")
    dma_filter = ", ".join(f"'{c}'" for c in dma_codes)
    q = f"""
    SELECT
        d.dma_name,
        l.dma                                                            AS dma_code,
        COUNT(DISTINCT m.luid)                                           AS panel_hhs,
        COUNT(DISTINCT CASE WHEN p.`{seg_id}` = 1 THEN m.luid END)      AS audience_hhs
    FROM inscape.inscape_locality_prod.inscape_standardpoliticaltaxonomy_identifiers p
    JOIN locality_dev.silver.experian_consolidated_id_map m
      ON m.identity = p.tv_ad_id AND m.id_type = 'ctv'
    JOIN locality_dev.silver.experian_location l ON l.luid = m.luid
    JOIN locality_dev.default.dma_codes_v3     d ON CAST(d.dma_code AS STRING) = l.dma
    WHERE l.dma IN ({dma_filter})
    GROUP BY d.dma_name, l.dma
    ORDER BY panel_hhs DESC
    """
    df = run_query(q)
    df["panel_hhs"]    = df["panel_hhs"].astype(int)
    df["audience_hhs"] = df["audience_hhs"].astype(int)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def load_inscape_market_data_bool(
    dma_codes: tuple,
    include_all: tuple,
    include_any: tuple,
    exclude: tuple,
) -> pd.DataFrame:
    """Boolean AND/OR/NOT Inscape political segments — panel HH counts per DMA."""
    for seg in list(include_all) + list(include_any) + list(exclude):
        if not re.match(r"^[A-Z0-9_]+$", seg):
            raise ValueError(f"Invalid segment ID: {seg}")
    filter_parts: list[str] = []
    for seg in include_all:
        filter_parts.append(f"p.`{seg}` = 1")
    if include_any:
        or_clauses = [f"p.`{s}` = 1" for s in include_any]
        filter_parts.append("(" + " OR ".join(or_clauses) + ")")
    for seg in exclude:
        filter_parts.append(f"(p.`{seg}` = 0 OR p.`{seg}` IS NULL)")
    filter_expr = " AND ".join(filter_parts) if filter_parts else "TRUE"
    dma_filter  = ", ".join(f"'{c}'" for c in dma_codes)
    q = f"""
    SELECT
        d.dma_name,
        l.dma                                                             AS dma_code,
        COUNT(DISTINCT m.luid)                                            AS panel_hhs,
        COUNT(DISTINCT CASE WHEN {filter_expr} THEN m.luid END)           AS audience_hhs
    FROM inscape.inscape_locality_prod.inscape_standardpoliticaltaxonomy_identifiers p
    JOIN locality_dev.silver.experian_consolidated_id_map m
      ON m.identity = p.tv_ad_id AND m.id_type = 'ctv'
    JOIN locality_dev.silver.experian_location l ON l.luid = m.luid
    JOIN locality_dev.default.dma_codes_v3     d ON CAST(d.dma_code AS STRING) = l.dma
    WHERE l.dma IN ({dma_filter})
    GROUP BY d.dma_name, l.dma
    ORDER BY panel_hhs DESC
    """
    df = run_query(q)
    df["panel_hhs"]    = df["panel_hhs"].astype(int)
    df["audience_hhs"] = df["audience_hhs"].astype(int)
    return df


# ── Helpers ────────────────────────────────────────────────────────────────────
def fmt_label(col: str) -> str:
    """snake_case column name → readable Title Case label."""
    label = col.replace("_", " ").title()
    # Strip leading 'Rc ' prefix from Experian raw column names
    if label.lower().startswith("rc "):
        label = label[3:]
    return label


def fmt_int(v) -> str:
    return f"{int(v):,}"


def fmt_pct(v) -> str:
    return f"{float(v):.1%}"


# ── Styles ─────────────────────────────────────────────────────────────────────
CSS = f"""
<style>
.block-container {{ padding-top: 1.25rem; }}
.header-bar {{
    background: linear-gradient(90deg, {NAVY} 0%, {DARK_BG} 100%);
    padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 1.25rem;
    border-left: 4px solid {CYAN};
}}
.header-bar h1 {{ color: {CYAN}; margin: 0; font-size: 1.7rem; }}
.header-bar p  {{ color: {LIGHT_CYAN}; margin: 0.2rem 0 0 0; font-size: 0.85rem; }}
.step-pill {{
    display: inline-block; background: {NAVY}; color: {CYAN};
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; padding: 0.2rem 0.65rem;
    border-radius: 999px; border: 1px solid {BORDER};
    margin-bottom: 0.5rem;
}}
.kpi-card {{
    background: {NAVY}; border-radius: 8px; padding: 0.85rem 1.1rem;
    border: 1px solid {BORDER};
}}
.kpi-card .val {{ font-size: 1.45rem; font-weight: 700; color: {LIME}; }}
.kpi-card .lbl {{ font-size: 0.72rem; color: {LIGHT_CYAN}; margin-top: 0.12rem; }}
</style>
"""


# ── Tab functions ──────────────────────────────────────────────────────────────────────
def _experian_tab(
    selected_names: list, selected_codes: list, n_markets: int, dma_df: pd.DataFrame
) -> None:
    """Experian ConsumerView + marketing_attributes audience workflow."""
    st.markdown('<div class="step-pill">Step 2 — Select Audience Segment(s)</div>', unsafe_allow_html=True)

    with st.spinner("Loading audience segments…"):
        aud_df = load_audience_columns()

    aud_meta: dict[str, dict] = {}
    col_to_table: dict[str, str] = {}
    for _, row in aud_df.iterrows():
        label = fmt_label(row["column_name"])
        aud_meta[label] = {"col": row["column_name"], "table": row["source_table"]}
        col_to_table[row["column_name"]] = row["source_table"]

    sorted_labels  = sorted(aud_meta.keys())
    total_segments = len(sorted_labels)

    aud_mode = st.radio(
        "Selection mode:",
        ["Single segment", "Boolean logic  (AND / OR / NOT)"],
        horizontal=True, label_visibility="collapsed", key="exp_mode",
    )

    inc_all: list[str] = []
    inc_any: list[str] = []
    exc:     list[str] = []

    if aud_mode == "Single segment":
        selected_aud = st.selectbox(
            f"Choose an audience  ({total_segments:,} segments available):",
            options=[""] + sorted_labels,
            format_func=lambda x: "— Select an audience segment —" if x == "" else x,
            key="exp_single",
        )
        aud_ready = bool(selected_aud)
    else:
        selected_aud = ""
        hdr = f'<div style="color:{CYAN};font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">'
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(hdr + "Must match ALL (AND)</div>", unsafe_allow_html=True)
            inc_all_labels = st.multiselect("", sorted_labels, key="exp_inc_all",
                                             placeholder="Add segment…", label_visibility="collapsed")
        with c2:
            st.markdown(hdr + "Must match ANY (OR)</div>", unsafe_allow_html=True)
            inc_any_labels = st.multiselect("", sorted_labels, key="exp_inc_any",
                                             placeholder="Add segment…", label_visibility="collapsed")
        with c3:
            st.markdown(hdr + "Exclude (AND NOT)</div>", unsafe_allow_html=True)
            exc_labels = st.multiselect("", sorted_labels, key="exp_exc",
                                         placeholder="Add segment…", label_visibility="collapsed")
        inc_all = [aud_meta[l]["col"] for l in inc_all_labels]
        inc_any = [aud_meta[l]["col"] for l in inc_any_labels]
        exc     = [aud_meta[l]["col"] for l in exc_labels]
        aud_ready = bool(inc_all or inc_any or exc)
        if aud_ready:
            parts = []
            if inc_all_labels:
                parts.append(" AND ".join(inc_all_labels))
            if inc_any_labels:
                inner = " OR ".join(inc_any_labels)
                parts.append(f"({inner})" if len(inc_any_labels) > 1 else inner)
            if exc_labels:
                inner = " OR ".join(exc_labels)
                parts.append(f"NOT ({inner})" if len(exc_labels) > 1 else f"NOT {exc_labels[0]}")
            selected_aud = " AND ".join(parts)

    st.markdown("---")

    if not aud_ready:
        sub = dma_df[dma_df["dma_name"].isin(selected_names)].copy().sort_values("hh_count", ascending=False)
        total_hhs = sub["hh_count"].sum()
        sub["footprint_pct"] = sub["hh_count"] / total_hhs
        st.markdown(f'<div class="step-pill">{n_markets}-Market Footprint · {total_hhs:,.0f} Total HHs</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "DMA": sub["dma_name"].values, "US HH Rank": sub["us_hh_rank"].values,
            "TV Households": [fmt_int(v) for v in sub["hh_count"]],
            f"% of {n_markets}-Mkt Footprint": [fmt_pct(v) for v in sub["footprint_pct"]],
        }), use_container_width=True, hide_index=True)
        st.info("ℹ️  Select an audience segment above to add penetration columns.")
        return

    with st.spinner(f"Calculating '{selected_aud}' penetration across {n_markets} market{'s' if n_markets != 1 else ''}…"):
        if aud_mode == "Single segment":
            aud_info = aud_meta[selected_aud]
            result = load_market_data(tuple(sorted(selected_codes)), aud_info["col"], aud_info["table"])
        else:
            aud_table_map = tuple((col, col_to_table[col]) for col in (inc_all + inc_any + exc) if col in col_to_table)
            result = load_market_data_bool(tuple(sorted(selected_codes)), tuple(inc_all), tuple(inc_any), tuple(exc), aud_table_map)

    result = result.merge(dma_df[["dma_code", "us_hh_rank"]], on="dma_code", how="left")
    result["dma_code"] = result["dma_code"].astype(str)
    total_hhs  = int(result["tv_households"].sum())
    total_aud  = int(result["audience_hhs"].sum())
    result["footprint_pct"] = result["tv_households"] / total_hhs
    result["audience_pct"]  = result["audience_hhs"] / result["tv_households"].replace(0, pd.NA)
    overall_pct = total_aud / total_hhs if total_hhs > 0 else 0.0

    _render_results(result, selected_aud, n_markets, total_hhs, total_aud, overall_pct,
                    hh_col="tv_households", hh_label="TV Households", pct_label="% of HHs",
                    bar_color=CYAN, csv_key="exp_csv", csv_file="target_market_overview.csv")


def _inscape_tab(
    selected_names: list, selected_codes: list, n_markets: int, dma_df: pd.DataFrame
) -> None:
    """Inscape political taxonomy audience workflow (LUID-linked via CTV crosswalk)."""
    st.markdown('<div class="step-pill">Step 2 — Select Political Audience(s)</div>', unsafe_allow_html=True)
    st.caption(
        "📺 Denominator = Inscape panel HHs (VIZIO CTV devices matched via LUID crosswalk, ∼7.7M nationally). "
        "Penetration shown as % of matched panel, not full TV HH universe."
    )

    with st.spinner("Loading Inscape political segments…"):
        segs_df = load_inscape_segments()

    seg_meta: dict[str, str] = dict(zip(segs_df["short_name"], segs_df["segment_id"]))
    sorted_seg_labels = sorted(seg_meta.keys())
    total_segs = len(sorted_seg_labels)

    pol_mode = st.radio(
        "Selection mode:",
        ["Single segment", "Boolean logic  (AND / OR / NOT)"],
        horizontal=True, label_visibility="collapsed", key="pol_mode",
    )

    selected_seg_id:   str       = ""
    audience_label:    str       = ""
    inc_all_ids:       list[str] = []
    inc_any_ids:       list[str] = []
    exc_ids:           list[str] = []

    if pol_mode == "Single segment":
        selected_seg_label = st.selectbox(
            f"Choose a political segment  ({total_segs:,} available):",
            options=[""] + sorted_seg_labels,
            format_func=lambda x: "— Select a segment —" if x == "" else x,
            key="pol_single",
        )
        pol_ready = bool(selected_seg_label)
        if pol_ready:
            selected_seg_id = seg_meta[selected_seg_label]
            audience_label  = selected_seg_label
    else:
        hdr = f'<div style="color:{AMBER};font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.07em;">'
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(hdr + "Must match ALL (AND)</div>", unsafe_allow_html=True)
            inc_all_labels = st.multiselect("", sorted_seg_labels, key="pol_inc_all",
                                             placeholder="Add segment…", label_visibility="collapsed")
        with c2:
            st.markdown(hdr + "Must match ANY (OR)</div>", unsafe_allow_html=True)
            inc_any_labels = st.multiselect("", sorted_seg_labels, key="pol_inc_any",
                                             placeholder="Add segment…", label_visibility="collapsed")
        with c3:
            st.markdown(hdr + "Exclude (AND NOT)</div>", unsafe_allow_html=True)
            exc_labels = st.multiselect("", sorted_seg_labels, key="pol_exc",
                                         placeholder="Add segment…", label_visibility="collapsed")
        inc_all_ids = [seg_meta[l] for l in inc_all_labels]
        inc_any_ids = [seg_meta[l] for l in inc_any_labels]
        exc_ids     = [seg_meta[l] for l in exc_labels]
        pol_ready   = bool(inc_all_ids or inc_any_ids or exc_ids)
        if pol_ready:
            parts = []
            if inc_all_labels:
                parts.append(" AND ".join(inc_all_labels))
            if inc_any_labels:
                inner = " OR ".join(inc_any_labels)
                parts.append(f"({inner})" if len(inc_any_labels) > 1 else inner)
            if exc_labels:
                inner = " OR ".join(exc_labels)
                parts.append(f"NOT ({inner})" if len(exc_labels) > 1 else f"NOT {exc_labels[0]}")
            audience_label = " AND ".join(parts)

    st.markdown("---")

    if not pol_ready:
        sub = dma_df[dma_df["dma_name"].isin(selected_names)].copy().sort_values("hh_count", ascending=False)
        total_hhs = sub["hh_count"].sum()
        sub["footprint_pct"] = sub["hh_count"] / total_hhs
        st.markdown(f'<div class="step-pill">{n_markets}-Market Footprint · {total_hhs:,.0f} Total HHs</div>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "DMA": sub["dma_name"].values, "US HH Rank": sub["us_hh_rank"].values,
            "TV Households": [fmt_int(v) for v in sub["hh_count"]],
            f"% of {n_markets}-Mkt Footprint": [fmt_pct(v) for v in sub["footprint_pct"]],
        }), use_container_width=True, hide_index=True)
        st.info("ℹ️  Select a political segment above to add penetration columns.")
        return

    with st.spinner(f"Calculating '{audience_label}' penetration across {n_markets} market{'s' if n_markets != 1 else ''}…"):
        if pol_mode == "Single segment":
            result = load_inscape_market_data(tuple(sorted(selected_codes)), selected_seg_id)
        else:
            result = load_inscape_market_data_bool(
                tuple(sorted(selected_codes)), tuple(inc_all_ids), tuple(inc_any_ids), tuple(exc_ids)
            )

    result = result.merge(dma_df[["dma_code", "us_hh_rank"]], on="dma_code", how="left")
    result["dma_code"] = result["dma_code"].astype(str)
    total_panel = int(result["panel_hhs"].sum())
    total_aud   = int(result["audience_hhs"].sum())
    result["footprint_pct"] = result["panel_hhs"] / total_panel
    result["audience_pct"]  = result["audience_hhs"] / result["panel_hhs"].replace(0, pd.NA)
    overall_pct = total_aud / total_panel if total_panel > 0 else 0.0

    _render_results(result, audience_label, n_markets, total_panel, total_aud, overall_pct,
                    hh_col="panel_hhs", hh_label="Inscape Panel HHs", pct_label="% of Panel",
                    bar_color=AMBER, csv_key="pol_csv", csv_file="target_market_overview_political.csv")


def _render_results(
    result: pd.DataFrame,
    audience_label: str,
    n_markets: int,
    total_hhs: int,
    total_aud: int,
    overall_pct: float,
    *,
    hh_col: str,
    hh_label: str,
    pct_label: str,
    bar_color: str,
    csv_key: str,
    csv_file: str,
) -> None:
    """Shared KPI cards, results table, bar chart, and CSV export."""
    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    for col_obj, val, lbl in [
        (c1, str(n_markets),       "Markets Selected"),
        (c2, fmt_int(total_hhs),   hh_label),
        (c3, fmt_int(total_aud),   f"{audience_label} HHs"),
        (c4, fmt_pct(overall_pct), f"Avg {pct_label}"),
    ]:
        col_obj.markdown(
            f'<div class="kpi-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    st.markdown(
        f'<div class="step-pill">Results — {n_markets}-Market Footprint · {audience_label}</div>',
        unsafe_allow_html=True,
    )

    # Results table
    rows = []
    for _, r in result.sort_values(hh_col, ascending=False).iterrows():
        aud_pct_val = fmt_pct(float(r["audience_pct"])) if pd.notna(r["audience_pct"]) else "—"
        rows.append({
            "DMA":                               r["dma_name"],
            "US HH Rank":                        int(r["us_hh_rank"]),
            hh_label:                            fmt_int(int(r[hh_col])),
            f"% of {n_markets}-Mkt Footprint":   fmt_pct(float(r["footprint_pct"])),
            f"{audience_label} (HHs)":           fmt_int(int(r["audience_hhs"])),
            f"{audience_label} {pct_label}":     aud_pct_val,
        })
    rows.append({
        "DMA":                             "TOTAL",
        "US HH Rank":                      "",
        hh_label:                          fmt_int(total_hhs),
        f"% of {n_markets}-Mkt Footprint": "100.0%",
        f"{audience_label} (HHs)":         fmt_int(total_aud),
        f"{audience_label} {pct_label}":   fmt_pct(overall_pct),
    })

    disp_df = pd.DataFrame(rows)
    last_idx = len(disp_df) - 1

    def _row_style(row):
        if row.name == last_idx:
            return [f"background-color: {NAVY}; font-weight: bold; color: {LIME}"] * len(row)
        return [""] * len(row)

    st.dataframe(disp_df.style.apply(_row_style, axis=1), use_container_width=True, hide_index=True)

    # Bar chart
    chart = result.sort_values("audience_pct", ascending=True).copy()
    chart["pct_f"] = chart["audience_pct"].apply(lambda v: float(v) * 100 if pd.notna(v) else 0.0)
    fig = go.Figure(go.Bar(
        x=chart["pct_f"].tolist(),
        y=chart["dma_name"].tolist(),
        orientation="h",
        marker_color=bar_color,
        text=chart["audience_pct"].apply(lambda v: fmt_pct(float(v)) if pd.notna(v) else "—").tolist(),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text=f"{audience_label} — {pct_label} by DMA", font_color=LIGHT_CYAN),
        xaxis=dict(title=pct_label, tickformat=".1f", ticksuffix="%",
                   gridcolor=BORDER, title_font_color=LIGHT_CYAN),
        yaxis=dict(gridcolor=BORDER),
        plot_bgcolor=DARK_BG, paper_bgcolor=DARK_BG, font_color=LIGHT_CYAN,
        height=max(320, 42 * len(chart)), margin=dict(l=210, r=90, t=55, b=40),
    )
    fig.add_vline(x=overall_pct * 100, line_dash="dash", line_color=LIME,
                  annotation_text=f"Avg {fmt_pct(overall_pct)}", annotation_font_color=LIME)
    st.plotly_chart(fig, use_container_width=True)

    csv_bytes = disp_df.to_csv(index=False).encode()
    st.download_button("📥 Export CSV", data=csv_bytes, file_name=csv_file,
                       mime="text/csv", key=csv_key)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(
        page_title="Target Market Overview",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="header-bar">
        <h1>📊 Target Market Overview</h1>
        <p>Select markets and an audience segment to analyze household penetration across DMAs</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Step 1: DMA multiselect ────────────────────────────────────────────
    st.markdown('<div class="step-pill">Step 1 — Select Markets</div>', unsafe_allow_html=True)

    with st.spinner("Loading DMA universe…"):
        dma_df = load_dma_list()

    dma_code_of = dict(zip(dma_df["dma_name"], dma_df["dma_code"]))
    hh_of       = dict(zip(dma_df["dma_name"], dma_df["hh_count"]))

    selected_names = st.multiselect(
        "Choose one or more DMAs:",
        options=dma_df["dma_name"].tolist(),
        default=None,
        format_func=lambda n: f"{n}  —  {hh_of[n]:,.0f} HHs",
        placeholder="Search or select markets…",
    )

    if not selected_names:
        st.info("ℹ️  Select at least one DMA to continue.")
        return

    selected_codes = [dma_code_of[n] for n in selected_names]
    n_markets      = len(selected_codes)

    # ── Tabs: Experian Audiences | Inscape Political ───────────────────────────
    st.markdown("---")
    tab1, tab2 = st.tabs(["  📊  Experian Audiences  ", "  📺  Inscape Political  "])
    with tab1:
        _experian_tab(selected_names, selected_codes, n_markets, dma_df)
    with tab2:
        _inscape_tab(selected_names, selected_codes, n_markets, dma_df)


if __name__ == "__main__":
    main()
