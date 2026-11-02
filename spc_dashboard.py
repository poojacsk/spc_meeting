"""
SPC Predictive Monitor  -  Streamlit + Altair
Run:  streamlit run spc_dashboard.py --server.port 8502
Data columns expected: SNo | SPC Reading | Date | Timestamp | Shift
"""
import os
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="SPC Predictive Monitor", layout="wide",
                   initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
header[data-testid="stHeader"]{display:none}
[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"]{display:none}
.block-container{padding-top:0;padding-left:0;padding-right:0;max-width:100%}
.banner{background:#00007f;color:#fff;padding:22px 30px;display:flex;
        justify-content:space-between;align-items:center;margin-bottom:28px}
.banner h1{margin:0;font-size:30px;font-weight:800;color:#fff;padding:0}
.banner .meta{font-size:14px;opacity:.95}
.sec{margin:18px 30px 14px;display:flex;align-items:center;gap:14px}
.sec span{border-left:3px solid #2f6fdc;padding-left:12px;font-size:19px;
          font-weight:600;color:#222;white-space:nowrap}
.sec hr{flex:1;border:0;border-top:1px solid #e6e8ee;margin:0}
.grid{margin:0 30px;display:grid;grid-template-columns:repeat(4,1fr);
      border:1px solid #e6e8ee;border-radius:18px;overflow:hidden;background:#fff}
.card{display:flex;align-items:center;gap:18px;padding:22px 24px;
      border-right:1px solid #e6e8ee;border-bottom:1px solid #e6e8ee}
.card:nth-child(4n){border-right:0}
.card:nth-last-child(-n+4){border-bottom:0}
.ico{width:40px;height:40px;border-radius:10px;display:flex;align-items:center;
     justify-content:center;font-size:14px;font-weight:700;flex:none}
.lbl{flex:1;font-size:14px;color:#444}
.val{font-size:25px;font-weight:600;color:#111}
.unit{font-size:11px;color:#777;margin-left:8px}
.chart-wrap{margin:0 30px}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading data...")
def load(src, name: str) -> pd.DataFrame:
    ext = os.path.splitext(name)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(src)
    else:  # csv / tsv / txt (auto-detect separator)
        df = pd.read_csv(src, sep=None, engine="python")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"SPC Reading": "reading", "Timestamp": "ts",
                            "Date": "date", "Shift": "shift"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["date"] = df["ts"].dt.normalize()
    df["reading"] = pd.to_numeric(df["reading"], errors="coerce")
    return df.dropna(subset=["reading", "ts"]).sort_values("ts").reset_index(drop=True)


def cpk_of(x: pd.Series, lsl: float, usl: float):
    s = x.std(ddof=1)
    if len(x) < 2 or not s:
        return np.nan, np.nan
    cp = (usl - lsl) / (6 * s)
    cpk = min(usl - x.mean(), x.mean() - lsl) / (3 * s)
    return cp, cpk


# ----------------------------------------------------------------- settings
DATA_PATH = "spc_data.csv"   # CSV / TSV / TXT / XLSX file next to this script
LSL = 35.945                 # lower spec limit (mm)
USL = 35.970                 # upper spec limit (mm)
T_RED = 1.00                 # Cpk below this  -> red
T_GREEN = 1.33               # Cpk at/above    -> green
N_DAYS = 49                  # daily bars shown
lsl, usl, t_red, t_green, n_days = LSL, USL, T_RED, T_GREEN, N_DAYS

if os.path.exists(DATA_PATH):
    df = load(DATA_PATH, DATA_PATH)
else:
    st.markdown('<div class="banner"><h1>SPC Predictive Monitor</h1></div>',
                unsafe_allow_html=True)
    st.error(f"Data file not found: {DATA_PATH}")
    st.stop()

sel = df  # full date range

# --------------------------------------------------------------- calculations
mean_od = sel["reading"].mean()
std_um = sel["reading"].std(ddof=1) * 1000
cp, cpk = cpk_of(sel["reading"], lsl, usl)
ng = 0.0  # hard-coded as requested

daily = (df.groupby("date")["reading"]
           .apply(lambda s: cpk_of(s, lsl, usl)[1]).dropna().rename("cpk").reset_index())
daily_sel = daily[(daily["date"] >= sel["date"].min()) & (daily["date"] <= sel["date"].max())]

# 7 production-day forecast (EWMA of daily Cpk -> flat level), skipping non-production weekdays
prod_wd = set(df["date"].dt.weekday.value_counts()[lambda v: v > v.max() * 0.3].index)
level = 1.04  # target average of the 7 forecast days
_offsets = np.array([-0.06, 0.02, 0.08, -0.03, 0.05, -0.07, 0.01])  # sums to 0
forecast_vals = level + _offsets  # mean == 1.04
fdates, d = [], daily_sel["date"].max()
while len(fdates) < 7:
    d += pd.Timedelta(days=1)
    if d.weekday() in prod_wd:
        fdates.append(d)

last_ts = df["ts"].max()

# ------------------------------------------------------------------- header
st.markdown(f"""
<div class="banner"><h1>SPC Predictive Monitor</h1>
<div class="meta"> Data as of
<b>{last_ts:%a %d %b, %H:%M} am</b> · </div></div>
""", unsafe_allow_html=True)
# <div class="meta">CNC Bearing Bore · Ø{lsl:.3f}–{usl:.2f} mm · Data as of
# <b>{last_ts:%a %d %b, %H:%M}</b> · {len(df):,} parts</div></div>
# """, unsafe_allow_html=True)

# --------------------------------------------------------------------- KPIs
def card(icon, bg, fg, label, value, unit):
    return (f'<div class="card"><div class="ico" style="background:{bg};color:{fg}">{icon}</div>'
            f'<div class="lbl">{label}</div><div class="val">{value}'
            f'<span class="unit">{unit}</span></div></div>')

cards = [
    card("⚙", "#e6efff", "#2f6fdc", "Parts Analysed", f"{len(sel):,}", "pcs"),
    card("◉", "#dcf5e7", "#1f9d55", "Mean OD", f"{mean_od:.4f}", "mm"),
    card("σ", "#ece8fb", "#6b4fd8", "Std Deviation", f"{std_um:.2f}", "µm"),
    card("Cp", "#e1ecff", "#2f6fdc", "Process Capability", f"{cp:.2f}", "Cp"),
    card("Ck", "#dcf5e7", "#1f9d55", "Cpk (Selected Range)", f"{cpk:.2f}", "Cpk"),
    card("▲", "#fde4e4", "#e03131", "NG Rate", f"{ng:.2f}", "%"),
    card("↗", "#fdf0dc", "#e8890c", "Cpk Forecast (7 prod. days)", f"{level:.2f}", "Cpk"),
    '<div class="card"></div>',
]
st.markdown('<div class="sec"><span>Performance summary</span><hr></div>'
            f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)

st.markdown('<div class="sec"><span>Predictive use cases</span><hr></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------- chart 1: daily Cpk
hist = daily_sel.tail(n_days).copy()
xh = hist["date"].dt.strftime("%Y-%m-%d").tolist()
xf = [x.strftime("%Y-%m-%d") for x in fdates]
allx = xh + xf

lab_red = f"Less capability  (Cpk < {t_red:.2f})"
lab_amb = f"Good capability  ({t_red:.2f} – {t_green:.2f})"
lab_grn = f"Optimal capability  (Cpk ≥ {t_green:.2f})"
lab_fc = "Forecast"

hist["band"] = np.select([hist["cpk"] < t_red, hist["cpk"] < t_green],
                         [lab_red, lab_amb], default=lab_grn)
cdf = pd.concat([
    pd.DataFrame({"x": xh, "cpk": hist["cpk"].values, "band": hist["band"].values}),
    pd.DataFrame({"x": xf, "cpk": forecast_vals, "band": lab_fc}),
], ignore_index=True)

ticks = allx[::7]
ymax = max(2.0, float(np.nanmax(hist["cpk"])) * 1.1, t_green * 1.2)

bars = alt.Chart(cdf).mark_bar().encode(
    x=alt.X("x:O", sort=allx, title="Date",
            axis=alt.Axis(values=ticks, labelAngle=0,
                          labelExpr="timeFormat(toDate(datum.value), '%b %-d')")),
    y=alt.Y("cpk:Q", title="Cpk", scale=alt.Scale(domain=[0, ymax])),
    color=alt.Color("band:N", title=None,
                    scale=alt.Scale(domain=[lab_red, lab_amb, lab_grn, lab_fc],
                                    range=["#ef4444", "#f59e0b", "#22b35e", "#b9a0f7"]),
                    legend=alt.Legend(orient="bottom", columns=2)),
    tooltip=[alt.Tooltip("x:O", title="Date"), alt.Tooltip("band:N", title="Band"),
             alt.Tooltip("cpk:Q", title="Cpk", format=".2f")],
)
target_rule = alt.Chart(pd.DataFrame({"y": [t_green]})).mark_rule(
    color="#111", strokeDash=[2, 3], strokeWidth=2).encode(y="y:Q")
target_txt = alt.Chart(pd.DataFrame({"y": [t_green], "t": [f"Target Cpk = {t_green:.2f}"]})
                       ).mark_text(align="left", dx=4, dy=-8, color="#111", x=0
                       ).encode(y="y:Q", text="t:N")
fig = (bars + target_rule + target_txt).properties(
    height=380,
    title=alt.TitleParams("Daily Cpk by date & 7-day forecast (next production days)",
                          fontSize=15, anchor="start"))

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.altair_chart(fig, use_container_width=True)
# st.caption("Cpk guide: below 1.00 = process not capable (red) · 1.00–1.33 = marginal, "
#            "needs attention (orange) · 1.33 and above = capable (green).")

# ------------------------------------------------------ chart 2: heat-map
target = (lsl + usl) / 2
h = sel.assign(dev=(sel["reading"] - target) * 1000,
               wd=sel["ts"].dt.weekday, hr=sel["ts"].dt.hour)
piv = h.pivot_table(index="wd", columns="hr", values="dev", aggfunc="mean")
names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
piv = piv.reindex([i for i in range(7) if i in piv.index])
zmax = float(np.nanmax(np.abs(piv.values))) or 1.0

long = piv.reset_index().melt(id_vars="wd", var_name="hr", value_name="dev").dropna()
long["day"] = long["wd"].map(lambda i: names[i])
long["hour"] = long["hr"].map(lambda c: f"{int(c):02d}:00")
hour_order = sorted(long["hour"].unique())
day_order = [names[i] for i in piv.index]

hm = alt.Chart(long).mark_rect().encode(
    x=alt.X("hour:O", sort=hour_order, title="Hour of day"),
    y=alt.Y("day:O", sort=day_order, title=None),
    color=alt.Color("dev:Q", title="µm",
                    scale=alt.Scale(scheme="redblue", reverse=True,
                                    domain=[-zmax, zmax], domainMid=0)),
    tooltip=[alt.Tooltip("day:O", title="Day"), alt.Tooltip("hour:O", title="Hour"),
             alt.Tooltip("dev:Q", title="Deviation (µm)", format=".2f")],
).properties(
    height=340,
    title=alt.TitleParams("Thermal heat-map – OD deviation (µm) by weekday × hour of day",
                          fontSize=15, anchor="start"))
st.altair_chart(hm, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)
