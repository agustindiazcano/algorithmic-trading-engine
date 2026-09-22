import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Callable, List, Tuple, Optional

# =========================
# ⚙️ CONFIG
# =========================
CSV_PATH = "./PEPE_MAGIC_SIGNALS.csv"   # <- tu archivo real
OUT_SUMMARY_CSV = "rule_backtest_summary.csv"
OUT_REPORT_JSON = "rule_backtest_report.json"

# Ventanas
ZWIN = 200               # rolling z-score window
QWIN = 200               # rolling quantiles window (p90/p95/p99)
BAND_WIN = 200           # window para band metrics (si querés otra)

# Backtest
HORIZONS = [1, 3, 5, 10, 30]     # como tu meta
FEE_PER_TRADE = 0.0000          # si querés meter fees: ej 0.0004 = 4 bps
SLIPPAGE = 0.0000               # idem
USE_NEXT_BAR_ENTRY = True       # ✅ evita lookahead

# Qué columna usar como "fvl"
# - si tu CSV trae "fvl", usamos eso
# - si no, intentamos "fuerza_vl"
# - si tampoco, la calculamos con close/vol (simple)
ALPHA_FVL = 0.2639

# NIVEL: intentamos leer NIVEL (N1/N2/N3) o level_num
# si no existe, derivamos por quantiles de dcpi (fallback)
FALLBACK_LEVEL_BY_DCPI = True

# =========================
# 🧰 HELPERS
# =========================

def _to_numeric(df: pd.DataFrame, cols: List[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def rolling_z(x: pd.Series, win: int) -> pd.Series:
    mu = x.rolling(win, min_periods=win).mean()
    sd = x.rolling(win, min_periods=win).std(ddof=0)
    return (x - mu) / (sd.replace(0, np.nan))

def safe_shift(s: pd.Series, n: int) -> pd.Series:
    return s.shift(n)

def quantile_roll(s: pd.Series, win: int, q: float, shift1: bool=True) -> pd.Series:
    out = s.rolling(win, min_periods=win).quantile(q)
    return out.shift(1) if shift1 else out

def future_return(close: pd.Series, h: int) -> pd.Series:
    # ret_h = close[t+h]/close[t]-1
    return (close.shift(-h) / close) - 1.0

def pct_rank_in_band(dcpi: pd.Series, p90: pd.Series, p99: pd.Series) -> pd.Series:
    # normaliza dcpi entre p90..p99
    denom = (p99 - p90).replace(0, np.nan)
    return (dcpi - p90) / denom

def infer_level(df: pd.DataFrame) -> pd.DataFrame:
    # Usa NIVEL si existe (N1/N2/N3), sino level_num, sino fallback por quantiles de dcpi
    if "NIVEL" in df.columns:
        # normaliza strings
        lvl = df["NIVEL"].astype(str).str.upper().str.strip()
        df["level_num"] = lvl.map({"N1": 1, "N2": 2, "N3": 3})
        return df

    if "level_num" in df.columns:
        df["level_num"] = pd.to_numeric(df["level_num"], errors="coerce")
        return df

    if FALLBACK_LEVEL_BY_DCPI and "dcpi" in df.columns:
        # fallback: N1/N2/N3 por terciles de dcpi (esto NO es lo mismo que tu lógica original, pero te deja correr)
        q1 = df["dcpi"].quantile(0.333)
        q2 = df["dcpi"].quantile(0.666)
        df["level_num"] = np.where(df["dcpi"] <= q1, 1, np.where(df["dcpi"] <= q2, 2, 3))
        return df

    df["level_num"] = np.nan
    return df

def compute_fvl(df: pd.DataFrame) -> pd.DataFrame:
    # preferimos fvl ya calculada
    if "fvl" in df.columns:
        df["fvl_used"] = pd.to_numeric(df["fvl"], errors="coerce")
        return df

    if "fuerza_vl" in df.columns:
        df["fvl_used"] = pd.to_numeric(df["fuerza_vl"], errors="coerce")
        return df

    # fallback: calculo simple con close + vol si está
    if "close" in df.columns:
        v = df["close"].diff()
        a = v.diff()
        if "vol" in df.columns:
            df["fvl_used"] = (df["vol"].clip(lower=0) ** ALPHA_FVL) * a
        else:
            df["fvl_used"] = a
        return df

    df["fvl_used"] = np.nan
    return df

def compute_band_flags(df: pd.DataFrame) -> pd.DataFrame:
    # band_width basado en p99 - p90 (como en tus columnas)
    if "p99_dynamic" in df.columns and "p90_dynamic" in df.columns:
        bw = df["p99_dynamic"] - df["p90_dynamic"]
    elif "p99" in df.columns and "p90" in df.columns:
        bw = df["p99"] - df["p90"]
    else:
        bw = pd.Series(np.nan, index=df.index)

    df["band_width"] = bw
    df["band_width_z"] = rolling_z(df["band_width"], ZWIN)

    # expansión/contracción por delta del ancho (simple y estable)
    dbw = df["band_width"].diff()
    df["band_expand"] = (dbw > 0).astype(int)
    df["band_contract"] = (dbw < 0).astype(int)

    return df

# =========================
# 🧠 FEATURE PIPELINE
# =========================
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # orden por tiempo si existe ts
    if "ts" in df.columns:
        # si ts es timestamp, igual ordenamos
        df = df.sort_values("ts").reset_index(drop=True)

    # numeric conversion
    _to_numeric(df, ["close", "dcpi", "vol", "df_vascular", "p90_dynamic", "p95_dynamic", "p99_dynamic"])

    # quantiles dinámicos si no vienen ya
    if "dcpi" in df.columns:
        if "p90_dynamic" not in df.columns:
            df["p90_dynamic"] = quantile_roll(df["dcpi"], QWIN, 0.90, shift1=True)
        if "p95_dynamic" not in df.columns:
            df["p95_dynamic"] = quantile_roll(df["dcpi"], QWIN, 0.95, shift1=True)
        if "p99_dynamic" not in df.columns:
            df["p99_dynamic"] = quantile_roll(df["dcpi"], QWIN, 0.99, shift1=True)

    # aliases cortos (útil)
    df["p90"] = df.get("p90_dynamic")
    df["p95"] = df.get("p95_dynamic")
    df["p99"] = df.get("p99_dynamic")

    # over flags (dinámicos, shift(1) ya aplicado en quantiles)
    df["over_p90"] = (df["dcpi"] > df["p90"]).astype(int)
    df["over_p95"] = (df["dcpi"] > df["p95"]).astype(int)
    df["over_p99"] = (df["dcpi"] > df["p99"]).astype(int)

    # pos 90..99 (entre p90 y p99)
    df["dcpi_pos_90_99"] = ((df["dcpi"] > df["p90"]) & (df["dcpi"] <= df["p99"])).astype(int)

    # pos_norm (normaliza dentro de p90..p99)
    df["pos_norm"] = pct_rank_in_band(df["dcpi"], df["p90"], df["p99"])
    df["pos_z"] = rolling_z(df["pos_norm"], ZWIN)

    # deltas
    df["dcpi_delta"] = df["dcpi"].diff()

    # fvl
    df = compute_fvl(df)
    df["fvl_abs"] = df["fvl_used"].abs()
    df["fvl_sign"] = np.sign(df["fvl_used"]).replace(0, 0)
    df["fvl_z"] = rolling_z(df["fvl_used"], ZWIN)
    df["fvl_delta"] = df["fvl_used"].diff()

    # dcpi z
    df["dcpi_z"] = rolling_z(df["dcpi"], ZWIN)

    # band metrics
    df = compute_band_flags(df)

    # nivel
    df = infer_level(df)
    df["level_change"] = (df["level_num"].diff().fillna(0) != 0).astype(int)

    # flip fvl
    df["fvl_flip"] = (np.sign(df["fvl_used"]).diff().fillna(0) != 0).astype(int)

    # returns futuros
    for h in HORIZONS:
        df[f"ret_{h}"] = future_return(df["close"], h)

    # limpieza mínima: necesitamos close/dcpi/p90/p99 + ret_*
    df = df.dropna(subset=["close", "dcpi", "p90", "p99"] + [f"ret_{h}" for h in HORIZONS]).reset_index(drop=True)
    return df

# =========================
# 🎯 RULES (estrategias “como el JSON”)
# =========================

def rule_overp99_and_fvl_neg(df: pd.DataFrame) -> pd.Series:
    return (df["over_p99"] == 1) & (df["fvl_sign"] < 0)

def rule_overp99_and_fvl_pos(df: pd.DataFrame) -> pd.Series:
    return (df["over_p99"] == 1) & (df["fvl_sign"] > 0)

def rule_n3_and_fvl_neg(df: pd.DataFrame) -> pd.Series:
    return (df["level_num"] == 3) & (df["fvl_sign"] < 0)

def rule_n2_and_fvl_neg(df: pd.DataFrame) -> pd.Series:
    return (df["level_num"] == 2) & (df["fvl_sign"] < 0)

def rule_n1_and_fvl_neg(df: pd.DataFrame) -> pd.Series:
    return (df["level_num"] == 1) & (df["fvl_sign"] < 0)

def rule_poshi_90_99(df: pd.DataFrame) -> pd.Series:
    # “pos high” dentro de 90..99: arriba del midpoint del rango
    # (si vos tenías otra definición, lo ajustamos, pero esto captura “zona alta del canal”)
    return (df["dcpi_pos_90_99"] == 1) & (df["pos_norm"] >= 0.5)

def rule_poslo_90_99(df: pd.DataFrame) -> pd.Series:
    return (df["dcpi_pos_90_99"] == 1) & (df["pos_norm"] < 0.5)

def rule_n3_fvl_pos_band_contract(df: pd.DataFrame) -> pd.Series:
    return (df["level_num"] == 3) & (df["fvl_sign"] > 0) & (df["band_contract"] == 1)

def rule_n3_fvl_pos_band_expand(df: pd.DataFrame) -> pd.Series:
    return (df["level_num"] == 3) & (df["fvl_sign"] > 0) & (df["band_expand"] == 1)

# 🧩 Pack
RULES: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "OverP99_and_fvl_neg": rule_overp99_and_fvl_neg,
    "OverP99_and_fvl_pos": rule_overp99_and_fvl_pos,
    "N3_and_fvl_neg": rule_n3_and_fvl_neg,
    "N2_and_fvl_neg": rule_n2_and_fvl_neg,
    "N1_and_fvl_neg": rule_n1_and_fvl_neg,
    "PosHi_90_99": rule_poshi_90_99,
    "PosLo_90_99": rule_poslo_90_99,
    "N3_fvl_pos_band_contract": rule_n3_fvl_pos_band_contract,
    "N3_fvl_pos_band_expand": rule_n3_fvl_pos_band_expand,
}

# =========================
# 📈 BACKTEST (hold fixed horizon)
# =========================

@dataclass
class RuleStats:
    rule: str
    direction: str  # LONG / SHORT
    count: int
    horizon: int
    mean: float
    median: float
    winrate: float
    p05: float

def eval_rule_on_horizon(
    df: pd.DataFrame,
    mask: pd.Series,
    horizon: int,
    direction: str
) -> Optional[RuleStats]:
    ret = df[f"ret_{horizon}"].copy()

    # entrada en la próxima vela para evitar lookahead (si está activado)
    # => si señal en t, la ganancia se mide desde t+1 hasta t+1+h
    if USE_NEXT_BAR_ENTRY:
        mask = mask.shift(1).fillna(False)
        ret = (df["close"].shift(-(horizon+1)) / df["close"].shift(-1)) - 1.0

    x = ret[mask].dropna()
    n = int(x.shape[0])
    if n < 20:
        return None

    # dirección
    if direction.upper() == "SHORT":
        x = -x

    # fees + slippage (simple: 1 trade por señal)
    x = x - (FEE_PER_TRADE + SLIPPAGE)

    mean = float(x.mean())
    median = float(x.median())
    winrate = float((x > 0).mean())
    p05 = float(np.quantile(x, 0.05))

    return RuleStats(
        rule="",
        direction=direction.upper(),
        count=n,
        horizon=horizon,
        mean=mean,
        median=median,
        winrate=winrate,
        p05=p05
    )

def score_10(mean_10: float, p05_10: float, winrate_10: float) -> float:
    # Score parecido a lo que venías usando: penaliza cola izquierda, recompensa media y winrate
    # Ajustalo a gusto 😼
    return (mean_10 * 100.0) + ((winrate_10 - 0.5) * 10.0) + (p05_10 * 50.0)

def run_all(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    rows = []
    report = {"meta": {}, "rules": {}}

    report["meta"] = {
        "rows": int(df.shape[0]),
        "horizons": HORIZONS,
        "zwin": ZWIN,
        "qwin": QWIN,
        "next_bar_entry": USE_NEXT_BAR_ENTRY,
        "fee_per_trade": FEE_PER_TRADE,
        "slippage": SLIPPAGE,
        "columns_inferred": {
            "has_p90_dynamic": "p90_dynamic" in df.columns,
            "has_p99_dynamic": "p99_dynamic" in df.columns,
            "has_fvl": "fvl" in df.columns,
            "has_fuerza_vl": "fuerza_vl" in df.columns,
            "has_nivel": "NIVEL" in df.columns,
        }
    }

    for rname, rfn in RULES.items():
        mask = rfn(df)
        out_rule = {"rule": rname, "count": int(mask.sum())}

        # evaluamos LONG y SHORT
        for direction in ["LONG", "SHORT"]:
            stats_by_h = {}
            stats10 = None

            for h in HORIZONS:
                st = eval_rule_on_horizon(df, mask, h, direction)
                if st is None:
                    continue
                st.rule = rname
                stats_by_h[h] = {
                    "mean": st.mean,
                    "median": st.median,
                    "winrate": st.winrate,
                    "n": st.count,
                    "p05": st.p05
                }
                rows.append({
                    "rule": rname,
                    "direction": direction,
                    "count": int(mask.sum()),
                    "horizon": h,
                    "n": st.count,
                    "mean": st.mean,
                    "median": st.median,
                    "winrate": st.winrate,
                    "p05": st.p05,
                })
                if h == 10:
                    stats10 = st

            if stats10 is not None:
                sc = score_10(stats10.mean, stats10.p05, stats10.winrate)
            else:
                sc = None

            out_rule[direction] = {
                "by_horizon": stats_by_h,
                "score_10": sc
            }

        report["rules"][rname] = out_rule

    summary = pd.DataFrame(rows)

    # ranking: ordenamos por score_10 (h=10) si existe
    if not summary.empty:
        # armamos score_10 en el summary
        def _score_row(row):
            if row["horizon"] != 10:
                return np.nan
            return score_10(row["mean"], row["p05"], row["winrate"])

        summary["score_10"] = summary.apply(_score_row, axis=1)

    return summary, report

# =========================
# 🚀 MAIN
# =========================
def main():
    print("📂 Cargando CSV… 😼")
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Rows raw: {len(df):,}")

    print("🧠 Construyendo features tipo JSON… 🧪")
    df2 = build_features(df)
    print(f"✅ Rows usable (post-feature): {len(df2):,}")
    print(f"🧾 Columns: {len(df2.columns)}")

    print("⚔️ Corriendo reglas (LONG y SHORT) por horizontes…")
    summary, report = run_all(df2)

    if summary.empty:
        print("❌ No hubo suficientes señales / datos para reportar (n<20 en todas).")
        return

    # Pivot lindo: mejor por score_10
    best10 = summary[summary["horizon"] == 10].copy()
    best10 = best10.sort_values("score_10", ascending=False)

    print("\n🏁 TOP por score_10 (h=10) 😼📈")
    cols_show = ["rule", "direction", "count", "n", "mean", "median", "winrate", "p05", "score_10"]
    print(best10[cols_show].head(20).to_string(index=False))

    # Guardar outputs
    summary.to_csv(OUT_SUMMARY_CSV, index=False)
    import json
    with open(OUT_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n💾 Guardado:")
    print(f" - CSV:  {OUT_SUMMARY_CSV}")
    print(f" - JSON: {OUT_REPORT_JSON}")
    print("✅ Listo 😼")

if __name__ == "__main__":
    main()
