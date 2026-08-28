
import streamlit as st
import pandas as pd

from model import (
    APP_VERSION,
    MODEL_VERSION,
    fetch_today_games,
    run_model,
    implied_prob,
    expected_value,
    fair_ml,
)

st.set_page_config(
    page_title="MLB Model",
    page_icon="⚾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.block-container {
    max-width: 760px;
    padding-top: 1rem;
    padding-bottom: 3rem;
}
div[data-testid="stMetricValue"] {
    font-size: 1.7rem;
}
.stButton > button {
    width: 100%;
    height: 3.2rem;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("⚾ MLB Model")
st.caption(f"App {APP_VERSION} • Engine {MODEL_VERSION}")

if "games" not in st.session_state:
    with st.spinner("Loading today's MLB schedule..."):
        st.session_state.games = fetch_today_games()

if st.button("🔄 Refresh Today's Games"):
    with st.spinner("Refreshing schedule..."):
        st.session_state.games = fetch_today_games()
    st.rerun()

games = st.session_state.games

if not games:
    st.warning("No MLB games were found for today.")
    st.stop()

labels = {}
for g in games:
    away_sp = g["Away_SP"] or "TBD"
    home_sp = g["Home_SP"] or "TBD"
    time_text = f" — {g['TimeLabel']} ET" if g["TimeLabel"] else ""
    label = f"{g['Away']} @ {g['Home']}{time_text} | {away_sp} vs {home_sp}"
    labels[label] = g["GamePk"]

mode = st.radio(
    "Run mode",
    ["Single Game", "Full Slate"],
    horizontal=True,
)

selected_game = None

if mode == "Single Game":
    selected_label = st.selectbox("Game", list(labels.keys()))
    selected_pk = labels[selected_label]
    selected_game = next(g for g in games if g["GamePk"] == selected_pk)

    st.info(
        f"**Probable pitchers:** {selected_game['Away_SP'] or 'TBD'} vs "
        f"{selected_game['Home_SP'] or 'TBD'}"
    )

run_clicked = st.button("▶️ Run Model", type="primary")

if run_clicked:
    selected_games = [selected_game] if mode == "Single Game" else games

    with st.spinner("Running MLB model. Statcast and bullpen data can take a little while..."):
        try:
            df = run_model(selected_games)
        except Exception as e:
            st.error(f"Model run failed: {e}")
            st.stop()

    if df.empty:
        st.warning("The model did not return any results.")
        st.stop()

    st.session_state.last_results = df

if "last_results" in st.session_state:
    df = st.session_state.last_results

    if len(df) == 1:
        row = df.iloc[0]

        st.divider()
        st.subheader(row["Game"])

        c1, c2, c3 = st.columns(3)
        c1.metric(row["Away"], f"{row['Away_Proj_Runs']:.2f}")
        c2.metric(row["Home"], f"{row['Home_Proj_Runs']:.2f}")
        c3.metric("Model Total", f"{row['Model_Total']:.2f}")

        c1, c2 = st.columns(2)
        c1.metric(
            f"{row['Away']} Win",
            f"{row['Away_WinProb']*100:.1f}%",
            f"Fair {int(row['Away_FairML']):+d}"
        )
        c2.metric(
            f"{row['Home']} Win",
            f"{row['Home_WinProb']*100:.1f}%",
            f"Fair {int(row['Home_FairML']):+d}"
        )

        st.caption(
            f"Confidence: {int(row['Model_Confidence'])}/100 "
            f"({row['Confidence_Grade']}) • {row['Data_Status']}"
        )

        with st.expander("Starter / bullpen details"):
            detail = pd.DataFrame({
                "": [row["Away"], row["Home"]],
                "Starter": [row["Away_SP"], row["Home_SP"]],
                "SP Role": [row["Away_SP_Role"], row["Home_SP_Role"]],
                "SP Quality": [row["Away_SP_Quality"], row["Home_SP_Quality"]],
                "Expected IP": [row["Away_SP_ExpIP"], row["Home_SP_ExpIP"]],
                "Bullpen Quality": [row["Away_BP_Quality"], row["Home_BP_Quality"]],
                "Bullpen Avail.": [row["Away_BP_Availability"], row["Home_BP_Availability"]],
            })
            st.dataframe(detail, hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Sportsbook Lines")
        st.caption("Optional — enter current odds to calculate model edge and EV.")

        away = row["Away"]
        home = row["Home"]

        ml1, ml2 = st.columns(2)
        away_ml = ml1.number_input(f"{away} ML", value=100, step=5, key="away_ml")
        home_ml = ml2.number_input(f"{home} ML", value=-110, step=5, key="home_ml")

        rl1, rl2 = st.columns(2)
        away_rl_side = rl1.selectbox(
            f"{away} run line",
            ["+1.5", "-1.5"],
            index=0,
            key="away_rl_side",
        )
        away_rl_odds = rl1.number_input(f"{away} RL odds", value=-110, step=5, key="away_rl_odds")

        home_rl_side = rl2.selectbox(
            f"{home} run line",
            ["-1.5", "+1.5"],
            index=0,
            key="home_rl_side",
        )
        home_rl_odds = rl2.number_input(f"{home} RL odds", value=100, step=5, key="home_rl_odds")

        t1, t2, t3 = st.columns(3)
        total_line = t1.number_input("Total", value=9.0, step=0.5, key="total_line")
        over_odds = t2.number_input("Over odds", value=-110, step=5, key="over_odds")
        under_odds = t3.number_input("Under odds", value=-110, step=5, key="under_odds")

        if st.button("Calculate Market Edge"):
            market_rows = []

            def add_market(name, model_prob, odds):
                ip = implied_prob(odds)
                ev = expected_value(model_prob, odds)
                edge = model_prob - ip
                market_rows.append({
                    "Bet": name,
                    "Odds": int(odds),
                    "Model Prob": model_prob,
                    "Implied Prob": ip,
                    "Edge": edge,
                    "EV": ev,
                    "Model Fair": fair_ml(model_prob),
                })

            add_market(f"{away} ML", row["Away_WinProb"], away_ml)
            add_market(f"{home} ML", row["Home_WinProb"], home_ml)

            away_rl_prob = row["Away_+1.5_Prob"] if away_rl_side == "+1.5" else row["Away_-1.5_Prob"]
            home_rl_prob = row["Home_+1.5_Prob"] if home_rl_side == "+1.5" else row["Home_-1.5_Prob"]

            add_market(f"{away} {away_rl_side}", away_rl_prob, away_rl_odds)
            add_market(f"{home} {home_rl_side}", home_rl_prob, home_rl_odds)

            # For totals, the current v0.5.2 engine produces a mean total but not
            # a calibrated total distribution. Show projection gap only rather than
            # inventing an Over/Under win probability.
            market_df = pd.DataFrame(market_rows).sort_values("EV", ascending=False)

            show = market_df.copy()
            show["Model Prob"] = (show["Model Prob"] * 100).map(lambda x: f"{x:.1f}%")
            show["Implied Prob"] = (show["Implied Prob"] * 100).map(lambda x: f"{x:.1f}%")
            show["Edge"] = (show["Edge"] * 100).map(lambda x: f"{x:+.1f}%")
            show["EV"] = (show["EV"] * 100).map(lambda x: f"{x:+.1f}%")
            show["Model Fair"] = show["Model Fair"].map(lambda x: f"{int(x):+d}")

            st.dataframe(show, hide_index=True, use_container_width=True)

            gap = row["Model_Total"] - total_line
            st.write(
                f"**Total:** model {row['Model_Total']:.2f} vs market {total_line:.1f} "
                f"→ projection gap **{gap:+.2f} runs**."
            )
            st.caption(
                "The current engine does not yet produce a calibrated Over/Under probability, "
                "so the app does not calculate total EV."
            )

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Result CSV",
            data=csv,
            file_name=f"mlb_model_{row['Away'].replace(' ', '_')}_at_{row['Home'].replace(' ', '_')}.csv",
            mime="text/csv",
        )

    else:
        st.divider()
        st.subheader("Full Slate")

        table = df[[
            "Game",
            "Away_Proj_Runs",
            "Home_Proj_Runs",
            "Model_Total",
            "Away_WinProb",
            "Home_WinProb",
            "Away_FairML",
            "Home_FairML",
            "Model_Confidence",
            "Confidence_Grade",
        ]].copy()

        table["Away_WinProb"] = (table["Away_WinProb"] * 100).round(1)
        table["Home_WinProb"] = (table["Home_WinProb"] * 100).round(1)

        st.dataframe(table, hide_index=True, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Full Slate CSV",
            data=csv,
            file_name="mlb_model_full_slate.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Run-line probabilities are experimental. Market odds are entered manually. "
    "The model is for analytical use and does not guarantee outcomes."
)
