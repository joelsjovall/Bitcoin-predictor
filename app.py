"""Frontend: Streamlit-app som visar Bitcoin-data och en prisprognos."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import ingest_csv, load_prices_df
from model import METHODS, build_features, predict_next_price, train_model

st.set_page_config(page_title="Bitcoin – pris & prognos", layout="wide")
st.title("Bitcoin – historik och prognos")

with st.sidebar:
    st.header("Data & modell")
    if st.button("Läs in CSV på nytt i databasen"):
        n = ingest_csv()
        st.success(f"Laddade in {n} rader.")
        st.cache_data.clear()
    if st.button("Träna om modellen"):
        st.cache_resource.clear()

method = st.segmented_control(
    "AI-metod",
    options=list(METHODS.keys()),
    default="Random Forest",
)
if method is None:
    method = "Random Forest"


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_prices_df()


@st.cache_resource
def get_model_and_metrics(selected_method: str):
    return train_model(get_data(), method=selected_method)


df = get_data()
model, metrics = get_model_and_metrics(method)
next_price = predict_next_price(df, model)
last_row = df.iloc[-1]
change_pct = (next_price - last_row["price"]) / last_row["price"] * 100

all_metrics = {m: get_model_and_metrics(m)[1] for m in METHODS}
best_method = min(all_metrics, key=lambda m: all_metrics[m]["rmse"])
best_rmse = all_metrics[best_method]["rmse"]

if method == best_method:
    st.success(f"🏆 **{method}** är just nu bästa metoden (lägst RMSE: {best_rmse:,.0f})")
else:
    st.info(
        f"🏆 Bästa metoden just nu är **{best_method}** "
        f"(RMSE {best_rmse:,.0f} mot {metrics['rmse']:,.0f} för {method})"
    )

col1, col2, col3 = st.columns(3)
col1.metric("Senaste pris", f"{last_row['price']:,.0f}", help=str(last_row["date"].date()))
col2.metric("Prognos nästa vecka", f"{next_price:,.0f}", f"{change_pct:+.1f}%")
col3.metric("Modellens RMSE (test)", f"{metrics['rmse']:,.0f}", f"naiv: {metrics['naive_rmse']:,.0f}")

st.subheader(f"Prishistorik – {method}")
feat = build_features(df)
forecast_date = last_row["date"] + pd.Timedelta(weeks=1)

fig = go.Figure()
fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Pris", mode="lines"))
fig.add_trace(
    go.Scatter(
        x=[last_row["date"], forecast_date],
        y=[last_row["price"], next_price],
        name="Prognos",
        mode="lines+markers",
        line=dict(dash="dot", color="orange"),
    )
)
fig.update_layout(
    xaxis_title="Datum",
    yaxis_title="Pris (USD)",
    height=550,
    xaxis=dict(
        type="date",
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Å", step="year", stepmode="backward"),
                dict(count=5, label="5Å", step="year", stepmode="backward"),
                dict(step="all", label="Allt"),
            ]
        ),
        rangeslider=dict(visible=True),
    ),
)
st.plotly_chart(fig, width="stretch")

st.subheader("Senaste veckorna")
st.dataframe(
    df.sort_values("date", ascending=False).head(10).set_index("date"),
    width="stretch",
)

with st.expander("Om modellen"):
    st.write(
        f"""
        **{method}** tränad på {metrics['n_train']} veckor och testad på
        {metrics['n_test']} veckor (senaste delen av tidsserien).

        - MAE: {metrics['mae']:.1f} (naiv baseline, dvs. "nästa vecka = samma som denna": {metrics['naive_mae']:.1f})
        - RMSE: {metrics['rmse']:.1f} (naiv baseline: {metrics['naive_rmse']:.1f})
        - R²: {metrics['r2']:.3f}

        Detta är en proof of concept – flödet (data → databas → AI-modell → frontend)
        är det viktiga, inte modellens exakta träffsäkerhet.
        """
    )

    st.write("**Jämförelse mellan metoder (test-RMSE, lägre är bättre):**")
    comparison = pd.DataFrame(
        [{"Metod": m, "RMSE": all_metrics[m]["rmse"], "MAE": all_metrics[m]["mae"]} for m in METHODS]
    ).sort_values("RMSE")
    st.dataframe(comparison.set_index("Metod"), width="stretch")
