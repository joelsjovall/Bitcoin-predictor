"""Frontend: Streamlit-app som visar Bitcoin-data och en prisprognos."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import ingest_csv, ingest_live, load_prices_df
from model import build_features, predict_next_price, train_model

st.set_page_config(page_title="Bitcoin – pris & prognos", layout="wide")
st.title("Bitcoin – historik och prognos")

with st.sidebar:
    st.header("Data & modell")
    if st.button("Läs in CSV på nytt i databasen"):
        n = ingest_csv()
        st.success(f"Laddade in {n} rader.")
        st.cache_data.clear()
    if st.button("Hämta senaste data (live)"):
        try:
            n = ingest_live()
            st.success(f"Hämtade/uppdaterade {n} rader från Yahoo Finance.")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Kunde inte hämta live-data: {e}")
    if st.button("Träna om modellen"):
        st.cache_resource.clear()


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_prices_df()


@st.cache_resource
def get_model_and_metrics():
    return train_model(get_data())


df = get_data()
model, metrics = get_model_and_metrics()
next_price = predict_next_price(df, model)
last_row = df.iloc[-1]
change_pct = (next_price - last_row["price"]) / last_row["price"] * 100

col1, col2, col3 = st.columns(3)
col1.metric("Senaste pris", f"{last_row['price']:,.0f}", help=str(last_row["date"].date()))
col2.metric("Prognos nästa vecka", f"{next_price:,.0f}", f"{change_pct:+.1f}%")
col3.metric("Modellens MAE (test)", f"{metrics['mae']:,.0f}", f"naiv: {metrics['naive_mae']:,.0f}")

st.subheader("Prishistorik")
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
fig.update_layout(xaxis_title="Datum", yaxis_title="Pris (USD)", height=500)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Senaste veckorna")
st.dataframe(
    df.sort_values("date", ascending=False).head(10).set_index("date"),
    use_container_width=True,
)

with st.expander("Om modellen"):
    st.write(
        f"""
        En RandomForestRegressor tränad på {metrics['n_train']} veckor och testad på
        {metrics['n_test']} veckor (senaste delen av tidsserien).

        - MAE: {metrics['mae']:.1f} (naiv baseline, dvs. "nästa vecka = samma som denna": {metrics['naive_mae']:.1f})
        - RMSE: {metrics['rmse']:.1f} (naiv baseline: {metrics['naive_rmse']:.1f})
        - R²: {metrics['r2']:.3f}

        Detta är en proof of concept – flödet (data → databas → AI-modell → frontend)
        är det viktiga, inte modellens exakta träffsäkerhet.
        """
    )
