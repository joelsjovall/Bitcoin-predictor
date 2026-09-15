"""Frontend: Streamlit-app som visar Bitcoin-data och en prisprognos."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import ingest_csv, ingest_live, load_prices_df
from model import FORECAST_HORIZONS, METHODS, predict_price, train_model

st.set_page_config(page_title="Bitcoin – pris & prognos", layout="wide")
st.title("Bitcoin – historik och prognos")

with st.sidebar:
    st.header("Data & modell")
    if st.button("Läs in CSV på nytt i databasen"):
        n = ingest_csv()
        st.success(f"Laddade in {n} rader.")
        st.cache_data.clear()
        st.cache_resource.clear()
    if st.button("Hämta senaste data (live)"):
        try:
            n = ingest_live()
            st.success(f"Hämtade/uppdaterade {n} rader från Yahoo Finance.")
            st.cache_data.clear()
            st.cache_resource.clear()
        except Exception as e:
            st.error(f"Kunde inte hämta live-data: {e}")
    if st.button("Träna om modellen"):
        st.cache_resource.clear()

method = st.segmented_control(
    "AI-metod",
    options=list(METHODS.keys()),
    default="Random Forest",
)
if method is None:
    method = "Random Forest"

horizon_label = st.segmented_control(
    "Prognoshorisont",
    options=list(FORECAST_HORIZONS.keys()),
    default="1 månad",
)
if horizon_label is None:
    horizon_label = "1 månad"
weeks_ahead = FORECAST_HORIZONS[horizon_label]


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_prices_df()


@st.cache_resource
def get_model_and_metrics(selected_method: str, selected_weeks: int):
    return train_model(get_data(), method=selected_method, horizon_weeks=selected_weeks)


df = get_data()
model, metrics = get_model_and_metrics(method, weeks_ahead)
next_price = predict_price(df, model, horizon_weeks=weeks_ahead)
last_row = df.iloc[-1]
change_pct = (next_price - last_row["price"]) / last_row["price"] * 100
forecast_date = last_row["date"] + pd.Timedelta(weeks=weeks_ahead)

all_metrics = {m: get_model_and_metrics(m, weeks_ahead)[1] for m in METHODS}
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
col2.metric(f"Prognos om {horizon_label}", f"{next_price:,.0f}", f"{change_pct:+.1f}%")
col3.metric("Modellens RMSE (test)", f"{metrics['rmse']:,.0f}", f"naiv: {metrics['naive_rmse']:,.0f}")

st.subheader(f"Prishistorik – {method} ({horizon_label} framåt)")

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
    if metrics["tuned"]:
        methodology = f"""
        **{method}** tränad direkt mot horisonten **{horizon_label}** ("vad blir priset
        om {weeks_ahead} veckor?") med en kronologisk **tränings-/validerings-/test**-
        uppdelning: {metrics['n_train']} tränings-, {metrics['n_val']} validerings- och
        {metrics['n_test']} testexempel (äldst → nyast).

        1. Ett par hyperparameter-kandidater tränas på träningsdelen och jämförs på
           valideringsdelen – bästa valet: `{metrics['best_params'] or "standardvärden"}`
           (val-RMSE: {metrics['val_rmse']:.1f}).
        2. De vinnande hyperparametrarna tränas om på träning+validering och testas en
           **enda gång** på den helt osedda testdelen – det är detta som rapporteras
           nedan som modellens riktiga prestanda.
        3. Den modell som faktiskt gör prognosen ovan tränas därefter om en sista gång
           på **all** tillgänglig data (träning+validering+test), så att prognosen får
           utnyttja så mycket historik som möjligt. Testdelens enda syfte är att ge en
           ärlig uppskattning av träffsäkerheten – inte att vara med i den slutliga
           prognosmodellen.
        """
    else:
        methodology = f"""
        **{method}** tränad direkt mot horisonten **{horizon_label}** ("vad blir priset
        om {weeks_ahead} veckor?"). Vår ~16-åriga historik räcker inte till tre helt
        separata, läckagefria fönster vid så här lång horisont (varje fönster behöver
        egen marginal på minst {weeks_ahead} veckor) – appen föll därför tillbaka på en
        enklare uppdelning: {metrics['n_train']} tränings- och {metrics['n_test']}
        testexempel (äldst → nyast), med metodens standardhyperparametrar (ingen
        validerings-tuning för den här horisonten).
        """
    st.write(
        methodology
        + f"""
        **Testresultat (helt osedd data):**
        - MAE: {metrics['mae']:.1f} (naiv baseline, dvs. "priset om {weeks_ahead} veckor
          = samma som nu": {metrics['naive_mae']:.1f})
        - RMSE: {metrics['rmse']:.1f} (naiv baseline: {metrics['naive_rmse']:.1f})
        - R²: {metrics['r2']:.3f}

        Detta är en proof of concept – flödet (data → databas → AI-modell → frontend)
        är det viktiga, inte modellens exakta träffsäkerhet.

        Varje horisont (1 månad, 3 månader, ..., 5 år) har en egen modell som tränas
        direkt mot faktiska historiska N-veckors-förändringar, istället för att kedja
        ihop upprepade enveckasprognoser. Det ger en mer statistiskt rimlig prognos,
        men ju längre horisont desto färre historiska exempel finns att träna på och
        desto osäkrare är prognosen – en 5-årsprognos ska ses som en illustration av
        flödet, inte som en tillförlitlig prisprognos.
        """
    )

    st.write("**Jämförelse mellan metoder (test-RMSE, lägre är bättre):**")
    comparison = pd.DataFrame(
        [{"Metod": m, "RMSE": all_metrics[m]["rmse"], "MAE": all_metrics[m]["mae"]} for m in METHODS]
    ).sort_values("RMSE")
    st.dataframe(comparison.set_index("Metod"), width="stretch")
