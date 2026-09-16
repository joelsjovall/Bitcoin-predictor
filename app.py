"""Frontend: Streamlit-app som visar Bitcoin-data och en prisprognos."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import ingest_csv, ingest_live, load_prices_df
from model import FORECAST_HORIZONS, METHODS, predict_price, train_model
from macro import MACRO_FEATURE_COLUMNS, build_macro_features, fetch_macro_prices

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


@st.cache_data(ttl=3600)
def get_macro_data(start: str) -> pd.DataFrame:
    return fetch_macro_prices(start)


@st.cache_resource
def get_macro_model_and_metrics(selected_method: str, selected_weeks: int, bitcoin_data: pd.DataFrame, macro_data: pd.DataFrame):
    return train_model(bitcoin_data, method=selected_method, horizon_weeks=selected_weeks, macro_df=macro_data)


@st.cache_resource
def get_model_and_metrics(selected_method: str, selected_weeks: int):
    return train_model(get_data(), method=selected_method, horizon_weeks=selected_weeks)


def show_evaluation(model_metrics):
    price_error, return_error = st.columns(2)
    price_error.metric(
        "RMSE (test, USD)", f"{model_metrics['rmse']:,.0f}",
        help=f"Naiv RMSE (oförändrat pris): {model_metrics['naive_rmse']:,.0f} USD",
    )
    return_error.metric(
        "RMSE (test, avkastning i procentenheter)", f"{model_metrics['return_rmse_pct']:.2f}",
    )
    st.caption(
        f"Bitcoin-historik: {model_metrics['data_start']:%Y-%m-%d} – {model_metrics['data_end']:%Y-%m-%d}. "
        f"Slutmodellen tränas om på alla {model_metrics['n_production']} kompletta exempel: "
        f"{model_metrics['production_train_start']:%Y-%m-%d} – {model_metrics['production_train_end']:%Y-%m-%d}, "
        f"med kända utfall till {model_metrics['production_target_end']:%Y-%m-%d}."
    )
    st.caption(
        f"RMSE-testets prognosdatum: {model_metrics['test_start']:%Y-%m-%d} – {model_metrics['test_end']:%Y-%m-%d}. "
        f"Utfallen som jämförs ligger {model_metrics['test_target_start']:%Y-%m-%d} – "
        f"{model_metrics['test_target_end']:%Y-%m-%d} ({model_metrics['n_test']} exempel). "
        "Testperioden är inte slutmodellens hela träningsperiod."
    )


df = get_data()
latest_date = pd.to_datetime(df["date"]).max()
if pd.Timestamp.now().normalize() - latest_date > pd.Timedelta(days=7):
    st.warning(
        f"Senaste Bitcoin-observation är {latest_date:%Y-%m-%d}. "
        "Välj Hämta senaste data (live) i sidopanelen för att uppdatera till senaste avslutade vecka."
    )
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

col1, col2 = st.columns(2)
col1.metric("Senaste pris", f"{last_row['price']:,.0f}", help=str(last_row["date"].date()))
col2.metric(f"Prognos om {horizon_label}", f"{next_price:,.0f}", f"{change_pct:+.1f}%")
show_evaluation(metrics)
st.info(
    f"Prognoserna utgår från senaste Bitcoin-observationen {last_row['date']:%Y-%m-%d} "
    f"och gäller {forecast_date:%Y-%m-%d}. Träning kräver ett känt utfall {weeks_ahead} veckor senare; "
    "de senaste veckorna används som prognosindata men kan inte vara träningsmål innan utfallet finns. "
    "De första 52 veckorna används för att bygga historiska features."
)

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

st.subheader(f"Bitcoin + makro – {method} ({horizon_label} framåt)")
st.caption("Separat Bitcoin-prognos: samtliga Bitcoin-features + S&P 500, guld och amerikansk tioårsränta.")
if st.button("Uppdatera makrodata"):
    get_macro_data.clear()
    get_macro_model_and_metrics.clear()

try:
    with st.spinner("Hämtar makrodata och tränar Bitcoin + makro…"):
        macro_data = get_macro_data((df["date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
        macro_model, macro_metrics = get_macro_model_and_metrics(method, weeks_ahead, df, macro_data)
        macro_price = predict_price(df, macro_model, horizon_weeks=weeks_ahead, macro_df=macro_data)
    macro_change_pct = (macro_price / last_row["price"] - 1) * 100
    st.metric(f"Makroprognos om {horizon_label}", f"{macro_price:,.0f} USD", f"{macro_change_pct:+.1f}%")
    show_evaluation(macro_metrics)
    macro_fig = go.Figure()
    macro_fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name="Bitcoin-pris", mode="lines"))
    macro_fig.add_trace(go.Scatter(
        x=[last_row["date"], forecast_date], y=[last_row["price"], macro_price],
        name="Prognos: Bitcoin + makro", mode="lines+markers", line=dict(dash="dot", color="orange"),
    ))
    macro_fig.update_layout(fig.layout)
    st.plotly_chart(macro_fig, width="stretch")
    st.caption(
        f"MAE: {macro_metrics['mae']:,.0f} USD. "
        f"Hyperparametertuning: {'ja' if macro_metrics['tuned'] else 'nej, för kort historik'}."
    )
    if macro_metrics["test_dates"] != metrics["test_dates"]:
        st.warning("Modellerna har olika testdatum på grund av datatäckningen. Deras RMSE är inte direkt jämförbara.")
    else:
        st.caption("Båda modellerna utvärderas på samma testdatum.")
    with st.expander("Makrofaktorer och enheter"):
        st.write(
            "S&P 500 och guld: 12- och 52-veckors prisavkastning (decimalform i modellen, % nedan). "
            "Tioårsränta: nivå i % och 12-veckors förändring i procentenheter, inte obligationsavkastning. "
            "Grafen visar prognostiserat Bitcoin-pris i USD; prognosens avkastning visas ovan. "
            "Inga framtida makrovärden matas in. Saknad eller mer än sju dagar gammal data fylls inte bakåt."
        )
        st.markdown(
            "Källor: Yahoo Finance [S&P 500 (^GSPC)](https://finance.yahoo.com/quote/%5EGSPC/), "
            "[guldterminer (GC=F)](https://finance.yahoo.com/quote/GC%3DF/) som guldproxy, "
            "[tioårsränta (^TNX)](https://finance.yahoo.com/quote/%5ETNX/). "
            "Prisavkastningen är inte totalavkastning; guldterminernas kontraktsbyten kan påverka serien."
        )
        macro_table = build_macro_features(df["date"], macro_data).tail(10).set_index("date")
        return_columns = [column for column in MACRO_FEATURE_COLUMNS if "_return_" in column]
        macro_table[return_columns] *= 100
        macro_table = macro_table.rename(columns={
            column: f"{column} ({'procentenheter' if '_change_' in column else '%'})"
            for column in MACRO_FEATURE_COLUMNS
        })
        st.dataframe(macro_table.sort_index(ascending=False), width="stretch")
except Exception as error:
    st.warning(f"Makroprognosen kunde inte visas: {error}")
    st.caption("Bitcoin-modellen ovan påverkas inte. Kontrollera anslutningen och välj Uppdatera makrodata för att försöka igen.")
