"""Frontend: Streamlit-app som visar Bitcoin-data och en prisprognos."""
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import ingest_csv, ingest_live, load_prices_df
from model import (
    FORECAST_HORIZONS,
    METHODS,
    predict_price,
    replace_latest_price_for_inference,
    train_model,
)
from macro import MACRO_FEATURE_COLUMNS, build_macro_features, fetch_macro_prices

AUTO_UPDATE_TTL_SECONDS = 3 * 60 * 60  # 3 timmar – färskt nog utan att hämta om vid varje interaktion


@st.cache_data(ttl=AUTO_UPDATE_TTL_SECONDS)
def get_data() -> pd.DataFrame:
    return load_prices_df()


def ensure_fresh_data() -> tuple[pd.DataFrame, str | None]:
    """Uppdatera äldre prisdata och behåll sparade data vid nätverksfel."""
    data = get_data()
    latest_date = pd.to_datetime(data["date"]).max()
    today = pd.Timestamp.now().normalize()
    if latest_date >= today - pd.Timedelta(days=1):
        return data, None
    try:
        ingest_live()
    except Exception:
        return data, "Automatisk live-uppdatering misslyckades just nu – visar senast sparade data."
    st.cache_data.clear()
    st.cache_resource.clear()
    return get_data(), None


@st.cache_data(ttl=20 * 60)  # 20 minuter – färskt utan onödiga anrop mot Yahoo Finance
def get_current_price() -> float:
    """Hämta senaste dagspriset för visning och aktuell prognos, inte träning/test."""
    import yfinance as yf

    raw = yf.download("BTC-USD", period="5d", interval="1d", progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError("Yahoo Finance returnerade inga priser.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return float(raw["Close"].dropna().iloc[-1])


@st.cache_data(ttl=3600)
def get_macro_data(start: str) -> pd.DataFrame:
    return fetch_macro_prices(start)


@st.cache_resource
def get_macro_model_and_metrics(
    selected_method: str,
    selected_weeks: int,
    bitcoin_data: pd.DataFrame,
    macro_data: pd.DataFrame,
    training_version=4,
):
    return train_model(bitcoin_data, method=selected_method, horizon_weeks=selected_weeks, macro_df=macro_data)


@st.cache_resource
def get_model_and_metrics(selected_method: str, selected_weeks: int, training_version=4):
    return train_model(get_data(), method=selected_method, horizon_weeks=selected_weeks)


def is_positive_finite(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def is_nonnegative_finite(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def format_optional_pct(value, prefix="") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "Ej tillgänglig"
    return f"{prefix}{number:.1f} %" if math.isfinite(number) else "Ej tillgänglig"


def show_best_method(metrics_by_method, selected_method, model_name):
    comparable = {
        name: float(values["relative_price_rmse_pct"])
        for name, values in metrics_by_method.items()
        if values.get("relative_price_rmse_pct") is not None
        and math.isfinite(float(values["relative_price_rmse_pct"]))
    }
    if not comparable:
        st.info(
            f"Bästa metoden för {model_name} kan inte utses eftersom relativ RMSE saknas "
            "för samtliga metoder."
        )
        return

    best_method = min(comparable, key=comparable.get)
    best_score = comparable[best_method]
    selected_score = comparable.get(selected_method)
    if selected_method == best_method:
        st.success(
            f"**{best_method}** är bästa metoden för {model_name} "
            f"(lägst relativ RMSE: {best_score:.1f} %)"
        )
    elif selected_score is None:
        st.info(
            f"Bästa metoden för {model_name} är **{best_method}** "
            f"(relativ RMSE: {best_score:.1f} %). Relativ RMSE kan inte beräknas för {selected_method}."
        )
    else:
        st.info(
            f"Bästa metoden för {model_name} är **{best_method}** "
            f"(relativ RMSE {best_score:.1f} % mot {selected_score:.1f} % för {selected_method})"
        )


def show_historical_margin(prediction, model_metrics, source="senaste sluttestet"):
    margin = model_metrics["historical_margin_pct"]
    if not is_positive_finite(prediction):
        st.info("Historisk felmarginal kan inte appliceras eftersom den aktuella prisprognosen inte är positiv och ändlig.")
        return
    if margin is None:
        st.info(
            "Historisk felmarginal och relativ RMSE kan inte beräknas eftersom sluttestet "
            "innehåller minst en icke-positiv eller ogiltig prisprognos. RMSE i USD visas fortfarande."
        )
        return
    low = max(0, prediction * (1 - margin / 100))
    high = prediction * (1 + margin / 100)
    coverage = model_metrics.get("interval_coverage_pct")
    calibration_count = model_metrics.get("n_margin_calibration", model_metrics.get("n_test", 0))
    coverage_count = model_metrics.get("n_coverage_test", 0)
    columns = st.columns(3 if coverage is not None else 2)
    error_col, interval_col = columns[:2]
    error_col.metric(f"Kalibrerad felmarginal (80 %, {source})", f"±{margin:.1f} %")
    interval_col.metric("Prisintervall utifrån historiska fel", f"{low:,.0f}–{high:,.0f} USD")
    if coverage is not None:
        columns[2].metric(
            "Täckning i senare kontroll",
            f"{coverage:.0f} %",
            help=f"{coverage_count} senare prognoser som inte användes för att bestämma marginalen.",
        )
        coverage_explanation = (
            f"Marginalen är 80:e percentilen av de absoluta relativa prisfelen i de {calibration_count} "
            f"äldre kalibreringsprognoserna. I {coverage_count} senare kontrollprognoser hamnade "
            f"{coverage:.0f} % av utfallen inom samma marginal. Kontrollvärdet är inte tvingat till 80 %."
        )
    else:
        coverage_explanation = (
            f"Marginalen är 80:e percentilen av de absoluta relativa prisfelen i {calibration_count} "
            "historiska prognoser. Underlaget är för litet för en separat senare täckningskontroll."
        )
    with st.popover("ℹ️ Information om intervallet"):
        st.write(coverage_explanation)
        st.write(
            "Prisintervallet är dagens prognos × (1 ± marginalen). Det är en historiskt kalibrerad "
            "osäkerhetsindikator, inte en garanti eller en formell sannolikhetsprognos."
        )
        if margin > 100:
            st.write("Intervallets nedre gräns visas som 0 USD eftersom felmarginalen överstiger 100 %.")


def show_error_metrics(model_metrics, source):
    usd_col, relative_col = st.columns(2)
    usd_col.metric(
        f"RMSE ({source}, USD)", f"{model_metrics['rmse']:,.0f}",
        help=f"Naiv RMSE med oförändrat pris: {model_metrics['naive_rmse']:,.0f} USD",
    )
    relative_col.metric(
        "Relativ RMSE (kalibrering)",
        format_optional_pct(model_metrics.get("relative_price_rmse_pct")),
        help="Använder samma kalibreringsprognoser och samma procentuella felbas som 80 %-marginalen.",
    )
    with st.popover("ℹ️ Information om felmåtten"):
        if model_metrics.get("relative_price_rmse_pct") is None:
            st.write(
                "Relativ RMSE saknas eftersom minst en historisk testprognos inte var positiv och ändlig. "
                "Ett procentfel med prognospriset i nämnaren är då inte meningsfullt. RMSE i USD kan fortfarande beräknas."
            )
        else:
            st.markdown(
                """
                Både **relativ RMSE** och **80 %-felmarginalen** utgår från samma kalibreringsprognoser
                och samma relativa prisfel:

                `(utfall − prognos) / prognos × 100`

                - **Relativ RMSE** beskriver modellens samlade procentuella felstorlek. Felen kvadreras,
                  så enstaka stora missar får extra stor påverkan. Måttet används för att bedöma
                  träffsäkerhet; lägre är bättre.
                - **80 %-felmarginalen** är den 80:e percentilen av felens absolutbelopp. Den beskriver
                  hur bred marginal som behövdes för att omfatta minst 80 % av kalibreringsfelen och
                  används därför för prisintervallet runt dagens prognos.
                - **Täckningen i senare kontroll** visar hur stor andel av senare, oanvända utfall som
                  faktiskt hamnade inom den kalibrerade felmarginalen.

                Måtten behöver inte vara lika och det finns ingen fast omräkning mellan dem. Relativ
                RMSE är ett genomsnittsliknande felmått, medan felmarginalen är en historisk gräns.
                Prisintervallet är ingen garanti eller formell sannolikhetsprognos.
                """
            )
            st.write(
                "RMSE i USD använder hela sluttestet och visar prisfelets storlek i dollar. Det påverkas "
                "därför av Bitcoins prisnivå och är främst användbart för jämförelser på samma testperiod."
            )


def show_evaluation(model_metrics):
    show_error_metrics(model_metrics, "hela sluttestet")
    history = model_metrics.get("history_weeks")
    label = "hela historiken" if history is None else f"upp till {history // 52} år"
    with st.popover("ℹ️ Information om testunderlaget"):
        st.write(
            f"Bitcoin-historik: {model_metrics['data_start']:%Y-%m-%d} – {model_metrics['data_end']:%Y-%m-%d}. "
            f"Slutmodellen tränas om på {model_metrics['n_production']} kompletta exempel i vald historik: "
            f"{model_metrics['production_train_start']:%Y-%m-%d} – {model_metrics['production_train_end']:%Y-%m-%d}, "
            f"med kända utfall till {model_metrics['production_target_end']:%Y-%m-%d}."
        )
        st.write(
            f"RMSE-testets prognosdatum: {model_metrics['test_start']:%Y-%m-%d} – {model_metrics['test_end']:%Y-%m-%d}. "
            f"Utfallen som jämförs ligger {model_metrics['test_target_start']:%Y-%m-%d} – "
            f"{model_metrics['test_target_end']:%Y-%m-%d} ({model_metrics['n_test']} exempel). "
            "Testperioden är inte slutmodellens hela träningsperiod."
        )
        st.write(f"Modellen använder {label}. Historikfönstret slutar vid senaste träningsexemplet med känt utfall.")
        if model_metrics["tuned"]:
            st.write("Historiklängd och modellens parametrar väljs gemensamt på valideringsdata. Sluttestet används inte för valet.")
        else:
            st.write("För lite historik för validering: hela historiken och standardinställningar används utan optimering.")


def forecast_chart(
    df, prediction, forecast_origin_date, forecast_date, forecast_base_price,
    price_label="Pris", forecast_label="Prognos",
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["price"], name=price_label, mode="lines"))
    if is_nonnegative_finite(prediction):
        fig.add_trace(
            go.Scatter(
                x=[forecast_origin_date, forecast_date],
                y=[forecast_base_price, prediction],
                name=forecast_label,
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
    return fig


def main():
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
        options=list(METHODS),
        default="Random Forest",
    )
    if method is None:
        method = "Random Forest"

    horizon_label = st.segmented_control(
        "Prognoshorisont",
        options=list(FORECAST_HORIZONS),
        default="1 månad",
    )
    if horizon_label is None:
        horizon_label = "1 månad"
    weeks_ahead = FORECAST_HORIZONS[horizon_label]

    df, auto_update_error = ensure_fresh_data()
    latest_date = pd.to_datetime(df["date"]).max()
    if pd.Timestamp.now().normalize() - latest_date > pd.Timedelta(days=7):
        st.warning(
            f"Senaste Bitcoin-observation är {latest_date:%Y-%m-%d}. "
            "Välj Hämta senaste data (live) i sidopanelen för att uppdatera till senaste avslutade vecka."
        )
    if auto_update_error:
        st.caption(f"ℹ️ {auto_update_error}")
    try:
        current_price = get_current_price()
    except Exception:
        current_price = None

    last_row = df.iloc[-1]
    if is_positive_finite(current_price):
        inference_df = replace_latest_price_for_inference(df, current_price)
        forecast_base_price = float(current_price)
        forecast_origin_date = pd.Timestamp.now().normalize()
    else:
        inference_df = df
        forecast_base_price = float(last_row["price"])
        forecast_origin_date = pd.Timestamp(last_row["date"])

    model, metrics = get_model_and_metrics(method, weeks_ahead)
    next_price = predict_price(inference_df, model, horizon_weeks=weeks_ahead)
    prediction_is_valid = is_nonnegative_finite(next_price)
    change_pct = ((next_price - forecast_base_price) / forecast_base_price * 100
                  if prediction_is_valid else None)
    forecast_date = forecast_origin_date + pd.Timedelta(weeks=weeks_ahead)

    all_metrics = {m: get_model_and_metrics(m, weeks_ahead)[1] for m in METHODS}
    show_best_method(all_metrics, method, "Bitcoin")

    col1, col_now, col2 = st.columns(3)
    col1.metric(
        "Senaste pris (vecka)",
        f"{last_row['price']:,.0f}",
        help=f"Veckosnapshotet från {last_row['date'].date()} som modellen tränas och testas på.",
    )
    if is_positive_finite(current_price):
        col_now.metric(
            "Pris just nu (idag)",
            f"{current_price:,.0f}",
            help="Dagens Bitcoin-pris ersätter tillfälligt senaste veckopriset när prognosens features "
                 "beräknas. Databasen, modellträningen och RMSE-testet ändras inte.",
        )
    else:
        col_now.caption("Kunde inte hämta dagens pris just nu.")
    if prediction_is_valid:
        col2.metric(f"Prognos om {horizon_label}", f"{next_price:,.0f}", f"{change_pct:+.1f}%")
        if next_price == 0:
            st.warning(
                f"{method} gav en rå prisprognos på 0 USD eller lägre för {horizon_label}. "
                "Prognosen visas därför med det ekonomiska golvet 0 USD. Ett procentbaserat "
                "prisintervall kan inte beräknas runt 0."
            )
    else:
        col2.metric(f"Prognos om {horizon_label}", "Ej giltig")
        st.warning(
            f"{method} gav en icke ändlig prisprognos för {horizon_label}. "
            "Prognosen visas därför inte. Välj en annan metod eller horisont."
        )
    show_historical_margin(next_price, metrics)
    show_evaluation(metrics)
    with st.popover("ℹ️ Information om prognosen"):
        if is_positive_finite(current_price):
            st.write(
                f"Prognosen gäller {forecast_date:%Y-%m-%d}. För just denna prognos ersätts priset i den "
                f"senaste veckoraden ({last_row['date']:%Y-%m-%d}) tillfälligt med dagens pris "
                f"{current_price:,.0f} USD, så prisbaserade features räknas om. Databasen och modellens "
                "träning/test behåller det riktiga veckopriset."
            )
        else:
            st.write(
                f"Dagens pris kunde inte användas, så prognosen utgår från senaste Bitcoin-observationen "
                f"{last_row['date']:%Y-%m-%d} och gäller {forecast_date:%Y-%m-%d}."
            )

    st.subheader(f"Prishistorik – {method} ({horizon_label} framåt)")

    fig = forecast_chart(df, next_price, forecast_origin_date, forecast_date, forecast_base_price)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Senaste veckorna")
    st.dataframe(
        df.sort_values("date", ascending=False).head(10).set_index("date"),
        width="stretch",
    )

    with st.expander("Om modellen"):
        relative_rmse_text = format_optional_pct(metrics.get("relative_price_rmse_pct"))
        margin_text = format_optional_pct(metrics.get("historical_margin_pct"), prefix="±")
        if metrics["tuned"]:
            methodology = f"""
            **{method}** tränad direkt mot horisonten **{horizon_label}** ("vad blir priset
            om {weeks_ahead} veckor?") med en kronologisk **tränings-/validerings-/test**-
            uppdelning: {metrics['n_train']} tränings-, {metrics['n_val']} validerings- och
            {metrics['n_test']} testexempel (äldst → nyast).

            1. Ett par hyperparameter-kandidater tränas på träningsdelen och jämförs på
               valideringsdelen – bästa valet: `{metrics['best_params'] or "standardvärden"}`
               (val-RMSE: {metrics['val_rmse']:.1f}).
            2. Den valda konfigurationen tränas om på träning+validering inom vald historiklängd och testas en
               **enda gång** på den helt osedda testdelen – det är detta som rapporteras
               nedan som modellens riktiga prestanda.
            3. Den modell som faktiskt gör prognosen ovan tränas därefter om en sista gång
               på kompletta exempel inom **vald historiklängd**. Testutfallen ingår först
               efter utvärderingen.
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
            - Relativ RMSE på intervallets kalibreringsdel: {relative_rmse_text}
            - Kalibrerad 80 %-marginal: {margin_text}
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

        st.write("**Jämförelse mellan metoder (relativ RMSE, lägre är bättre):**")
        comparison = pd.DataFrame(
            [{
                "Metod": m,
                "Relativ RMSE (%)": all_metrics[m].get("relative_price_rmse_pct"),
                "RMSE (USD)": all_metrics[m]["rmse"],
                "MAE (USD)": all_metrics[m]["mae"],
            } for m in METHODS]
        ).sort_values("Relativ RMSE (%)", na_position="last")
        st.dataframe(comparison.set_index("Metod"), width="stretch")

    st.subheader(f"Bitcoin + makro – {method} ({horizon_label} framåt)")
    st.caption("Separat Bitcoin-prognos: samtliga Bitcoin-features + S&P 500, guld och amerikansk tioårsränta.")
    if st.button("Uppdatera makrodata"):
        get_macro_data.clear()
        get_macro_model_and_metrics.clear()

    try:
        with st.spinner("Hämtar makrodata och tränar Bitcoin + makro…"):
            macro_data = get_macro_data((df["date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d"))
            all_macro_results = {
                name: get_macro_model_and_metrics(name, weeks_ahead, df, macro_data)
                for name in METHODS
            }
            macro_model, macro_metrics = all_macro_results[method]
            macro_price = predict_price(inference_df, macro_model, horizon_weeks=weeks_ahead, macro_df=macro_data)
        all_macro_metrics = {name: result[1] for name, result in all_macro_results.items()}
        show_best_method(all_macro_metrics, method, "Bitcoin + makro")
        macro_prediction_is_valid = is_nonnegative_finite(macro_price)
        if macro_prediction_is_valid:
            macro_change_pct = (macro_price / forecast_base_price - 1) * 100
            st.metric(f"Makroprognos om {horizon_label}", f"{macro_price:,.0f} USD", f"{macro_change_pct:+.1f}%")
            if macro_price == 0:
                st.warning(
                    f"Bitcoin + makro med {method} gav en rå prisprognos på 0 USD eller lägre för "
                    f"{horizon_label}. Prognosen visas därför med golvet 0 USD; något procentbaserat "
                    "prisintervall kan inte beräknas runt 0."
                )
        else:
            st.metric(f"Makroprognos om {horizon_label}", "Ej giltig")
            st.warning(
                f"Bitcoin + makro med {method} gav en icke ändlig prisprognos för {horizon_label}. "
                "Prognosen visas därför inte."
            )
        show_historical_margin(macro_price, macro_metrics)
        show_evaluation(macro_metrics)
        macro_fig = forecast_chart(
            df, macro_price, forecast_origin_date, forecast_date, forecast_base_price,
            price_label="Bitcoin-pris", forecast_label="Prognos: Bitcoin + makro",
        )
        st.plotly_chart(macro_fig, width="stretch")
        with st.popover("ℹ️ Information om makromodellen"):
            st.write(
                f"MAE: {macro_metrics['mae']:,.0f} USD. "
                f"Hyperparametertuning: {'ja' if macro_metrics['tuned'] else 'nej, för kort historik'}."
            )
            if macro_metrics["test_dates"] == metrics["test_dates"]:
                st.write("Bitcoin- och makromodellen utvärderas på samma testdatum.")
        if macro_metrics["test_dates"] != metrics["test_dates"]:
            st.warning("Modellerna har olika testdatum på grund av datatäckningen. Deras RMSE är inte direkt jämförbara.")
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


if __name__ == "__main__":
    main()
