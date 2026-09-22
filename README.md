# Bitcoin-predictor

Utforska Bitcoins prishistorik och jämför hur olika maskininlärningsmodeller bedömer framtida priser — direkt i webbläsaren.

Appen visar två perspektiv: en prognos baserad på Bitcoins historik och en som även tar hänsyn till S&P 500, guld och amerikansk tioårsränta. Grafer och historiska tester hjälper dig att jämföra resultaten.

## Vad kan du göra?

- Följa Bitcoins prishistorik och hämta uppdaterade priser.
- Jämföra fem modeller: Linjär regression, Ridge, SVR, Random Forest och XGBoost.
- Välja prognoser från en månad till fem år.
- Se hur modellerna har presterat historiskt, jämfört med att priset hade varit oförändrat.

Byggt med **Python, Streamlit, Plotly och SQLite**. All träning och lagring sker lokalt. Pris- och makrodata hämtas via Yahoo Finance, och Bitcoin-historik följer med som CSV.

## Kom igång

Du behöver **Git, Python 3.11 och en webbläsare**. Internet behövs för installation och hämtning av nya data. Ingen API-nyckel eller separat databasserver krävs.

### Windows · PowerShell

Öppna PowerShell och kör:

```powershell
git clone https://github.com/joelsjovall/Bitcoin-predictor.git
cd Bitcoin-predictor
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe db.py
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

### macOS / Linux

Öppna terminalen och kör:

```bash
git clone https://github.com/joelsjovall/Bitcoin-predictor.git
cd Bitcoin-predictor
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python db.py
python -m streamlit run app.py --server.address 127.0.0.1
```

Öppna **http://127.0.0.1:8501** om webbläsaren inte öppnas automatiskt. Första körningen kan ta lite tid medan modellerna tränas. Stoppa appen med `Ctrl+C` i terminalen.

### Starta igen senare

Öppna terminalen i projektmappen och kör:

**Windows:**

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

**macOS / Linux:**

```bash
source .venv/bin/activate
python -m streamlit run app.py --server.address 127.0.0.1
```

Du behöver bara installera beroenden och importera CSV-filen första gången. **En ny körning av `db.py` ersätter sparade priser med CSV-innehållet.** Använd appens liveuppdatering för att hämta nya priser.

## Så använder du appen

1. Välj modell och hur långt fram prognosen ska sträcka sig.
2. Jämför Bitcoin-prognosen med prognosen som även använder makrodata.
3. Titta på historiska testresultat och felmarginaler tillsammans med prognosen.

Med **Hämta senaste data (live)** uppdaterar du Bitcoin-priserna. **Uppdatera makrodata** hämtar nya makrodata, och **Träna om modellen** räknar om resultaten.

Prognosen utgår från det senaste veckopriset. Det dagliga pris som visas separat kan därför skilja sig från prognosens startpris.

## Bra att veta

- **Långsam första start?** Träning och historiska tester tar tid. Börja gärna med Linjär regression och en kort horisont.
- **Gamla resultat?** Uppdatera datan i appen och träna om. En omstart tömmer appens minnescache; sparad historik finns kvar.
- **Ingen makroprognos?** Kontrollera anslutningen och försök uppdatera makrodata igen. Bitcoin-prognosen kan fortfarande fungera.
- **För lite historik?** Välj en kortare horisont. Långa prognoser kräver mer historik för att kunna utvärderas.
- **Vill du säkerhetskopiera?** Stoppa appen och kopiera `bitcoin.db` till en säker plats.

Appen är avsedd för lokal användning och saknar inloggning. Prognoser och historiska felmarginaler är experimentella och garanterar inte framtida resultat.

## För dig som vill utveckla vidare

Projektet är uppdelat i fyra delar: `app.py` för gränssnittet, `db.py` för Bitcoin-data, `macro.py` för makrodata och `model.py` för träning och prognoser.

Kör testerna från projektmappen:

```powershell
# Windows
.\.venv\Scripts\python.exe -m pytest -q
```

```bash
# macOS / Linux, med den virtuella miljön aktiverad
python -m pytest -q
```
