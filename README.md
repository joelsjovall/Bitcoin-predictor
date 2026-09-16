# Bitcoin-predictor

## Två prognosmodeller

Graf 1 behåller Bitcoin-modellen och dess `FEATURE_COLUMNS`. Graf 2 använder
samma Bitcoin-features plus sex makrofaktorer från `macro.py`:

- `sp500_return_12w`, `sp500_return_52w`: S&P 500-prisavkastning.
- `gold_return_12w`, `gold_return_52w`: guldterminernas prisavkastning.
- `treasury_10y_change_12w`: tioårsräntans förändring i procentenheter.
- `treasury_10y_level`: tioårsräntan i procent.

Avkastning lagras i decimalform (0,05 = 5 %) och visas i procent i faktortabellen.
Ränteförändring är inte obligationsavkastning. Båda modellerna förutspår Bitcoins
framtida avkastning; graferna visar motsvarande Bitcoin-pris i USD, med prognosens
procentuella förändring ovanför. Metod- och horisontvalet styr båda graferna.

Dagliga stängningar hämtas via yfinance: `^GSPC`, `GC=F` (guldterminer, inte spot)
och `^TNX` (räntenivå i procent). S&P-serien inkluderar inte återinvesterade
utdelningar och guldserien kan påverkas av kontraktsbyten. För varje Bitcoin-datum
används endast strikt tidigare observationer, högst sju dagar gamla. Inga framtida
värden eller bakåtfyllning används. Tolv och 52 veckor räknas på den obrutna
Bitcoin-veckokalendern innan rader med saknade features tas bort.

Makrodata cachas en timme i appen och kan hämtas om med **Uppdatera makrodata**.
Makromodellen har separat cache och filprefix `model_v4_macro_v1_`; Bitcoin-modeller
skrivs inte över. Saknad historik eller nedladdningsfel visas vid graf 2 utan att
stoppa graf 1. Även femårsmodellen kräver tillräcklig överlappande historik.
Testdatum visas och appen varnar om modellernas RMSE gäller olika testdatum.
Makrografen visar även RMSE för avkastning i procentenheter. Fler faktorer garanterar
inte bättre prognoser. Historiska leverantörsdata är inte en point-in-time-databas
och kan ha reviderats. Bitcoin-datum måste motsvara när priset faktiskt var känt.

Källor: [yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
[S&P 500](https://finance.yahoo.com/quote/%5EGSPC/),
[guldterminer](https://finance.yahoo.com/quote/GC%3DF/),
[tioårsränta](https://finance.yahoo.com/quote/%5ETNX/).

## Bitcoin-features

Båda prognoserna visar samma RMSE-mått: test-RMSE i USD och test-RMSE för
avkastning i procentenheter. Appen skiljer mellan hela Bitcoin-historiken,
slutmodellens kompletta träningsexempel, testets prognosdatum och deras utfallsdatum.
En femårsprognos kan bara tränas på startdatum vars utfall 260 veckor senare redan
är känt. Slutmodellen använder alla kompletta exempel, inklusive testdelen efter
utvärderingen; den aktuella prognosen använder senaste veckans features.
Historikens första 52 veckor behövs för att bygga features. Appen varnar om senaste
Bitcoin-priset är äldre än sju dagar; uppdatera då via **Hämta senaste data (live)**.

`weeks_to_halving` uppskattar återstående tid som
`max(208 - weeks_since_halving, 0)` veckor. Den använder endast redan inträffade
halveringar, inte framtida faktiska datum. Före första kända halveringen sätts
värdet till 0 (`has_halving = 0`). Efter 208 veckor stannar nedräkningen på 0
tills nästa registrerade halvering inträffar. Detta är en grov uppskattning,
inte en exakt prognos, och tillför ingen oberoende information utöver
`weeks_since_halving` och `has_halving`.

`pct_change` finns kvar i datan men används inte som feature eftersom den
dubblerar `return_1w`. Modeller sparas nu med prefixet `model_v4_`, så äldre
modeller laddas inte. Starta om appen efter kodändringen.

Modellen använder veckopriser och följande faktorer i `model.py`:

- Avkastning över 1, 4, 12 och 52 veckor.
- Volatilitet: standardavvikelse för veckoavkastning över 12 och 52 veckor.
- Avstånd till högsta observerade pris hittills i datasetet (`distance_from_ath`).
- Antal veckor sedan senast inträffade halvering och en indikator för om någon
  halvering har inträffat. Datumen finns i `HALVINGS`; uppdatera efter nya halveringar.
- Befintliga kortsiktiga avkastningsmått.

Alla faktorer beräknas i `build_features` och skickas till samtliga modeller via
`FEATURE_COLUMNS`. Modellerna lär sig sambanden vid omträning. Halveringar får
ingen manuellt bestämd positiv eller negativ effekt. Övriga nya mått är fortfarande
omvandlingar av historiska priser, inte nya externa datakällor.

Målet är framtida avkastning: `price.shift(-horizon_weeks) / price - 1`.
1, 2 och 3 år motsvarar 52, 104 och 156 veckor. Varje horisont tränas separat;
framtida avkastning används aldrig som indata. Prognospriset beräknas som
`senaste_pris * (1 + prognostiserad_avkastning)`.

Datan måste ha en observation var sjunde dag. Måtten förutsätter att radens pris
är känt vid radens datum; kontrollera detta särskilt för veckostaplar märkta med
veckans startdatum. Prognosen kräver minst 53 veckopriser. Träning kräver dessutom
historik för vald målhorisont, testperiod och ett mellanrum som hindrar träningsmål
från att överlappa testperioden. ATH avser datasetets historik, inte nödvändigtvis
Bitcoins verkliga rekord om tidig historik saknas.

Starta med `streamlit run app.py` och välj **Träna om modellen** om appen redan
körs. Nya sparade modeller använder prefixet `model_v4_` så att gamla modeller
med andra indatakolumner inte laddas. Test-RMSE och MAE jämförs med oförändrat
pris som referens; fler faktorer garanterar inte bättre prognoser. Jämför även
med modellen utan de nya faktorerna på samma testdatum för att mäta förbättring.
