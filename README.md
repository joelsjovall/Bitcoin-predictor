# Bitcoin-predictor

Volym hämtas och sparas i databasen för framtida experiment, men används för
närvarande inte som modellfeature. Ett kontrollerat test på samma data gav högre
Ridge-fel med volym (12 367 USD) än utan volym (10 532 USD). Volym ska aktiveras
igen först efter jämförelse på samma sluttestdatum.

## Modellval

Appen erbjuder linjär regression, Ridge, SVR, Random Forest och XGBoost.
Ridge använder standardiserade faktorer och L2-regularisering. Styrkan `alpha`
väljs bland 0.1, 1, 10, 100, 300, 1000 och 3000 via validerings-RMSE, inte sluttestet.
För varje metod väljs samtidigt hela historiken eller ett fönster på upp till 208/416 veckor
(fyra/åtta år). Alla kandidater bedöms på samma valideringsdatum. Faktorerna
beräknas före avgränsningen. Fönstret räknas bakåt från senaste tillgängliga
träningsexemplets prognosdatum, inte från dagens datum; dess utfall måste vara känt.
Vald längd används även vid sluttest och omträning för den aktuella prognosen.
Vid lika resultat behålls första alternativet, hela historiken.
Appen visar vald historiklängd för alla modeller.
Ridge fungerar med både Bitcoin-faktorer och makrodata.
Om historiken inte räcker för validering används hela historiken och `alpha=1`.
Modellfilerna använder versionsprefixet `history_v2` för att inte läsa äldre
modeller som tränats med annan historiklängd.
Jämför testresultaten med oförändrat pris; Ridge garanterar inte lägre fel.
Starta om Streamlit efter att det nya modellvalet lagts till.

Appens markering av **bästa metod** baseras på lägst relativ RMSE i
kalibreringsdelen, separat för Bitcoin och Bitcoin + makro. Metoder där relativ
RMSE inte kan beräknas på grund av en historisk nollprognos deltar inte i den
jämförelsen. RMSE i USD och MAE visas fortfarande som kompletterande mått.

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
Makrografen visar RMSE i USD. Fler faktorer garanterar
inte bättre prognoser. Historiska leverantörsdata är inte en point-in-time-databas
och kan ha reviderats. Bitcoin-datum måste motsvara när priset faktiskt var känt.

Källor: [yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html),
[S&P 500](https://finance.yahoo.com/quote/%5EGSPC/),
[guldterminer](https://finance.yahoo.com/quote/GC%3DF/),
[tioårsränta](https://finance.yahoo.com/quote/%5ETNX/).

## Bitcoin-features

Båda prognoserna visar samma RMSE-mått: test-RMSE i USD.
Appen skiljer mellan hela Bitcoin-historiken,
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
`senaste_pris * (1 + prognostiserad_avkastning)`, med 0 USD som lägsta möjliga
pris. Samma golv används vid validering, sluttest och aktuell prognos. Om råvärdet
är negativt visas därför 0 USD och -100 procent; ett procentbaserat prisintervall
kan då inte beräknas eftersom prognospriset ligger i nämnaren.

För den aktuella prognosen skapar appen en tillfällig kopia av veckodatan och
ersätter priset i den senaste veckoraden med det senast hämtade dagspriset.
Prisbaserade features räknas därmed om med det aktuella priset för både Bitcoin
och Bitcoin + makro. Kopian används bara vid inferens: databasen, träningen och
sluttestets mått behåller de ursprungliga veckopriserna. Om dagspriset inte kan
hämtas används senaste lagrade veckopris som tidigare.

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

### Sluttest och felintervall

Appen visar RMSE i USD och en 80-procentsfelmarginal från modellens kronologiska
sluttest. När det finns minst tio
prognoser används de äldsta 70 procenten för att kalibrera marginalen och de
senare 30 procenten för en separat täckningskontroll. Felmarginalen är 80:e
percentilen (avrundad uppåt till ett observerat fel) av
`abs(utfall - prognos) / prognos`. Relativ RMSE beräknas på samma
kalibreringsprognoser och med samma procentbas, men kvadrerar felen och påverkas
därför mer av stora missar. Det beskriver modellens samlade procentuella
felstorlek, medan 80-procentsfelmarginalen beskriver den historiska gräns som
behövdes för att omfatta minst 80 procent av kalibreringsfelen. Måtten behöver
inte vara lika och det finns ingen fast omräkning mellan dem. Prisintervallet
använder felmarginalen; den senare kontrolltäckningen visar hur intervallet
fungerade på utfall som inte användes för kalibreringen. Vid färre än tio
prognoser används hela underlaget
för marginalen och ingen separat täckning visas. Intervallet är en historiskt
kalibrerad osäkerhetsindikator, inte en formell sannolikhet för nästa prognos.
