# Bitcoin-predictor

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
körs. Nya sparade modeller använder prefixet `model_v2_` så att gamla modeller
med andra indatakolumner inte laddas. Test-RMSE och MAE jämförs med oförändrat
pris som referens; fler faktorer garanterar inte bättre prognoser. Jämför även
med modellen utan de nya faktorerna på samma testdatum för att mäta förbättring.
