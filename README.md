# Health_study

Python 3.11.9

## Skapa och åteskapa miljön
Reproducera miljön

Projektet använder en virtuell miljö (venv).
För att återskapa miljön kör:

python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

requirements.txt = komplett miljö
requirements_minimal.txt = endast nödvändiga paket

## Sammanfattning av analysen
Dataset: 800 patienter
Kategoriska variabler: kön, rökstatus, sjukdom
Numeriska variabler: ålder, längd, vikt, kolesterol, systoliskt blodtryck

## Viktiga fynd
-   Systoliskt blodtryck korrelerar starkast med:
        -   Ålder (r≈0.61)
        -   Kolesterol (r≈0.37)

-   Konfidensintervall (norm/bootstrap/BCa) gav samma resultat → robust skattning

-   Hypotes "rökare har högre blodtryck" var inte signifikant, men power var låg (~20%)

-   Regression: ålder & vikt är bästa prediktorer för blodtryck

-   PCA: variation kopplad till blodtryck förklaras främst av ålder & kolesterol (71%)


