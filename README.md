# Health_study

Python 3.11.9

Reproducera miljön

Projektet använder en virtuell miljö (venv).
För att återskapa miljön kör:

python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt


requirements.txt innehåller en fullständig lista över alla paket som installerades i projektets venv, inklusive Jupyter-relaterade beroenden.
Filen gör det möjligt att reproducera exakt samma miljö som användes vid analysen.

För en mer minimalistisk installation finns även requirements_minimal.txt med endast de paket som används direkt i koden.
