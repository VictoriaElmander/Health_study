import pandas as pd
import re

# -------------------------------------------------------
# 0. Ladda data + krav på kolumner
# -------------------------------------------------------

REQUIRED = ["id", "age", "sex", "height", "weight", "systolic_bp", "cholesterol", "smoker", "disease"]

def load_data(file_path: str) -> pd.DataFrame:  
    """
    Reads data from a CSV file and creates a DataFrame.
    """
    
    df = pd.read_csv(file_path, encoding="utf-8") 
    missing = [col for col in REQUIRED if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}") #Check for required columns
    return df


# -------------------------------------------------------
# 1. DataCleaner-klass
# -------------------------------------------------------

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()  # behåll originaldata orörd
        self.pattern = r"^\s+|\s+$" #r"^\s|\s$"


    # 1. Rensa whitespace + konvertera till kategori
    def clean_and_categorize(self, columns):
        # förväntade kategoriska värden
        expected = {
        "sex": {"M", "F"},
        "smoker": {"Yes", "No"}
    }
        
        for col in columns:
            # Kolla whitespace
            if self.df[col].astype(str).str.contains(self.pattern, regex=True, na=False).any():
                print(f"{col}: whitespace detected - cleaning")
                self.df[col] = self.df[col].astype(str).str.strip()
            else:
                print(f"{col}: clean")
            
            # Validera värden om kolumnen finns i expected
            if col in expected:
                unique_vals = set(self.df[col].dropna().unique())
                if not unique_vals <= expected[col]:
                    print(f"VARNING: '{col}' innehåller oväntade värden: {unique_vals - expected[col]}")

            # Konvertera till kategori
            self.df[col] = self.df[col].astype("category")

        return self


    # 2. Konvertera disease till bool
    def convert_disease_to_bool(self, col="disease"):
        unique_vals = set(self.df[col].dropna().unique())

        try:
            unique_vals = {int(v) for v in unique_vals}
        except ValueError:
            # Om något värde inte går att konvertera till int hoppa över konverteringen
            # och skriver ut vilka värden som orsakar problem
            print(f"VARNING: '{col}' innehåller icke-numeriska värden: {unique_vals}")
            return self

        if unique_vals <= {0, 1}:
            self.df[col] = self.df[col].astype(int).astype(bool)
            print(f"Kolumnen '{col}' innehåller endast 0 och 1 - konverterad till bool.")
        else:
            print(f"Kolumnen '{col}' innehåller andra värden ({unique_vals}) - ingen konvertering gjord.")

        return self


    # 3. Ta bort dubbletter baserat på id
    def remove_duplicate_ids(self, id_col="id"):
        unique_ids = self.df[id_col].nunique()
        total_rows = len(self.df)

        if unique_ids < total_rows:
            duplicates = total_rows - unique_ids
            print(f"Dubbletter hittades baserat på '{id_col}': {duplicates} st")
            self.df = self.df.drop_duplicates(subset=id_col, keep="first") #behåller det första id vid dubletter
            print(f"Dubbletter borttagna. Ny datastorlek: {len(self.df)} rader")
        else:
            print(f"Alla ID i '{id_col}' är unika – inga dubbletter hittades.")

        return self

    # 4. Utskrift av kolumninformation
    def show_info(self):
        print("\ncolumn".ljust(12), "dtype".ljust(12), "unique values")
        print("-" * 80)

        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            uniques = self.df[col].dropna().unique()

            # Om kolumnen är kategorisk eller bool: visa alla unika värden
            if str(self.df[col].dtype) in ["category", "bool", "object"]:
                values_str = str(uniques.tolist())
            else:
                # Numeriska kolumner: visa bara antal unika + min/max
                n_unique = len(uniques)
                col_min = self.df[col].min()
                col_max = self.df[col].max()
                values_str = f"{n_unique} uniques, min={col_min}, max={col_max}"

            print(col.ljust(12), dtype.ljust(12), values_str)

        print("-" * 80)


    # 5. Kör hela rensningspipen
    def process(self):
        self.clean_and_categorize(["sex", "smoker"]) \
            .convert_disease_to_bool("disease") \
            .remove_duplicate_ids("id")

        # Automatisk utskrift
        self.show_info()

        return self.df

