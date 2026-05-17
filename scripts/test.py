
import sys
from pathlib import Path
import pandas as pd


ROOT = Path("").resolve()
sys.path.insert(0, str(ROOT))
DEBUG_PARQUET = ROOT / "data/debug_scanpaths.parquet"
df = pd.read_parquet(DEBUG_PARQUET)
locs = df.iloc[0]["locations"].reshape(-1, 2)
print(locs.min(), locs.max())
print(locs[:5])  # erste paar Fixationen anschauen

