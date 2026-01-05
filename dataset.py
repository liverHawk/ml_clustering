import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    return Path, pl


@app.cell
def _():
    DATASET = "CICIDS2017_improved"
    return (DATASET,)


@app.cell
def _(DATASET, Path, pl):
    path = Path(f"/Users/toshi_pro/Documents/school/dataset/project/cleaned/{DATASET}")

    files = list(path.glob("*.csv"))

    for file in files:
        df = pl.read_csv(file)
    
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
