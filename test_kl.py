import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import numpy as np

    from pathlib import Path
    from typing import get_args

    from dataset.cicids2017 import relabeled_dataset
    from lib.data import get_schema, encode_categorical, EncodeMethod, NormalizeMethod, normalize

    DATASET = "CICIDS2017_improved"
    return (
        DATASET,
        EncodeMethod,
        NormalizeMethod,
        Path,
        encode_categorical,
        get_args,
        get_schema,
        mo,
        normalize,
        np,
        pl,
        relabeled_dataset,
    )


@app.cell
def _(DATASET, Path, get_schema, mo, pl):
    path = Path(f"/home/hawk/Documents/school/dataset/project/cleaned/{DATASET}")
    files = list(path.glob("*.csv"))
    schema = get_schema(files)

    dfs = []
    for file in mo.status.progress_bar(files, title="Loading CSV files", show_eta=True, show_rate=True):
        df = pl.scan_csv(file, schema_overrides=schema)
        dfs.append(df)

    df = pl.concat(dfs).collect()
    return (df,)


@app.cell
def _(
    EncodeMethod,
    NormalizeMethod,
    df,
    encode_categorical,
    get_args,
    mo,
    normalize,
):
    delete_columns = ["Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP", "SimillarHTTP"]
    category_columns = ["Source Port", "Destination Port", "Protocol", "Src Port", "Dst Port"]
    delete_columns = [col for col in delete_columns if col in df.columns]
    category_columns = [col for col in category_columns if col in df.columns]

    _df_delete_columns = df.drop(delete_columns)

    _encode = get_args(EncodeMethod)
    _normalize = get_args(NormalizeMethod)

    encode_choice = _encode[1]
    normalize_choice = _normalize[1]

    _df = encode_categorical(
        df=_df_delete_columns,
        method=encode_choice,
        columns=category_columns
    )
    _df = normalize(
        df=_df,
        method=normalize_choice,
        except_original_columns=category_columns
    )
    df_prepared = _df
    mo.md(f"encode: {encode_choice}, normalize: {normalize_choice}")
    return (df_prepared,)


@app.cell
def _(df_prepared, relabeled_dataset):
    df_relabeled = relabeled_dataset(df=df_prepared)

    n_labels = df_relabeled["Label"].n_unique()
    unique_labels = df_relabeled["Label"].unique().to_list()
    # print(f"Number of unique labels: {n_labels}, Labels: {unique_labels}")
    return df_relabeled, n_labels, unique_labels


@app.cell
def _():
    # mo.ui.dataframe(df_relabeled.head())
    # df_relabeled.columns
    return


@app.cell
def _(df_relabeled, pl):
    known_labels = ["Portscan", "FTP-Patator", "Web Attack"]


    _df_unknown = df_relabeled.filter(~pl.col("Label").is_in(known_labels))
    return (known_labels,)


@app.cell
def _(df_relabeled, known_labels, pl):
    _df_known = df_relabeled.filter(pl.col("Label").is_in(known_labels))

    _df_known = _df_known.with_row_index()

    # add column (values all -1)
    _df_known = _df_known.with_columns(
        pl.lit(-1).alias("selected_index")
    )
    group_mapping = (
        _df_known.select("Label")
        .unique()
        .with_row_index("group_id")
        .with_columns(pl.col("group_id") + 1)
    )

    _df_known = _df_known.join(
        group_mapping,
        on="Label",
        how="left"
    )

    # select 2 samples from each known label and "selected_index" -> cluster index
    _df_known = _df_known.with_columns(
        (pl.int_range(pl.len()).over("Label") < 2).alias("is_selected")
    )
    _df_known = _df_known.with_columns(
        pl.when(pl.col("is_selected"))
        .then(pl.col("group_id"))
        .otherwise(pl.col("selected_index"))
        .alias("selected_index")
    )
    _df_known = _df_known.drop(["is_selected"])
    df_initial_selected = _df_known

    # mo.ui.dataframe(_df_known.group_by("Label").head(3))
    return (df_initial_selected,)


@app.cell
def _(df_initial_selected):
    _X_known = df_initial_selected.drop(["Label", "index", "group_id"])
    # show negative values in _X_known
    # _X_known.filter(pl.any_horizontal(pl.col(pl.Float64).lt(0)))
    return


@app.cell
def _():
    from clustring_methods.configs import create_frobenius_config
    from clustring_methods.ssnmf_refactored import SSNMFFrobenius
    return SSNMFFrobenius, create_frobenius_config


@app.cell
def _(
    SSNMFFrobenius,
    create_frobenius_config,
    df_initial_selected,
    known_labels,
    np,
    pl,
):
    _X_known = df_initial_selected.drop(["Label", "index", "group_id"])
    _y_known = df_initial_selected.select(["Label", "index", "selected_index", "group_id"])

    _labeled_count = _X_known.filter(pl.col("selected_index") != -1).height
    # print(f"Number of initially labeled samples: {_labeled_count}")

    labeled_indices = _y_known.filter(pl.col("selected_index") != -1)["index"].to_list()

    W_init = []
    for index in labeled_indices:
        id = _y_known.filter(pl.col("index") == index)["group_id"][0]
        row = np.zeros(len(known_labels))
        row[id - 1] = 1.0
        W_init.append(row)
    W_init = np.array(W_init)


    config = create_frobenius_config(
        n_components=len(known_labels),
        alpha=0.5,
        max_iter=100,
        verbose=True,
        random_state=42
    )
    _X_known = np.maximum(_X_known.to_numpy(), 0)
    model = SSNMFFrobenius(config)
    model.fit(X=_X_known, labels=W_init, labeled_indices=labeled_indices)
    return (model,)


@app.cell
def _(df_initial_selected, model, np, pl):
    model.W
    W_normalized = model.W / (model.W.sum(axis=1, keepdims=True) + 1e-10)
    confidence = W_normalized.max(axis=1)
    _y_known = df_initial_selected.select(["Label", "index", "selected_index", "group_id"])
    high_confidence_mask = (confidence > 0.8) & (_y_known["selected_index"].to_numpy() == -1)

    selected = np.where(high_confidence_mask)[0]
    selected

    df_known_selected = df_initial_selected
    df_known_selected = df_known_selected.with_columns(
        pl.when(pl.int_range(pl.len()).is_in(selected))
        .then(pl.col("group_id"))
        .otherwise(pl.col("selected_index"))
        .alias("selected_index")
    )
    df_known_selected = df_known_selected.drop(["group_id", "index"])

    return (df_known_selected,)


@app.cell
def _(df_known_selected, df_relabeled, known_labels, pl):
    _df_unknown = df_relabeled.filter(~pl.col("Label").is_in(known_labels))
    _df_unknown = _df_unknown.with_columns(
        pl.lit(-1).cast(pl.Int64).alias("selected_index")
    )

    schema_final = df_known_selected.schema
    unknown_schema = _df_unknown.schema

    # print("Final known schema:", schema_final)
    # print("Final unknown schema:", unknown_schema)

    df_final = pl.concat([
        df_known_selected,
        _df_unknown
    ],)
    return (df_final,)


@app.cell
def _(SSNMFFrobenius, create_frobenius_config, df_final, np):
    final_config = create_frobenius_config(
        n_components=9,
        alpha=0.5,
        max_iter=200,
        verbose=True,
        random_state=42
    )
    X_final = np.maximum(df_final.drop("Label").to_numpy(), 0)

    ssnmf_final = SSNMFFrobenius(final_config)
    model_final = ssnmf_final.fit(X=X_final)
    return (model_final,)


@app.cell
def _():
    # len(model_final.W.argmax(axis=1)), len(df_final)
    return


@app.cell
def _(df_final, model_final, pl):
    result_df = df_final.with_columns(
        pl.Series(model_final.W.argmax(axis=1)).alias("Predicted Cluster")
    )
    # mo.ui.dataframe(result_df.head(50))
    return (result_df,)


@app.cell
def _(result_df):
    result_list = result_df.select(["Label", "Predicted Cluster"]).to_dicts()
    from collections import Counter
    counter = Counter()
    for item in result_list:
        key = (item["Label"], item["Predicted Cluster"])
        counter[key] += 1
    # for key, count in counter.items():
    #     print(f"{key}: {count}")
    return (result_list,)


@app.cell
def _(n_labels, np, result_list, unique_labels):
    confusion_matrix = np.zeros((n_labels, 9), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    for _item in result_list:
        label_idx = label_to_index[_item["Label"]]
        cluster_idx = _item["Predicted Cluster"]
        confusion_matrix[label_idx, cluster_idx] += 1

    confusion_matrix
    return (confusion_matrix,)


@app.cell
def _(confusion_matrix, pl, unique_labels):
    cf_df = pl.DataFrame(
        confusion_matrix,
        schema=[f"Cluster {i}" for i in range(9)],
    ).with_columns(
        pl.Series("True Label", unique_labels)
    )

    # mo.ui.dataframe(cf_df)
    return (cf_df,)


@app.cell
def _(cf_df, mo):
    import altair as alt

    # ロング形式（長形式）に変換 - ヒートマップ用
    cf_long = cf_df.unpivot(
        index="True Label",  # 固定する列
        variable_name="Predicted Label",  # 列名が入る新しい列
        value_name="Count"  # 値が入る新しい列
    )
    # Altairでヒートマップを作成
    base = alt.Chart(cf_long.to_pandas()).encode(
        x=alt.X("Predicted Label:N", title="予測ラベル"),
        y=alt.Y("True Label:N", title="真のラベル")
    )
    # 色付きセル
    heatmap = base.mark_rect().encode(
        color=alt.Color(
            "Count:Q",
            scale=alt.Scale(scheme="blues"),
            title="件数"
        ),
        tooltip=[
            alt.Tooltip("True Label:N", title="真のラベル"),
            alt.Tooltip("Predicted Label:N", title="予測ラベル"),
            alt.Tooltip("Count:Q", title="件数")
        ]
    )
    # 数値ラベル
    text = base.mark_text(baseline="middle", fontSize=11).encode(
        text=alt.Text("Count:Q", format=".0f"),
        color=alt.condition(
            alt.datum.Count > 100,  # 閾値を調整
            alt.value("white"),
            alt.value("black")
        )
    )
    # レイヤーを重ねる
    chart = (heatmap + text).properties(
        width=500,
        height=500,
        title="混同行列ヒートマップ"
    )
    mo.ui.altair_chart(chart)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
