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

    DATASET = "CICIDS2017_flow_improved"
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
def _(EncodeMethod, NormalizeMethod, get_args, mo):
    _encode = get_args(EncodeMethod)
    _normalize = get_args(NormalizeMethod)

    encode_select = mo.ui.dropdown(label="Encoding Method", options=list(_encode), value=_encode[1])
    normalize_select = mo.ui.dropdown(label="Normalization Method", options=list(_normalize), value=_normalize[1])

    mo.hstack([encode_select, normalize_select])
    return encode_select, normalize_select


@app.cell
def _(df, encode_categorical, encode_select, normalize, normalize_select):
    delete_columns = ["Src IP", "Dst IP", "Timestamp", "Source IP", "Destination IP", "SimillarHTTP"]
    category_columns = ["Source Port", "Destination Port", "Protocol", "Src Port", "Dst Port"]
    delete_columns = [col for col in delete_columns if col in df.columns]
    category_columns = [col for col in category_columns if col in df.columns]

    _df_delete_columns = df.drop(delete_columns)

    _df = encode_categorical(
        df=_df_delete_columns,
        method=encode_select.value,
        columns=category_columns
    )
    _df = normalize(
        df=_df,
        method=normalize_select.value,
        except_original_columns=category_columns
    )
    df_prepared = _df
    return (df_prepared,)


@app.cell
def _(df_prepared, relabeled_dataset):
    df_relabeled_before = relabeled_dataset(df=df_prepared)

    n_labels = df_relabeled_before["Label"].n_unique()
    unique_labels = df_relabeled_before["Label"].unique().to_list()
    # print(f"Number of unique labels: {n_labels}, Labels: {unique_labels}")
    return df_relabeled_before, unique_labels


@app.cell
def _(mo, unique_labels):
    limit_options = unique_labels
    limit_select = mo.ui.multiselect(options=limit_options)
    limit_select
    return (limit_select,)


@app.cell
def _(df_relabeled_before, limit_select, mo, pl):
    df_relabeled = df_relabeled_before.filter(pl.col("Label").is_in(limit_select.value))
    options = df_relabeled["Label"].unique().to_list()
    known_select = mo.ui.multiselect(
        options=options,
    )
    mo.md(f"Select labels: {limit_select.value}")
    return df_relabeled, known_select


@app.cell
def _(known_select, mo):
    _text = mo.md(f"Selected known labels: {known_select.value}")
    number_field = mo.ui.number(label="Number of samples to show", value=5, start=1, stop=100)

    mo.hstack([
        known_select,
        _text,
        number_field
    ])
    return (number_field,)


@app.cell
def _(mo):
    set_known = mo.ui.run_button()
    set_known
    return (set_known,)


@app.cell
def _(df_relabeled, known_select, mo, pl, set_known):
    mo.stop(not set_known.value, "Process stopped by user.")
    known_labels = known_select.value


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
def _(mo):
    from clustring_methods.configs import create_frobenius_config
    from clustring_methods.ssnmf_refactored import SSNMFFrobenius

    alpha_input = mo.ui.number(label="Alpha parameter", value=0.5, start=0.0, stop=1.0, step=0.1)
    max_iter_input = mo.ui.number(label="Max iterations", value=200, start=10, stop=1000, step=10)

    mo.hstack([alpha_input, max_iter_input])
    return SSNMFFrobenius, alpha_input, create_frobenius_config, max_iter_input


@app.cell
def _(
    SSNMFFrobenius,
    alpha_input,
    create_frobenius_config,
    df_initial_selected,
    known_labels,
    max_iter_input,
    mo,
    np,
    number_field,
    pl,
):
    df_known_selected = df_initial_selected

    config = create_frobenius_config(
        n_components=len(known_labels),
        alpha=alpha_input.value,
        max_iter=max_iter_input.value,
        verbose=True,
        random_state=42
    )
    for _ in mo.status.progress_bar(range(number_field.value), title="SSNMF fitting", show_eta=False):
        # データの準備（更新された df_known_selected から取得）
        _X_known = df_known_selected.drop(["Label", "index", "group_id", "selected_index"])
        _y_known = df_known_selected.select(["Label", "index", "selected_index", "group_id"])

        # 非負値に変換
        X_known_np = np.maximum(_X_known.to_numpy(), 0)

        # ラベル付きサンプルの特定（行インデックスとして）
        _labeled_mask = _y_known["selected_index"] != -1
        labeled_row_indices = np.where(_labeled_mask.to_numpy())[0].tolist()

        # W_init の構築（更新されたラベル情報から）
        _W_init = []
        for row_idx in labeled_row_indices:
            group_id = _y_known["group_id"][row_idx]
            _row = np.zeros(len(known_labels))
            _row[group_id - 1] = 1.0
            _W_init.append(_row)
        _W_init = np.array(_W_init)

        print(f"selected_index counts: {_y_known['selected_index'].value_counts()}")
        print(f"Number of labeled samples: {len(labeled_row_indices)}")

        # モデルのフィッティング
        model = SSNMFFrobenius(config)
        model.fit(X=X_known_np, labels=_W_init, labeled_indices=labeled_row_indices)

        # 信頼度の計算
        W_normalized = model.W / (model.W.sum(axis=1, keepdims=True) + 1e-10)
        confidence = W_normalized.max(axis=1)

        # 予測クラスタの取得（0始まり）
        predicted_cluster = model.W.argmax(axis=1)

        # 正解ラベル（Label）を取得
        actual_labels = _y_known["Label"].to_numpy()
        selected_index_array = _y_known["selected_index"].to_numpy()

        # 既に選択されているサンプル（selected_index != -1）の情報を取得
        already_labeled_mask = selected_index_array != -1

        # 既に選択されているサンプルで、予測クラスタと正解ラベルが一致するものを特定
        # （予測クラスタに分類され、かつそのクラスタ内の既存サンプルの正解ラベルと一致）
        valid_labeled_mask = already_labeled_mask

        # 各クラスタごとに、既存の正解ラベルを記録
        cluster_to_labels = {}
        for _idx in np.where(valid_labeled_mask)[0]:
            cluster = predicted_cluster[_idx]
            _label = actual_labels[_idx]
            if cluster not in cluster_to_labels:
                cluster_to_labels[cluster] = set()
            cluster_to_labels[cluster].add(_label)

        # 未ラベルサンプルで、信頼度が高く、有効なクラスタに分類されるものを選択
        unlabeled = selected_index_array == -1
        high_confidence = confidence > 0.9

        # 各未ラベルサンプルについて、同じクラスタに分類された既存サンプルの正解ラベルと一致するか確認
        selected_mask = np.zeros(len(_y_known), dtype=bool)
        for _idx in np.where(unlabeled & high_confidence)[0]:
            cluster = predicted_cluster[_idx]
            _label = actual_labels[_idx]
            # 同じクラスタに既存サンプルが存在し、その正解ラベルと一致する場合
            if cluster in cluster_to_labels and _label in cluster_to_labels[cluster]:
                selected_mask[_idx] = True

        selected = np.where(selected_mask)[0]

        print(f"Clusters with valid labels: {len(cluster_to_labels)}")
        print(f"High confidence unlabeled: {high_confidence.sum()}, Selected: {len(selected)}")

        # selected_index の更新
        df_known_selected = df_known_selected.with_columns(
            pl.when(pl.int_range(pl.len()).is_in(selected))
            .then(pl.col("group_id"))
            .otherwise(pl.col("selected_index"))
            .alias("selected_index")
        )

    # ループ終了後の処理
    df_known_selected = df_known_selected.drop(["group_id", "index"])
    return (df_known_selected,)


@app.cell
def _(mo):
    button = mo.ui.run_button()
    button
    return


@app.cell
def _(df_known_selected, df_relabeled, known_labels, pl):
    # mo.stop(not button.value, "Process stopped by user.")

    _df_unknown = df_relabeled.filter(~pl.col("Label").is_in(known_labels))
    # _df_unknownにgroup_idカラムがない場合は追加（null値）
    if "group_id" not in _df_unknown.columns:
        _df_unknown = _df_unknown.with_columns(
            pl.lit(None).cast(pl.Int64).alias("group_id")
        )
    _df_unknown = _df_unknown.with_columns(
        pl.lit(-1).cast(pl.Int64).alias("selected_index")
    )

    # known_labelsのデータは選択データを1つ以上含む必要がある
    # それ以外のデータはランダムに10個取る

    # known_labelsのデータを処理
    df_known_sampled = df_known_selected.group_by("Label").head(10)

    # known_labelsの各ラベルで選択データが1つ以上含まれるようにする
    known_labels_with_selection = (
        df_known_sampled.filter(pl.col("selected_index") != -1)
        .select("Label")
        .unique()
    )

    # 選択されていないknown_labelsを特定
    all_known_labels = df_known_sampled.filter(pl.col("Label").is_in(known_labels)).select("Label").unique()
    known_labels_without_selection = all_known_labels.join(
        known_labels_with_selection,
        on="Label",
        how="anti"
    )

    # 選択されていないknown_labelsについて、最初の1つを選択する
    if len(known_labels_without_selection) > 0:
        df_known_sampled = df_known_sampled.with_columns(
            pl.when(
                pl.col("Label").is_in(known_labels_without_selection["Label"]) &
                (pl.int_range(pl.len()).over("Label") == 0)
            )
            .then(pl.col("group_id"))
            .otherwise(pl.col("selected_index"))
            .alias("selected_index")
        )

    # 未知ラベルのデータはランダムに10個取る
    unknown_labels = _df_unknown.select("Label").unique()
    unknown_samples = []
    for label_row in unknown_labels.iter_rows(named=True):
        _label = label_row["Label"]
        label_data = _df_unknown.filter(pl.col("Label") == _label)
        n_samples = min(10, len(label_data))
        sampled = label_data.sample(n_samples, seed=42)
        unknown_samples.append(sampled)

    if len(unknown_samples) > 0:
        df_unknown_sampled = pl.concat(unknown_samples)
    else:
        df_unknown_sampled = _df_unknown.filter(pl.lit(False))  # 空のDataFrame

    # 合体前に列を揃える
    # df_known_selectedは既にgroup_idとindexが削除されている
    # df_known_sampledからもgroup_idとindexを削除（存在する場合）
    if "group_id" in df_known_sampled.columns:
        df_known_sampled = df_known_sampled.drop(["group_id"])
    if "index" in df_known_sampled.columns:
        df_known_sampled = df_known_sampled.drop(["index"])

    # df_unknown_sampledからもgroup_idとindexを削除（存在する場合）
    if "group_id" in df_unknown_sampled.columns:
        df_unknown_sampled = df_unknown_sampled.drop(["group_id"])
    if "index" in df_unknown_sampled.columns:
        df_unknown_sampled = df_unknown_sampled.drop(["index"])

    # 共通の列のみを使用
    known_columns = set(df_known_sampled.columns)
    unknown_columns = set(df_unknown_sampled.columns)
    common_columns = sorted(list(known_columns & unknown_columns))

    # 両方のDataFrameを共通の列のみで選択
    df_known_sampled = df_known_sampled.select(common_columns)
    df_unknown_sampled = df_unknown_sampled.select(common_columns)
    return df_known_sampled, df_unknown_sampled


@app.cell
def _(df_known_sampled, df_unknown_sampled, pl):
    df_final = pl.concat([
        df_known_sampled,
        df_unknown_sampled
    ])
    return (df_final,)


@app.cell
def _(SSNMFFrobenius, create_frobenius_config, df_final, limit_select, np):
    final_config = create_frobenius_config(
        n_components=len(limit_select.value),
        alpha=0.5,
        max_iter=200,
        verbose=True,
        random_state=42
    )
    X_final = np.maximum(df_final.drop("Label").to_numpy(), 0)

    # 選択されたサンプル（selected_index != -1）かつlimit_select.valueに含まれるラベルの情報を取得
    # limit_select.valueのラベルとインデックスのマッピング
    _label_to_index = {label: idx for idx, label in enumerate(limit_select.value)}

    # ラベル付きサンプルを特定（selected_index != -1 かつ Labelがlimit_select.valueに含まれる）
    _labeled_mask = (
        (df_final["selected_index"] != -1) & 
        (df_final["Label"].is_in(limit_select.value))
    )
    labeled_indices = np.where(_labeled_mask.to_numpy())[0].tolist()

    # W_initの構築（選択されたサンプルのラベル情報から）
    if len(labeled_indices) > 0:
        W_init = []
        for idx in labeled_indices:
            label = df_final["Label"][idx]
            if label in _label_to_index:
                row = np.zeros(len(limit_select.value))
                row[_label_to_index[label]] = 1.0
                W_init.append(row)
            else:
                # 念のため（通常はここには来ない）
                rng = np.random.default_rng(42)
                row = rng.random(len(limit_select.value))
                row = row / row.sum()
                W_init.append(row)
        W_init = np.array(W_init)

        print(f"Final SSNMF (Semi-supervised): {len(labeled_indices)} labeled samples out of {len(X_final)}")

        # 半教師あり学習でフィッティング
        ssnmf_final = SSNMFFrobenius(final_config)
        model_final = ssnmf_final.fit(X=X_final, labels=W_init, labeled_indices=labeled_indices)
    else:
        # ラベルがない場合は警告を出して教師なし
        print("Warning: No labeled samples found. Using unsupervised NMF.")
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
def _(limit_select, np, result_list):
    confusion_matrix = np.zeros((len(limit_select.value), len(limit_select.value)), dtype=int)
    label_to_index = {label: idx for idx, label in enumerate(limit_select.value)}
    for _item in result_list:
        label_idx = label_to_index[_item["Label"]]
        cluster_idx = _item["Predicted Cluster"]
        confusion_matrix[label_idx, cluster_idx] += 1

    confusion_matrix
    return (confusion_matrix,)


@app.cell
def _(confusion_matrix, limit_select, pl):
    # 混同行列の作成: 行=真のラベル, 列=予測クラスタ
    # confusion_matrix[label_idx, cluster_idx] の構造を保持
    # label_idx は unique_labels の順序と一致
    cf_df = pl.DataFrame(
        confusion_matrix,
        schema=[f"Cluster {i}" for i in range(len(limit_select.value))],
    ).with_columns(
        pl.Series("True Label", limit_select.value)
    )

    # mo.ui.dataframe(cf_df)
    return (cf_df,)


@app.cell
def _(cf_df, known_labels, limit_select, mo):
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
    _known_labels = [label.replace(" ", "_") for label in known_labels]
    _known_labels = sorted(_known_labels)
    # レイヤーを重ねる
    chart = (heatmap + text).properties(
        width=500,
        height=500,
        title=f"混同行列ヒートマップ: {_known_labels}"
    )
    chart.save(f"plots/cm_{'_'.join(_known_labels)}.html")
    mo.vstack([mo.md(f"{limit_select.value}"), mo.ui.altair_chart(chart)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
