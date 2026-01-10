import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import time
    import logging
    import coloredlogs
    import marimo as mo
    import polars as pl
    from typing import get_args
    import numpy as np
    return get_args, mo, np, time


@app.cell
def _():
    from lib.data import encode_categorical, EncodeMethod, NormalizeMethod, normalize, make_sample_data
    return (
        EncodeMethod,
        NormalizeMethod,
        encode_categorical,
        make_sample_data,
        normalize,
    )


@app.cell
def _(EncodeMethod, NormalizeMethod, get_args, make_sample_data, time):
    time_string = f'{time.strftime("%Y%m%d-%H%M%S")}'
    df_original, metadata = make_sample_data()

    encode_methods = get_args(EncodeMethod)
    normalize_methods = get_args(NormalizeMethod)

    df_copy = df_original.clone()
    return df_copy, encode_methods, normalize_methods, time_string


@app.cell
def _(encode_methods, mo):
    encode_method_ui = mo.ui.dropdown(
        options=list(encode_methods),
        value=encode_methods[0],
        label="Encode method",
    )
    return (encode_method_ui,)


@app.cell
def _(mo, normalize_methods):
    normalize_method_ui = mo.ui.dropdown(
        options=list(normalize_methods),
        value=normalize_methods[0],
        label="Normalize method",
    )
    return (normalize_method_ui,)


@app.cell
def _(
    df_copy,
    encode_categorical,
    encode_method_ui,
    normalize,
    normalize_method_ui,
):
    df = encode_categorical(
        df_copy,
        ["region", "rank"],
        method=encode_method_ui.value,
    )
    df = normalize(
        df,
        ["region", "rank"],
        method=normalize_method_ui.value,
    )
    return (df,)


@app.cell
def _(encode_method_ui, mo, normalize_method_ui):
    samples_per_label = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=12,
        label="Samples per label",
    )


    mo.hstack([encode_method_ui, normalize_method_ui, samples_per_label])
    return (samples_per_label,)


@app.cell
def _(df, samples_per_label):
    df_sampling = df.group_by("Label", maintain_order=True).map_groups(
        lambda group: group.sample(
            n=int(samples_per_label.value),
            seed=42,
        )
    )
    return (df_sampling,)


@app.cell
def _():
    # mo.ui.dataframe(df_sampling)
    return


@app.cell
def _(df_sampling, np):
    X_polars = df_sampling.drop("Label").to_numpy()
    y_polars = df_sampling.select("Label").to_numpy().flatten()

    X_np = X_polars.T.astype(np.float64)

    X_np = np.clip(X_np, a_min=0, a_max=None)
    return X_np, y_polars


@app.cell
def _():
    from clustering_methods import ssnmf
    from clustering_methods.utils.nndsvd import nndsvd_initialization, NNDSVDConfig
    return NNDSVDConfig, nndsvd_initialization, ssnmf


@app.cell
def _(mo):
    lambda_reg = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
    )
    return (lambda_reg,)


@app.cell
def _(lambda_reg, mo):
    gamma_reg = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.01,
    )
    mo.hstack([mo.md("lambda_reg"), lambda_reg, mo.md("gamma_reg"), gamma_reg])
    return (gamma_reg,)


@app.cell
def _(mo):
    tol = mo.ui.number(
        start=0.0,
        stop=1.0,
        step=1e-12,
        value=1e-4,
        label="Tolerance (tol)",
    )
    return (tol,)


@app.cell
def _(mo):
    n_components_ui = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        value=5,
        label="Number of components",
    )
    return (n_components_ui,)


@app.cell
def _(mo):
    max_iter_ui = mo.ui.number(
        start=1,
        stop=1000,
        step=1,
        value=100,
        label="Max iterations",
    )
    return (max_iter_ui,)


@app.cell
def _(mo):
    random_state_ui = mo.ui.number(
        start=0,
        stop=10000,
        step=1,
        value=42,
        label="Random state",
    )
    return (random_state_ui,)


@app.cell
def _(max_iter_ui, mo, n_components_ui, random_state_ui, tol):
    n_neighbors_ui = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        value=3,
        label="Number of neighbors (graph)",
    )
    n_neighbors_ui
    mo.hstack([tol, n_components_ui, max_iter_ui, random_state_ui, n_neighbors_ui])
    return (n_neighbors_ui,)


@app.cell
def _(mo):
    verbose_ui = mo.ui.checkbox(
        label="Verbose logging",
        value=True,
    )

    verbose_ui
    return (verbose_ui,)


@app.cell
def _(mo):
    nndsvd_variant = mo.ui.dropdown(
        options=["zero", "average", "random"],
        value="zero",
        label="NNDSVD variant",
    )

    nndsvd_variant
    return (nndsvd_variant,)


@app.cell
def _(mo):
    nndsvd_eps = mo.ui.number(
        start=1e-8,
        stop=1e-1,
        step=1e-8,
        value=1e-6,
        label="NNDSVD eps",
    )

    nndsvd_eps
    return (nndsvd_eps,)


@app.cell
def _(
    NNDSVDConfig,
    X_np,
    gamma_reg,
    lambda_reg,
    max_iter_ui,
    n_components_ui,
    n_neighbors_ui,
    nndsvd_eps,
    nndsvd_initialization,
    nndsvd_variant,
    random_state_ui,
    ssnmf,
    tol,
    verbose_ui,
):
    base_config = ssnmf.BaseNMFConfig(
        n_components=int(n_components_ui.value),
        max_iter=int(max_iter_ui.value),
        tol=tol.value,
        random_state=int(random_state_ui.value),
        verbose=bool(verbose_ui.value),
    )
    config = ssnmf.SSNMFDConfig(
        ssnmf_config=base_config,
        lambda_reg=lambda_reg.value,
        gamma_reg=gamma_reg.value,
        n_neighbors=int(n_neighbors_ui.value),
    )

    model = ssnmf.SSNMFD(config)

    initialize_config = NNDSVDConfig(
        X=X_np,
        rank=base_config.n_components,
        variant=nndsvd_variant.value,
        eps=float(nndsvd_eps.value),
    )
    W0, H0 = nndsvd_initialization(initialize_config)

    model.W = W0
    model.H = H0
    return (model,)


@app.cell
def _(X_np, model):
    test = model.fit(X_np)
    return


@app.cell
def _(X_np, model, np):
    W_opt = model.W
    H_opt = model.H

    recon = model.reconstruct()
    fro_err = np.linalg.norm(X_np - recon, "fro")
    sparsity = np.mean(H_opt == 0)

    # print(f"Frobenius norm of reconstruction error: {fro_err:.4f}")
    # print(f"Sparsity of H matrix: {sparsity:.4f}")
    return H_opt, fro_err


@app.cell
def _(H_opt, fro_err, time_string, y_polars):
    with open(f"ssnmf_results_{time_string}.txt", "w") as f:
        for row in H_opt:
            for item in row:
                f.write(f"{item:.4f}\t")
            f.write("\n")

        y = y_polars.tolist()
        for item in y:
            f.write(f"{item:6d}\t",)
        f.write("\n\n")
        f.write(f"Frobenius norm of reconstruction error: {fro_err:.4f}\n")
    return


@app.cell
def _():
    # y_polars
    return


@app.cell
def _(H_opt, np, y_polars):
    # H_optとy_polarsから直接confusion matrixを計算
    if H_opt is not None and y_polars is not None:
        # 各サンプルを「最大値の成分」でクラスタ割り当て
        cluster_assign = np.argmax(H_opt, axis=0)

        clusters = np.unique(cluster_assign)
        uniq_labels = np.unique(y_polars)

        conf = np.zeros((clusters.size, uniq_labels.size), dtype=int)
        for i, cl in enumerate(clusters):
            for j, lb in enumerate(uniq_labels):
                mask = y_polars == lb
                conf[i, j] = np.sum(cluster_assign[mask] == cl)

        # marimoが変更を検知できるように、confをリストに変換
        conf_list = conf.tolist()
    else:
        clusters = None
        conf_list = None
        uniq_labels = None
    return clusters, conf_list, uniq_labels


@app.cell
def _(H_opt, clusters, mo, n_components_ui):
    # 予測クラスタ数とn_componentsの比較
    if clusters is not None and n_components_ui is not None and H_opt is not None:
        predicted_cluster_count = len(clusters)
        n_components_value = int(n_components_ui.value)

        match_status = "✅ 一致" if predicted_cluster_count == n_components_value else "❌ 不一致"

        comparison_text = f"""
    **クラスタ数比較:**
    - 予測されたクラスタ数: {predicted_cluster_count}
    - n_components: {n_components_value}
    - 状態: {match_status}
    """
        if predicted_cluster_count != n_components_value:
            # どの成分が使われていないかを確認
            used_components = sorted(clusters.tolist())
            all_components = list(range(n_components_value))
            unused_components = [c for c in all_components if c not in used_components]

            comparison_text += f"\n⚠️ 注意: {n_components_value - predicted_cluster_count}個の成分が使用されていません。\n"
            comparison_text += f"- 使用されている成分: {used_components}\n"
            comparison_text += f"- 使用されていない成分: {unused_components}\n\n"

            # 使用されていない成分のH行列の値を確認
            if len(unused_components) > 0:
                comparison_text += "**使用されていない成分の情報:**\n"
                for comp_idx in unused_components:
                    comp_values = H_opt[comp_idx, :]
                    max_val = comp_values.max()
                    mean_val = comp_values.mean()
                    comparison_text += f"- 成分{comp_idx}: 最大値={max_val:.4f}, 平均値={mean_val:.4f}\n"

            comparison_text += "\n**なぜこのようなことが起こるのか？**\n"
            comparison_text += """
    1. **NMFの学習過程**: NMFは`n_components`個の成分を学習しますが、最適化の過程で一部の成分が他の成分より常に小さくなる場合があります。

    2. **argmaxによる割り当て**: 各サンプルに対して`argmax(H, axis=0)`で最大値の成分を選ぶため、すべてのサンプルで最大にならない成分は「使われない成分」になります。

    3. **考えられる原因**:
       - 初期化の問題（NNDSVDで一部の成分が弱く初期化された）
       - 正則化（lambda_reg, gamma_reg）が強すぎて一部の成分が抑制された
       - データの構造上、実際にはそれだけの成分数が必要なかった
       - 収束が不十分で、一部の成分が発達しなかった

    4. **対処法**:
       - `lambda_reg`や`gamma_reg`を小さくする
       - `max_iter`を増やして収束を改善する
       - NNDSVDの`variant`を変更する（"average"や"random"を試す）
       - `n_components`を実際に使われている数に合わせる
    """

    mo.md(comparison_text)
    return


@app.cell
def _(clusters, conf_list, mo, np, uniq_labels):
    import altair as alt
    import pandas as pd

    # 混同行列のヒートマップ（Altair）
    conf_chart = None
    if conf_list is not None and clusters is not None and uniq_labels is not None:
        # conf_listからnumpy配列を再構築（marimoが変更を検知できるように）
        _conf = np.array(conf_list, dtype=int)

        # 混同行列をDataFrameに変換
        confusion_data = []
        for ii, clr in enumerate(clusters):
            for jj, ul in enumerate(uniq_labels):
                confusion_data.append({
                    "Cluster": f"Cluster {clr}",
                    "Label": f"Label {ul}",
                    "Count": int(_conf[ii, jj]),
                })
        confusion_df = pd.DataFrame(confusion_data)

        confusion_chart = (
            alt.Chart(confusion_df)
            .mark_rect()
            .encode(
                x=alt.X("Label:O", title="True Label", sort=None),
                y=alt.Y("Cluster:O", title="Predicted Cluster", sort=None),
                color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), title="Count"),
                tooltip=["Cluster", "Label", "Count"],
            )
            .properties(
                width=400,
                height=300,
                title="Confusion Matrix: Cluster Assignment vs True Labels",
            )
        )

        # 数値を表示するテキストレイヤーを追加
        confusion_text = (
            alt.Chart(confusion_df)
            .mark_text(baseline="middle", fontSize=12, fontWeight="bold")
            .encode(
                x="Label:O",
                y="Cluster:O",
                text="Count:Q",
                color=alt.condition(
                    alt.datum.Count > _conf.max() / 2,
                    alt.value("white"),
                    alt.value("black"),
                ),
            )
        )

        confusion_chart = (confusion_chart + confusion_text).configure_view(
            strokeWidth=0
        )
        conf_chart = mo.ui.altair_chart(confusion_chart)
    else:
        print("Cannot plot confusion matrix: data not available.")


    return alt, conf_chart, pd


@app.cell
def _(H_opt, alt, conf_chart, mo, np, pd, y_polars):
    # H行列のヒートマップ（Altair）
    h_matrics_chart = None
    if H_opt is not None and y_polars is not None:
        # ラベル順にソート
        sort_idx = np.argsort(y_polars)
        H_sorted = H_opt[:, sort_idx]
        labels_sorted = y_polars[sort_idx]

        # H行列をDataFrameに変換
        h_data = []
        for iii in range(H_sorted.shape[0]):
            for jjj in range(H_sorted.shape[1]):
                h_data.append({
                    "Component": f"Component {iii}",
                    "Sample": jjj,
                    "Value": float(H_sorted[iii, jjj]),
                    "Label": int(labels_sorted[jjj]),
                })
        h_df = pd.DataFrame(h_data)

        h_chart = (
            alt.Chart(h_df)
            .mark_rect()
            .encode(
                x=alt.X("Sample:O", title="Sample (sorted by label)", axis=alt.Axis(labels=False)),
                y=alt.Y("Component:O", title="Component", sort=None),
                color=alt.Color("Value:Q", scale=alt.Scale(scheme="viridis"), title="Value"),
                tooltip=["Component", "Sample", "Value", "Label"],
            )
            .properties(
                width=600,
                height=300,
                title="H Matrix Heatmap (sorted by true label)",
            )
        )

        # ラベルの境界線を追加（ルールとして）
        label_changes = np.where(np.diff(labels_sorted) != 0)[0] + 1
        if len(label_changes) > 0:
            boundary_data = pd.DataFrame({
                "x": label_changes.tolist(),
            })
            boundary_rule = (
                alt.Chart(boundary_data)
                .mark_rule(color="red", strokeDash=[5, 5], strokeWidth=1, opacity=0.7)
                .encode(x=alt.X("x:Q", scale=alt.Scale(domain=[0, H_sorted.shape[1]])))
            )
            h_chart = h_chart + boundary_rule

        h_matrics_chart = mo.ui.altair_chart(h_chart)

    mo.hstack([h_matrics_chart, conf_chart])
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
