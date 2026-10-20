"""Streamlit interface for DataSense AI."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common.config import DISCLAIMER_ANALYTICS, DATASENSE_NAME, MAX_UPLOAD_MB
from common.utils import dataframe_to_csv_bytes, safe_json
from datasense.cleaning import CleaningPlan, ColumnCleaningRule, apply_cleaning_plan
from datasense.eda import (
    categorical_distribution,
    correlation_matrix,
    group_summary,
    numeric_distribution,
    top_correlations,
    value_counts_table,
)
from datasense.interpretation import build_interpretation, interpretation_to_markdown
from datasense.io import DatasetValidationError, load_dataset
from datasense.modeling import train_model
from datasense.profiling import profile_dataframe
from datasense.reporting import build_report_pdf
from ui.shared import (
    download_row,
    format_number,
    hero,
    metric_card,
    notice,
    sample_files,
    show_json,
    step_label,
)

STATE_KEYS = {
    "dataset": "ds_dataset",
    "cleaned": "ds_cleaned",
    "model": "ds_model",
    "report": "ds_report",
}


def _reset_state() -> None:
    for key in STATE_KEYS.values():
        st.session_state.pop(key, None)


def _store_dataset(loaded) -> None:
    _reset_state()
    st.session_state[STATE_KEYS["dataset"]] = loaded


def _current_dataframe() -> pd.DataFrame:
    cleaned = st.session_state.get(STATE_KEYS["cleaned"])
    if cleaned is not None:
        return cleaned.dataframe
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    return dataset.dataframe if dataset is not None else pd.DataFrame()


def _cleaning_actions() -> list[str]:
    cleaned = st.session_state.get(STATE_KEYS["cleaned"])
    return list(cleaned.actions) if cleaned is not None else []


def _render_upload() -> None:
    step_label("Langkah 1")
    st.subheader("Unggah dataset")
    st.caption(
        f"Format yang didukung: CSV dan XLSX. Ukuran maksimum {MAX_UPLOAD_MB} MB. "
        "File hanya diproses di mesin ini dan tidak dikirim ke layanan eksternal."
    )

    uploaded = st.file_uploader(
        "Pilih file dataset",
        type=["csv", "xlsx"],
        key="ds_uploader",
        help="Baris pertama harus berisi nama kolom yang unik.",
    )
    samples = sample_files(exclude={"flood_history.csv"})
    sample_choice = st.selectbox(
        "Atau gunakan dataset contoh",
        options=["(tidak menggunakan contoh)"] + [path.name for path in samples],
        key="ds_sample_choice",
    )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        load_clicked = st.button("Muat dataset", type="primary", width="stretch")
    with col_b:
        if st.button("Bersihkan sesi", width="content"):
            _reset_state()
            st.rerun()

    if not load_clicked:
        return

    try:
        if uploaded is not None:
            loaded = load_dataset(uploaded.name, uploaded.getvalue())
        elif sample_choice != "(tidak menggunakan contoh)":
            path = next(path for path in samples if path.name == sample_choice)
            loaded = load_dataset(path.name, path.read_bytes())
        else:
            st.warning("Pilih file terlebih dahulu atau gunakan dataset contoh.")
            return
    except DatasetValidationError as exc:
        st.error(f"Dataset ditolak: {exc}")
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Terjadi kesalahan saat membaca dataset: {exc}")
        return

    _store_dataset(loaded)
    st.success(f"Dataset '{loaded.name}' berhasil dimuat.")


def _render_overview() -> None:
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is None:
        notice("Belum ada dataset. Unggah dataset pada tab <b>Upload</b>.", "info")
        return

    profile = profile_dataframe(dataset.dataframe)
    st.subheader("Ringkasan dataset")
    cards = st.columns(4)
    cards[0].markdown(metric_card("Baris", f"{profile.row_count:,}", "observasi"), unsafe_allow_html=True)
    cards[1].markdown(metric_card("Kolom", f"{profile.column_count:,}", "variabel"), unsafe_allow_html=True)
    cards[2].markdown(metric_card("Sel kosong", f"{profile.missing_pct:.1f}%", f"{profile.missing_cells:,} sel"), unsafe_allow_html=True)
    cards[3].markdown(metric_card("Skor kualitas", f"{profile.quality_score}/100", "heuristik"), unsafe_allow_html=True)

    if dataset.warnings:
        for warning in dataset.warnings:
            notice(warning, "info")
    if dataset.sheet_name:
        st.caption(f"Sheet dibaca: {dataset.sheet_name}")

    show_json(
        {
            "file": dataset.name,
            "ukuran_byte": dataset.size_bytes,
            "sha256": dataset.sha256,
            "sheet": dataset.sheet_name,
        },
        "Metadata file",
    )

    st.markdown("#### Pratinjau data")
    rows = st.slider("Jumlah baris pratinjau", 5, min(200, max(5, profile.row_count)), 10, key="ds_preview_rows")
    st.dataframe(dataset.dataframe.head(rows), width="stretch")

    st.markdown("#### Profil kolom")
    table = pd.DataFrame(
        [
            {
                "Kolom": column.name,
                "Tipe": column.dtype,
                "Jenis": column.kind,
                "Non-null": column.non_null,
                "Kosong": column.null_count,
                "Kosong %": round(column.null_pct, 2),
                "Unik": column.unique_count,
                "Masalah": "; ".join(column.issues) if column.issues else "-",
            }
            for column in profile.columns
        ]
    )
    st.dataframe(table, width="stretch", hide_index=True)

    numeric = profile.numeric_columns
    if numeric:
        st.markdown("#### Statistik deskriptif kolom numerik")
        describe = dataset.dataframe[numeric].describe().T
        describe.insert(0, "Kolom", describe.index)
        st.dataframe(describe.reset_index(drop=True), width="stretch", hide_index=True)

    if profile.warnings:
        st.markdown("#### Peringatan kualitas data")
        for warning in profile.warnings:
            notice(warning, "danger")


def _render_cleaning() -> None:
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is None:
        notice("Unggah dataset terlebih dahulu.", "info")
        return

    step_label("Langkah 2")
    st.subheader("Pembersihan data")
    st.caption(
        "Setiap tindakan dicatat dan ditampilkan setelah diterapkan. Data asli tetap tersimpan "
        "pada sesi ini sampai Anda memuat ulang dataset."
    )
    df = dataset.dataframe

    with st.form("ds_cleaning_form"):
        general_a, general_b = st.columns(2)
        with general_a:
            drop_duplicates = st.checkbox("Hapus baris duplikat", value=True)
            strip_whitespace = st.checkbox("Hapus spasi berlebih pada teks", value=True)
        with general_b:
            drop_empty_columns = st.checkbox("Hapus kolom yang seluruhnya kosong", value=True)
            keep_columns = st.multiselect(
                "Kolom yang dipertahankan (kosongkan untuk semua)",
                options=list(df.columns),
                default=[],
            )

        st.markdown("##### Aturan per kolom")
        st.caption("Pilih strategi nilai kosong dan konversi tipe bila diperlukan.")
        rules: list[ColumnCleaningRule] = []
        strategy_options = {
            "Biarkan": "keep",
            "Hapus baris": "drop_rows",
            "Hapus kolom": "drop_column",
            "Isi rata-rata": "fill_mean",
            "Isi median": "fill_median",
            "Isi modus": "fill_mode",
            "Isi konstanta": "fill_constant",
        }
        convert_options = {
            "Tidak diubah": "none",
            "Numerik": "numeric",
            "Kategorikal": "categorical",
            "Tanggal/waktu": "datetime",
        }
        for index, column in enumerate(df.columns):
            with st.expander(f"{column}  ·  {df[column].dtype}  ·  kosong {int(df[column].isna().sum())}"):
                col_a, col_b, col_c = st.columns([2, 2, 1])
                with col_a:
                    strategy_label = st.selectbox(
                        "Strategi nilai kosong",
                        options=list(strategy_options),
                        index=0,
                        key=f"ds_strategy_{index}",
                    )
                with col_b:
                    convert_label = st.selectbox(
                        "Konversi tipe",
                        options=list(convert_options),
                        index=0,
                        key=f"ds_convert_{index}",
                    )
                with col_c:
                    constant = st.text_input("Konstanta", key=f"ds_constant_{index}")
                strategy = strategy_options[strategy_label]
                if strategy == "keep" and convert_options[convert_label] == "none":
                    continue
                parsed_constant = constant
                if strategy == "fill_constant":
                    try:
                        parsed_constant = float(constant)
                    except (TypeError, ValueError):
                        parsed_constant = constant
                rules.append(
                    ColumnCleaningRule(
                        column=column,
                        strategy=strategy,
                        constant=parsed_constant,
                        convert_to=convert_options[convert_label],
                    )
                )

        submitted = st.form_submit_button("Terapkan pembersihan", type="primary")

    if submitted:
        plan = CleaningPlan(
            drop_duplicates=drop_duplicates,
            keep_columns=keep_columns or None,
            strip_whitespace=strip_whitespace,
            drop_fully_empty_columns=drop_empty_columns,
            rules=rules,
        )
        try:
            result = apply_cleaning_plan(df, plan)
        except ValueError as exc:
            st.error(str(exc))
        else:
            st.session_state[STATE_KEYS["cleaned"]] = result
            st.session_state.pop(STATE_KEYS["model"], None)
            st.session_state.pop(STATE_KEYS["report"], None)
            st.success("Pembersihan diterapkan.")

    cleaned = st.session_state.get(STATE_KEYS["cleaned"])
    if cleaned is not None:
        st.markdown("#### Tindakan yang dilakukan")
        for action in cleaned.actions:
            st.markdown(f"- {action}")
        for warning in cleaned.warnings:
            notice(warning, "danger")
        before, after = cleaned.before_shape, cleaned.after_shape
        st.caption(
            f"Bentuk data: {before[0]}x{before[1]} -> {after[0]}x{after[1]} "
            f"({cleaned.rows_removed} baris, {cleaned.columns_removed} kolom dihapus)."
        )
        st.dataframe(cleaned.dataframe.head(20), width="stretch")
        download_row(
            [
                (
                    "Unduh dataset bersih (CSV)",
                    dataframe_to_csv_bytes(cleaned.dataframe),
                    "datasense_clean.csv",
                    "text/csv",
                )
            ],
            key_prefix="clean",
        )
        if st.button("Batalkan pembersihan", key="ds_undo_cleaning"):
            st.session_state.pop(STATE_KEYS["cleaned"], None)
            st.session_state.pop(STATE_KEYS["model"], None)
            st.rerun()
    else:
        notice("Belum ada tindakan pembersihan yang diterapkan.", "info")


def _render_eda() -> None:
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is None:
        notice("Unggah dataset terlebih dahulu.", "info")
        return

    df = _current_dataframe()
    profile = profile_dataframe(df)
    if st.session_state.get(STATE_KEYS["cleaned"]) is not None:
        st.caption("Analisis menggunakan dataset yang sudah dibersihkan.")

    step_label("Langkah 3")
    st.subheader("Exploratory Data Analysis")

    numeric_tab, categorical_tab, correlation_tab, group_tab = st.tabs(
        ["Distribusi numerik", "Kategori", "Korelasi", "Agregasi"]
    )

    with numeric_tab:
        if not profile.numeric_columns:
            notice("Tidak ada kolom numerik pada dataset.", "info")
        else:
            column = st.selectbox("Kolom numerik", profile.numeric_columns, key="ds_eda_numeric")
            try:
                distribution = numeric_distribution(df, column)
            except ValueError as exc:
                st.error(str(exc))
            else:
                figure = go.Figure(
                    go.Bar(
                        x=[
                            (distribution["bins"][index] + distribution["bins"][index + 1]) / 2
                            for index in range(len(distribution["counts"]))
                        ],
                        y=distribution["counts"],
                        marker_color="#12708C",
                    )
                )
                figure.update_layout(
                    title=f"Distribusi {column}",
                    xaxis_title=column,
                    yaxis_title="Frekuensi",
                    bargap=0.05,
                    height=380,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(figure, width="stretch")

                stats = st.columns(5)
                stats[0].metric("Rata-rata", format_number(distribution["mean"]))
                stats[1].metric("Median", format_number(distribution["median"]))
                stats[2].metric("Std dev", format_number(distribution["std"]))
                stats[3].metric("Min / Maks", f"{format_number(distribution['min'])} / {format_number(distribution['max'])}")
                stats[4].metric("Outlier (IQR)", f"{distribution['outliers']}")
                if abs(distribution["skew"]) > 1:
                    notice(
                        f"Distribusi miring (skew = {distribution['skew']:.2f}); "
                        "pertimbangkan transformasi atau metode statistik non-parametrik.",
                        "info",
                    )

                st.markdown("##### Nilai terbanyak")
                st.dataframe(value_counts_table(df, column, top_n=10), width="stretch", hide_index=True)

    with categorical_tab:
        candidates = profile.categorical_columns + profile.boolean_columns + profile.text_columns
        if not candidates:
            notice("Tidak ada kolom kategorikal pada dataset.", "info")
        else:
            column = st.selectbox("Kolom kategorikal", candidates, key="ds_eda_categorical")
            top_n = st.slider("Jumlah kategori ditampilkan", 5, 30, 12, key="ds_eda_topn")
            try:
                distribution = categorical_distribution(df, column, top_n=top_n)
            except ValueError as exc:
                st.error(str(exc))
            else:
                figure = px.bar(
                    x=distribution["labels"],
                    y=distribution["values"],
                    labels={"x": column, "y": "Jumlah"},
                    title=f"Frekuensi kategori {column}",
                    color=distribution["values"],
                    color_continuous_scale="Teal",
                )
                figure.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(figure, width="stretch")
                st.caption(
                    f"{distribution['unique_count']} kategori unik. "
                    f"Kategori teratas '{distribution['top']}' sebanyak {distribution['top_count']}."
                )
                st.dataframe(value_counts_table(df, column, top_n=top_n), width="stretch", hide_index=True)

    with correlation_tab:
        if len(profile.numeric_columns) < 2:
            notice("Korelasi membutuhkan minimal dua kolom numerik.", "info")
        else:
            method = st.selectbox(
                "Metode korelasi",
                options=["pearson", "spearman", "kendall"],
                key="ds_eda_method",
                help="Pearson untuk hubungan linear; Spearman/Kendall untuk hubungan monotonik.",
            )
            try:
                matrix = correlation_matrix(df, profile.numeric_columns, method=method)
            except ValueError as exc:
                st.error(str(exc))
            else:
                figure = px.imshow(
                    matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    aspect="auto",
                    title=f"Matriks korelasi ({method})",
                )
                figure.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(figure, width="stretch")

                pairs = top_correlations(matrix, limit=8)
                if pairs:
                    st.markdown("##### Korelasi terkuat")
                    st.dataframe(pd.DataFrame(pairs), width="stretch", hide_index=True)
                    notice(
                        "Korelasi mengukur keterkaitan statistik, bukan hubungan sebab-akibat.",
                        "info",
                    )

    with group_tab:
        if not profile.numeric_columns or not (profile.categorical_columns or profile.boolean_columns):
            notice("Agregasi membutuhkan minimal satu kolom kategorikal dan satu kolom numerik.", "info")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                group_column = st.selectbox(
                    "Kelompokkan berdasarkan",
                    profile.categorical_columns + profile.boolean_columns,
                    key="ds_eda_group",
                )
            with col_b:
                value_column = st.selectbox("Nilai numerik", profile.numeric_columns, key="ds_eda_value")
            try:
                summary = group_summary(df, group_column, value_column)
            except ValueError as exc:
                st.error(str(exc))
            else:
                st.dataframe(summary, width="stretch", hide_index=True)
                figure = px.bar(
                    summary.head(15),
                    x=group_column,
                    y="rata_rata",
                    error_y="standar_deviasi",
                    title=f"Rata-rata {value_column} per {group_column}",
                    color="rata_rata",
                    color_continuous_scale="Teal",
                )
                figure.update_layout(height=380, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(figure, width="stretch")


def _render_modeling() -> None:
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is None:
        notice("Unggah dataset terlebih dahulu.", "info")
        return

    df = _current_dataframe()
    profile = profile_dataframe(df)
    step_label("Langkah 4")
    st.subheader("Pemodelan machine learning")
    st.caption(
        "Model dilatih pada data uji terpisah dan dievaluasi dengan metrik yang sesuai. "
        "Hasil tidak boleh dianggap sebagai prediksi pasti."
    )

    with st.form("ds_model_form"):
        target = st.selectbox("Kolom target", options=list(df.columns), key="ds_target")
        default_task = (
            "regression"
            if target in profile.numeric_columns
            else "classification"
        )
        task_label = st.radio(
            "Jenis masalah",
            options=["Regresi (nilai numerik)", "Klasifikasi (kategori)"],
            index=0 if default_task == "regression" else 1,
            horizontal=True,
        )
        task = "regression" if task_label.startswith("Regresi") else "classification"

        available_features = [col for col in df.columns if col != target]
        features = st.multiselect(
            "Fitur (kosongkan untuk semua kolom lain)",
            options=available_features,
            default=[],
        )

        model_options = (
            ["auto", "linear", "ridge", "random_forest"]
            if task == "regression"
            else ["auto", "logistic", "random_forest"]
        )
        model_choice = st.selectbox(
            "Model",
            options=model_options,
            format_func=lambda value: {
                "auto": "Otomatis (bandingkan kandidat)",
                "linear": "Linear Regression",
                "ridge": "Ridge Regression",
                "random_forest": "Random Forest",
                "logistic": "Logistic Regression",
            }[value],
        )
        test_size = st.slider("Proporsi data uji", 0.1, 0.4, 0.2, step=0.05)
        submitted = st.form_submit_button("Latih dan evaluasi model", type="primary")

    if submitted:
        try:
            result = train_model(
                df,
                target=target,
                features=features or None,
                task=task,
                model_choice=model_choice,
                test_size=test_size,
            )
        except (ValueError, KeyError) as exc:
            st.error(f"Pelatihan gagal: {exc}")
        else:
            st.session_state[STATE_KEYS["model"]] = result
            st.session_state.pop(STATE_KEYS["report"], None)
            st.success(f"Model '{result.model_name}' berhasil dilatih.")

    result = st.session_state.get(STATE_KEYS["model"])
    if result is None:
        notice("Belum ada model yang dilatih.", "info")
        return

    st.markdown("#### Hasil evaluasi")
    cards = st.columns(4)
    if result.task == "regression":
        cards[0].markdown(metric_card("R²", format_number(result.metrics.get("r2"), 3)), unsafe_allow_html=True)
        cards[1].markdown(metric_card("RMSE", format_number(result.metrics.get("rmse"), 3)), unsafe_allow_html=True)
        cards[2].markdown(metric_card("MAE", format_number(result.metrics.get("mae"), 3)), unsafe_allow_html=True)
        cards[3].markdown(metric_card("MAPE", format_number(result.metrics.get("mape_pct"), 1, "%")), unsafe_allow_html=True)
    else:
        cards[0].markdown(metric_card("Akurasi", format_number(result.metrics.get("accuracy"), 3)), unsafe_allow_html=True)
        cards[1].markdown(metric_card("F1 (weighted)", format_number(result.metrics.get("f1_weighted"), 3)), unsafe_allow_html=True)
        cards[2].markdown(metric_card("Precision", format_number(result.metrics.get("precision_weighted"), 3)), unsafe_allow_html=True)
        cards[3].markdown(metric_card("Recall", format_number(result.metrics.get("recall_weighted"), 3)), unsafe_allow_html=True)

    st.caption(
        f"Model: {result.model_name} · Target: {result.target} · "
        f"Latih {result.train_rows} baris · Uji {result.test_rows} baris"
    )
    if result.cv_metrics.get("mean") is not None:
        st.caption(
            f"Cross-validation {result.cv_metrics['metric']}: "
            f"{result.cv_metrics['mean']:.3f} ± {result.cv_metrics['std']:.3f}"
        )

    for warning in result.warnings:
        notice(warning, "danger")
    for limitation in result.limitations:
        notice(limitation, "info")

    chart_tab, importance_tab, detail_tab, config_tab = st.tabs(
        ["Aktual vs Prediksi", "Fitur penting", "Detail metrik", "Konfigurasi"]
    )

    with chart_tab:
        frame = result.predictions
        if result.task == "regression":
            figure = px.scatter(
                frame,
                x="actual",
                y="predicted",
                trendline="ols",
                labels={"actual": "Aktual", "predicted": "Prediksi"},
                title="Aktual vs Prediksi (data uji)",
            )
            figure.add_shape(
                type="line",
                x0=float(frame["actual"].min()),
                y0=float(frame["actual"].min()),
                x1=float(frame["actual"].max()),
                y1=float(frame["actual"].max()),
                line=dict(dash="dash", color="#C0392B"),
            )
            figure.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(figure, width="stretch")
        else:
            comparison = pd.concat(
                [
                    frame["actual"].astype(str).value_counts().rename("Aktual"),
                    frame["predicted"].astype(str).value_counts().rename("Prediksi"),
                ],
                axis=1,
            ).fillna(0)
            figure = px.bar(
                comparison.reset_index(),
                x="index",
                y=["Aktual", "Prediksi"],
                barmode="group",
                labels={"index": "Kelas", "value": "Jumlah"},
                title="Perbandingan kelas aktual dan prediksi",
            )
            figure.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(figure, width="stretch")
        st.dataframe(frame.head(50), width="stretch", hide_index=True)

    with importance_tab:
        if result.feature_importance:
            importance = pd.DataFrame(result.feature_importance)
            figure = px.bar(
                importance.sort_values("importance_pct"),
                x="importance_pct",
                y="feature",
                orientation="h",
                title="Kepentingan fitur (relatif)",
                labels={"importance_pct": "Kepentingan (%)", "feature": "Fitur"},
                color="importance_pct",
                color_continuous_scale="Teal",
            )
            figure.update_layout(height=420, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(figure, width="stretch")
            notice(
                "Kepentingan fitur menunjukkan kontribusi statistik terhadap prediksi model, "
                "bukan bukti sebab-akibat.",
                "info",
            )
        else:
            notice("Model ini tidak menyediakan kepentingan fitur.", "info")

    with detail_tab:
        st.markdown("##### Metrik data uji")
        metric_frame = pd.DataFrame(
            [{"metrik": key, "nilai": value} for key, value in result.metrics.items()]
        )
        st.dataframe(metric_frame, width="stretch", hide_index=True)
        if result.confusion is not None:
            st.markdown("##### Confusion matrix")
            labels = [item["label"] for item in result.per_class]
            matrix = result.confusion
            figure = px.imshow(
                matrix,
                x=labels,
                y=labels,
                text_auto=True,
                color_continuous_scale="Blues",
                labels={"x": "Prediksi", "y": "Aktual"},
            )
            figure.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(figure, width="stretch")
            st.markdown("##### Metrik per kelas")
            st.dataframe(pd.DataFrame(result.per_class), width="stretch", hide_index=True)

    with config_tab:
        show_json(result.config, "Konfigurasi pelatihan")
        download_row(
            [
                (
                    "Unduh prediksi (CSV)",
                    dataframe_to_csv_bytes(result.predictions),
                    "datasense_predictions.csv",
                    "text/csv",
                )
            ],
            key_prefix="model",
        )


def _render_report() -> None:
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is None:
        notice("Unggah dataset terlebih dahulu.", "info")
        return

    step_label("Langkah 5")
    st.subheader("Interpretasi dan laporan")
    df = _current_dataframe()
    profile = profile_dataframe(df)
    model_result = st.session_state.get(STATE_KEYS["model"])
    interpretation = build_interpretation(df, profile, _cleaning_actions(), model_result)

    notice(interpretation["disclaimer"], "info")

    sections = [
        ("Gambaran umum", interpretation["overview"]),
        ("Kualitas data", interpretation["quality"]),
        ("Tindakan pembersihan", interpretation["cleaning"]),
        ("Analitik diagnostik", interpretation["diagnostics"]),
        ("Hasil model", interpretation["model"]),
        ("Rekomendasi", interpretation["recommendations"]),
        ("Keterbatasan", interpretation["limitations"]),
    ]
    for title, items in sections:
        with st.expander(title, expanded=title in {"Rekomendasi", "Keterbatasan"}):
            for item in items:
                st.markdown(f"- {item}")

    with st.expander("Lihat interpretasi dalam format Markdown"):
        st.code(interpretation_to_markdown(interpretation), language="markdown")

    if st.button("Susun laporan PDF", type="primary"):
        try:
            pdf = build_report_pdf(
                df=df,
                profile=profile,
                dataset_name=dataset.name,
                dataset_sha256=dataset.sha256,
                cleaning_actions=_cleaning_actions(),
                model_result=model_result,
                interpretation=interpretation,
            )
        except Exception as exc:  # pragma: no cover - report generation guard
            st.error(f"Laporan gagal dibuat: {exc}")
        else:
            st.session_state[STATE_KEYS["report"]] = pdf
            st.success("Laporan PDF siap diunduh.")

    pdf = st.session_state.get(STATE_KEYS["report"])
    if pdf is not None:
        downloads = [
            ("Unduh laporan PDF", pdf, "datasense_report.pdf", "application/pdf"),
            (
                "Unduh dataset bersih (CSV)",
                dataframe_to_csv_bytes(df),
                "datasense_clean.csv",
                "text/csv",
            ),
            (
                "Unduh konfigurasi (JSON)",
                safe_json(
                    {
                        "dataset": dataset.name,
                        "dataset_sha256": dataset.sha256,
                        "profile": profile.to_dict(),
                        "cleaning_actions": _cleaning_actions(),
                        "model": model_result.to_dict() if model_result else None,
                    }
                ).encode("utf-8"),
                "datasense_config.json",
                "application/json",
            ),
        ]
        download_row(downloads, key_prefix="report")
    else:
        notice("Laporan PDF belum dibuat. Klik tombol di atas untuk menyusun laporan.", "info")


def render_datasense() -> None:
    hero(
        f"{DATASENSE_NAME}",
        "Analisis data dan pendukung pengambilan keputusan: unggah, bersihkan, eksplorasi, "
        "modelkan, lalu unduh laporan.",
    )
    dataset = st.session_state.get(STATE_KEYS["dataset"])
    if dataset is not None:
        st.caption(
            f"Dataset aktif: **{dataset.name}** · {dataset.row_count:,} baris · "
            f"{dataset.column_count:,} kolom · SHA-256 `{dataset.sha256[:12]}…`"
        )

    tabs = st.tabs(
        ["1. Upload", "2. Profil", "3. Pembersihan", "4. EDA", "5. Model", "6. Laporan"]
    )
    with tabs[0]:
        _render_upload()
    with tabs[1]:
        _render_overview()
    with tabs[2]:
        _render_cleaning()
    with tabs[3]:
        _render_eda()
    with tabs[4]:
        _render_modeling()
    with tabs[5]:
        _render_report()

    st.divider()
    st.caption(DISCLAIMER_ANALYTICS)
