import os
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from data_logic import (
    BulkFillConfig,
    build_column_description,
    build_dataset_overview_text,
    bulk_fill_missing,
    compute_outlier_records as logic_compute_outlier_records,
    convert_series_type,
    fill_missing_values,
    infer_series_hints,
    summarize_column,
)

HISTORY_LIMIT = 15
DEFAULT_PLOT = "Авто"
NUMERIC_PLOTS = (
    DEFAULT_PLOT,
    "Гистограмма",
    "Боксплот",
    "Линейный график",
    "Диаграмма рассеяния",
)
CATEGORICAL_PLOTS = (
    DEFAULT_PLOT,
    "Столбчатая диаграмма",
    "Круговая диаграмма",
)
NO_GROUPING = "Без группировки"
TYPE_OPTIONS = ["Авто", "Строка", "Категория", "Целое", "Дробное", "Дата/время", "Логический"]
METHOD_TO_KEY = {
    "Медиана": "median",
    "Среднее": "mean",
    "Мода": "mode",
    "Константа": "constant",
}
TYPE_TO_KEY = {
    "Авто": "auto",
    "Строка": "string",
    "Категория": "category",
    "Целое": "integer",
    "Дробное": "float",
    "Дата/время": "datetime",
    "Логический": "boolean",
}
settings_state = {
    "separator": "Авто",
    "encoding": "Авто",
    "ask_load_options": True,
    "plot_color": "#2f6690",
    "plot_bins": 20,
    "top_n": 10,
    "marker_size": 40,
    "show_grid": True,
    "show_mean": True,
    "show_median": True,
    "outlier_multiplier": 1.5,
    "bulk_numeric_method": "median",
    "bulk_categorical_method": "mode",
    "bulk_numeric_constant": "",
    "bulk_categorical_constant": "",
    "bulk_fallback_to_global": True,
}

# Глобальные переменные
my_df = None
history = deque(maxlen=HISTORY_LIMIT)
selected_now = None
now_sep = settings_state["separator"]
current_file_path = None
last_conversion_preview = None
last_conversion_target = None

current_outlier_records = []
current_outlier_lookup = {}
selected_outlier_rows = set()
current_outlier_artist = None
current_boxplot_bounds = None


def configure_styles():
    style.configure("Green.TButton", background="#3caa3c", foreground="white", font=("Arial", 12, "bold"))
    style.configure("Danger.TButton", background="#d62828", foreground="white", font=("Arial", 11, "bold"))
    style.configure("Accent.TButton", background="#2f6690", foreground="white", font=("Arial", 11, "bold"))


def format_value(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def set_text_widget(widget, text):
    widget.configure(state=tk.NORMAL)
    widget.delete("1.0", tk.END)
    widget.insert("1.0", text)
    widget.configure(state=tk.DISABLED)


def get_int_var(var_obj, default_value, min_value):
    try:
        return max(min_value, int(var_obj.get()))
    except (TypeError, ValueError, tk.TclError):
        return default_value


def get_float_setting(name, default_value, min_value):
    value = settings_state.get(name, default_value)
    try:
        return max(min_value, float(value))
    except (TypeError, ValueError):
        return default_value


def update_color_swatch():
    if "color_swatch" in globals() and color_swatch.winfo_exists():
        color_swatch.configure(bg=settings_state["plot_color"])


def apply_grid_if_needed():
    if "show_grid_var" in globals() and show_grid_var.get():
        ax.grid(alpha=0.25)


def read_preview_text(path):
    for encoding_name in ("utf-8", "cp1251"):
        try:
            with open(path, "r", encoding=encoding_name) as source:
                return "".join(source.readlines()[:8]).strip()
        except UnicodeDecodeError:
            continue
        except OSError:
            return "Не удалось открыть файл для предпросмотра."
    return "Не удалось построить предпросмотр. Выберите кодировку вручную."


def ask_load_options(path):
    dialog = tk.Toplevel(root)
    dialog.title("Параметры загрузки")
    dialog.geometry("620x430")
    dialog.transient(root)
    dialog.grab_set()
    dialog.columnconfigure(0, weight=1)
    dialog.rowconfigure(2, weight=1)

    separator_var = tk.StringVar(value=settings_state["separator"])
    encoding_var = tk.StringVar(value=settings_state["encoding"])
    result = {}

    ttk.Label(dialog, text=f"Файл: {os.path.basename(path)}", padding=(12, 12, 12, 6)).grid(row=0, column=0, sticky="w")

    options_frame = ttk.Frame(dialog, padding=(12, 0, 12, 8))
    options_frame.grid(row=1, column=0, sticky="ew")
    ttk.Label(options_frame, text="Разделитель:").grid(row=0, column=0, sticky="w")
    ttk.Combobox(options_frame, textvariable=separator_var, values=("Авто", ";", ",", "|", ":", "Tab"), state="readonly", width=12).grid(
        row=0,
        column=1,
        sticky="w",
        padx=(8, 16),
    )
    ttk.Label(options_frame, text="Кодировка:").grid(row=0, column=2, sticky="w")
    ttk.Combobox(options_frame, textvariable=encoding_var, values=("Авто", "utf-8", "cp1251"), state="readonly", width=12).grid(
        row=0,
        column=3,
        sticky="w",
        padx=(8, 0),
    )

    preview_frame = ttk.LabelFrame(dialog, text="Предпросмотр первых строк", padding=10)
    preview_frame.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))
    preview_frame.rowconfigure(0, weight=1)
    preview_frame.columnconfigure(0, weight=1)
    preview_text = tk.Text(preview_frame, wrap="none", padx=10, pady=10)
    preview_text.grid(row=0, column=0, sticky="nsew")
    preview_text.insert("1.0", read_preview_text(path))
    preview_text.configure(state=tk.DISABLED)

    button_frame = ttk.Frame(dialog, padding=(12, 0, 12, 12))
    button_frame.grid(row=3, column=0, sticky="e")

    def submit_load_options():
        result["separator"] = separator_var.get()
        result["encoding"] = encoding_var.get()
        dialog.destroy()

    ttk.Button(button_frame, text="Отмена", command=dialog.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(button_frame, text="Загрузить", style="Green.TButton", command=submit_load_options).pack(side="right")

    root.wait_window(dialog)
    return result or None


def read_csv_with_options(path, separator_label, encoding_label):
    separator_map = {"Авто": None, ";": ";", ",": ",", "|": "|", ":": ":", "Tab": "\t"}
    current_separator = separator_map.get(separator_label, None)
    engine = "python" if current_separator is None else None
    encoding_candidates = ["utf-8", "cp1251"] if encoding_label == "Авто" else [encoding_label]
    last_error = None

    for encoding_name in encoding_candidates:
        try:
            return pd.read_csv(path, sep=current_separator, engine=engine, encoding=encoding_name)
        except Exception as error:
            last_error = error

    raise last_error


def get_selected_group_columns():
    result = []
    for value in (group_fill_var_1.get(), group_fill_var_2.get()):
        if value and value != NO_GROUPING and value not in result:
            result.append(value)
    return tuple(result)


def parse_constant_for_selected_column(raw_value):
    if my_df is not None and selected_now and pd.api.types.is_numeric_dtype(my_df[selected_now]):
        try:
            return float(raw_value)
        except ValueError as error:
            raise ValueError("Для числового столбца константа тоже должна быть числом.") from error
    return raw_value


def draw_placeholder(message):
    ax.clear()
    ax.set_axis_off()
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    hide_outlier_panel()
    fig.tight_layout()
    schedule.draw_idle()


def reset_plot_selector():
    plot_type_combo.configure(values=(DEFAULT_PLOT,), state="disabled")
    plot_type_var.set(DEFAULT_PLOT)


def get_available_plot_types(series):
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC_PLOTS
    return CATEGORICAL_PLOTS


def refresh_plot_selector():
    if my_df is None or not selected_now or selected_now not in my_df.columns:
        reset_plot_selector()
        return

    available_plots = get_available_plot_types(my_df[selected_now])
    plot_type_combo.configure(values=available_plots, state="readonly")
    if plot_type_var.get() not in available_plots:
        plot_type_var.set(DEFAULT_PLOT)


def refresh_grouping_combos():
    if "group_combo_1" not in globals():
        return

    if my_df is None or not selected_now or selected_now not in my_df.columns:
        values = (NO_GROUPING,)
        state = "disabled"
    else:
        values = tuple([NO_GROUPING] + [column_name for column_name in my_df.columns if column_name != selected_now])
        state = "readonly"

    group_combo_1.configure(values=values, state=state)
    group_combo_2.configure(values=values, state=state)

    if group_fill_var_1.get() not in values:
        group_fill_var_1.set(NO_GROUPING)
    if group_fill_var_2.get() not in values:
        group_fill_var_2.set(NO_GROUPING)


def refresh_column_details():
    if "column_info_text" not in globals():
        return

    if my_df is None or not selected_now or selected_now not in my_df.columns:
        set_text_widget(column_info_text, "Выберите столбец, чтобы увидеть его описание.")
        if "column_type_label" in globals():
            column_type_label.configure(text="Текущий тип: -")
        if "type_tab_type_label" in globals():
            type_tab_type_label.configure(text="Текущий тип: -")
        return

    set_text_widget(column_info_text, build_column_description(my_df, selected_now))
    if "column_type_label" in globals():
        column_type_label.configure(text=f"Текущий тип: {my_df[selected_now].dtype}")
    if "type_tab_type_label" in globals():
        type_tab_type_label.configure(text=f"Текущий тип: {my_df[selected_now].dtype}")


def refresh_overview_panel():
    if "overview_text" not in globals():
        return

    if my_df is None:
        set_text_widget(overview_text, "После загрузки файла здесь появится краткое описание набора данных.")
        return

    text = build_dataset_overview_text(my_df)
    if selected_now and selected_now in my_df.columns:
        hints = "\n".join(f"- {hint}" for hint in infer_series_hints(my_df[selected_now]))
        text += f"\n\nПодсказки по выбранному столбцу:\n{hints}"
    set_text_widget(overview_text, text)


def build_bulk_settings_text():
    numeric_text = {"median": "Медиана", "mean": "Среднее", "constant": "Константа"}[settings_state["bulk_numeric_method"]]
    categorical_text = {"mode": "Мода", "constant": "Константа"}[settings_state["bulk_categorical_method"]]
    text = (
        f"Числовые столбцы: {numeric_text}; "
        f"категориальные: {categorical_text}; "
        f"fallback к общему столбцу: {'включён' if settings_state['bulk_fallback_to_global'] else 'выключен'}."
    )
    if settings_state["bulk_numeric_method"] == "constant" and settings_state["bulk_numeric_constant"].strip():
        text += f" Числовая константа: {settings_state['bulk_numeric_constant']}."
    if settings_state["bulk_categorical_method"] == "constant" and settings_state["bulk_categorical_constant"].strip():
        text += f" Категориальная константа: {settings_state['bulk_categorical_constant']}."
    return text


def open_dataset_description():
    if my_df is None:
        messagebox.showinfo("Описание данных", "Сначала загрузите таблицу.")
        return

    dialog = tk.Toplevel(root)
    dialog.title("Описание данных")
    dialog.geometry("880x640")

    notebook = ttk.Notebook(dialog)
    notebook.pack(fill="both", expand=True, padx=12, pady=12)

    overview_tab = ttk.Frame(notebook)
    columns_tab = ttk.Frame(notebook)
    notebook.add(overview_tab, text="Обзор")
    notebook.add(columns_tab, text="Столбцы")

    overview_tab.rowconfigure(0, weight=1)
    overview_tab.columnconfigure(0, weight=1)
    overview_widget = tk.Text(overview_tab, wrap="word", padx=10, pady=10)
    overview_widget.grid(row=0, column=0, sticky="nsew")
    overview_widget.insert("1.0", build_dataset_overview_text(my_df))
    overview_widget.configure(state=tk.DISABLED)

    columns_tab.rowconfigure(1, weight=1)
    columns_tab.columnconfigure(0, weight=1)
    columns_tab.columnconfigure(1, weight=1)
    ttk.Label(columns_tab, text="Подробности по столбцу").grid(row=0, column=1, sticky="w", pady=(0, 8))

    columns_tree = ttk.Treeview(columns_tab, columns=("dtype", "missing", "unique", "hint"), show="headings")
    columns_tree.heading("dtype", text="Тип")
    columns_tree.heading("missing", text="Пропуски")
    columns_tree.heading("unique", text="Уникальные")
    columns_tree.heading("hint", text="Подсказка")
    columns_tree.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 8))

    details_widget = tk.Text(columns_tab, wrap="word", padx=10, pady=10)
    details_widget.grid(row=1, column=1, sticky="nsew")
    details_widget.configure(state=tk.DISABLED)

    for column_name in my_df.columns:
        summary = summarize_column(my_df, column_name)
        first_hint = summary["hints"][0] if summary["hints"] else "-"
        columns_tree.insert("", tk.END, iid=column_name, values=(summary["dtype"], summary["missing"], summary["unique"], first_hint))

    def on_description_select(event=None):
        selection = columns_tree.selection()
        if not selection:
            return
        set_text_widget(details_widget, build_column_description(my_df, selection[0]))

    columns_tree.bind("<<TreeviewSelect>>", on_description_select)
    if selected_now and selected_now in my_df.columns:
        columns_tree.selection_set(selected_now)
        on_description_select()


def choose_plot_color():
    color = colorchooser.askcolor(color=settings_state["plot_color"], title="Выберите цвет графика", parent=root)[1]
    if color:
        settings_state["plot_color"] = color
        update_color_swatch()
        show_plot()


def reset_plot_view():
    if "plot_toolbar" in globals():
        plot_toolbar.home()
        schedule.draw_idle()


def preview_type_conversion():
    global last_conversion_preview, last_conversion_target

    if my_df is None or not selected_now:
        messagebox.showwarning("Типы данных", "Сначала выберите столбец.")
        return

    target_key = TYPE_TO_KEY[type_combo_var.get()]
    try:
        last_conversion_preview = convert_series_type(my_df[selected_now], target_key)
    except Exception as error:
        messagebox.showerror("Типы данных", str(error))
        return

    last_conversion_target = target_key
    non_null_count = int(my_df[selected_now].notna().sum())
    lines = [
        f"Столбец: {selected_now}",
        f"Текущий тип: {my_df[selected_now].dtype}",
        f"Целевой тип: {type_combo_var.get()}",
        f"Итоговый dtype: {last_conversion_preview.converted.dtype}",
        f"Непустых значений: {non_null_count}",
        f"Проблемных значений: {last_conversion_preview.invalid_count}",
    ]
    if last_conversion_preview.invalid_examples:
        lines.append("")
        lines.append("Примеры значений, которые не удастся преобразовать:")
        lines.extend(f"- {value}" for value in last_conversion_preview.invalid_examples)
    else:
        lines.append("")
        lines.append("Преобразование выглядит безопасным.")
    set_text_widget(type_preview_text, "\n".join(lines))


def apply_type_conversion():
    global my_df, last_conversion_preview, last_conversion_target

    if my_df is None or not selected_now:
        messagebox.showwarning("Типы данных", "Сначала выберите столбец.")
        return

    target_key = TYPE_TO_KEY[type_combo_var.get()]
    if last_conversion_preview is None or last_conversion_target != target_key:
        preview_type_conversion()
        if last_conversion_preview is None:
            return

    non_null_count = int(my_df[selected_now].notna().sum())
    if last_conversion_preview.invalid_count == non_null_count and target_key not in {"string", "category", "auto"}:
        messagebox.showerror("Типы данных", "Ни одно непустое значение не удалось преобразовать. Применение отменено.")
        return

    if last_conversion_preview.invalid_count > 0:
        if not messagebox.askyesno("Подтверждение", f"{last_conversion_preview.invalid_count} значений станут пустыми. Продолжить?"):
            return

    save_state()
    my_df[selected_now] = last_conversion_preview.converted
    last_conversion_preview = None
    last_conversion_target = None
    update()
    show_plot()
    messagebox.showinfo("Типы данных", "Тип столбца обновлён.")


def set_outlier_details(text):
    outlier_details.configure(state=tk.NORMAL)
    outlier_details.delete("1.0", tk.END)
    outlier_details.insert("1.0", text)
    outlier_details.configure(state=tk.DISABLED)


def show_outlier_panel():
    if not outlier_frame.winfo_manager():
        outlier_frame.pack(fill="x", pady=(10, 0))


def hide_outlier_panel():
    if outlier_frame.winfo_manager():
        outlier_frame.pack_forget()


def update_outlier_buttons_state():
    has_selection = bool(selected_outlier_rows)
    has_outliers = bool(current_outlier_records)
    delete_selected_button.configure(state="normal" if has_selection else "disabled")
    clear_selection_button.configure(state="normal" if has_selection else "disabled")
    delete_all_button.configure(state="normal" if has_outliers else "disabled")


def reset_outlier_state(hide_panel=True):
    global current_outlier_records, current_outlier_lookup
    global selected_outlier_rows, current_outlier_artist, current_boxplot_bounds

    current_outlier_records = []
    current_outlier_lookup = {}
    selected_outlier_rows = set()
    current_outlier_artist = None
    current_boxplot_bounds = None

    for item in outlier_tree.get_children():
        outlier_tree.delete(item)

    outlier_summary_label.configure(text="Переключитесь на boxplot для анализа выбросов.")
    set_outlier_details("Щелкните по точке на boxplot, чтобы увидеть сведения о строке.")
    update_outlier_buttons_state()

    if hide_panel:
        hide_outlier_panel()


def populate_outlier_panel(records, bounds):
    global current_outlier_records, current_outlier_lookup
    global selected_outlier_rows, current_boxplot_bounds

    current_outlier_records = records
    current_outlier_lookup = {record["row_index"]: record for record in records}
    current_boxplot_bounds = bounds
    selected_outlier_rows = set()

    for item in outlier_tree.get_children():
        outlier_tree.delete(item)

    for record in records:
        side_text = "Ниже нижней границы" if record["side"] == "below" else "Выше верхней границы"
        outlier_tree.insert(
            "",
            tk.END,
            iid=str(record["row_index"]),
            values=(record["row_index"] + 1, format_value(record["value"]), side_text),
        )

    summary = (
        f"Границы boxplot: {format_value(bounds['lower'])} .. {format_value(bounds['upper'])}. "
        f"Выбросов найдено: {len(records)}."
    )
    outlier_summary_label.configure(text=summary)

    if records:
        set_outlier_details("Выберите одну или несколько точек на графике либо строки в списке ниже.")
    else:
        set_outlier_details("Выбросы по правилу 1.5 IQR для текущего столбца не найдены.")

    show_outlier_panel()
    update_outlier_buttons_state()


def update_outlier_artist_selection():
    if current_outlier_artist is None or not current_outlier_records:
        return

    facecolors = []
    edgecolors = []
    sizes = []
    base_size = get_int_var(marker_size_var, settings_state["marker_size"], 10) + 20

    for record in current_outlier_records:
        is_selected = record["row_index"] in selected_outlier_rows
        if is_selected:
            facecolors.append("#d62828")
            edgecolors.append("#5f0f40")
            sizes.append(base_size + 40)
        else:
            facecolors.append("#ff9f1c")
            edgecolors.append("#7f5539")
            sizes.append(base_size)

    current_outlier_artist.set_facecolors(facecolors)
    current_outlier_artist.set_edgecolors(edgecolors)
    current_outlier_artist.set_sizes(sizes)
    schedule.draw_idle()


def update_outlier_details():
    if not selected_outlier_rows:
        if current_outlier_records:
            set_outlier_details("Выберите одну или несколько точек на графике либо строки в списке ниже.")
        else:
            set_outlier_details("Выбросы по правилу 1.5 IQR для текущего столбца не найдены.")
        update_outlier_buttons_state()
        return

    selected_rows = sorted(selected_outlier_rows)
    if len(selected_rows) == 1:
        row_index = selected_rows[0]
        row_data = my_df.loc[row_index]
        lines = [
            f"Строка: {row_index + 1}",
            f"Значение '{selected_now}': {format_value(row_data[selected_now])}",
        ]
        if current_boxplot_bounds is not None:
            lines.append(
                f"Границы выбросов: {format_value(current_boxplot_bounds['lower'])} .. "
                f"{format_value(current_boxplot_bounds['upper'])}"
            )
        lines.append("")
        lines.append("Данные строки:")
        for column_name, value in row_data.items():
            lines.append(f"{column_name}: {value}")
        set_outlier_details("\n".join(lines))
    else:
        rows_text = ", ".join(str(row + 1) for row in selected_rows)
        lines = [
            f"Выбрано выбросов: {len(selected_rows)}",
            f"Строки: {rows_text}",
        ]
        if current_boxplot_bounds is not None:
            lines.append(
                f"Границы выбросов: {format_value(current_boxplot_bounds['lower'])} .. "
                f"{format_value(current_boxplot_bounds['upper'])}"
            )
        lines.append("")
        lines.append("Можно удалить только выбранные строки или все найденные выбросы текущего столбца.")
        set_outlier_details("\n".join(lines))

    update_outlier_buttons_state()


def on_outlier_tree_select(event=None):
    global selected_outlier_rows
    selected_outlier_rows = {int(item_id) for item_id in outlier_tree.selection()}
    update_outlier_artist_selection()
    update_outlier_details()


def clear_outlier_selection():
    if not outlier_tree.selection():
        return
    outlier_tree.selection_remove(outlier_tree.selection())
    on_outlier_tree_select()


def on_plot_pick(event):
    if event.artist is not current_outlier_artist or not current_outlier_records:
        return

    toggled_rows = {current_outlier_records[index]["row_index"] for index in event.ind}
    current_selection = {int(item_id) for item_id in outlier_tree.selection()}

    for row_index in toggled_rows:
        if row_index in current_selection:
            current_selection.remove(row_index)
        else:
            current_selection.add(row_index)

    if current_selection:
        ordered_selection = tuple(str(row_index) for row_index in sorted(current_selection))
        outlier_tree.selection_set(ordered_selection)
        outlier_tree.see(ordered_selection[-1])
    else:
        outlier_tree.selection_remove(outlier_tree.selection())

    on_outlier_tree_select()


def compute_outlier_records(series):
    return logic_compute_outlier_records(series, multiplier=get_float_setting("outlier_multiplier", 1.5, 0.5))


def delete_rows_from_dataframe(row_indexes, confirmation_message, success_message):
    global my_df

    unique_indexes = sorted(set(row_indexes))
    if not unique_indexes:
        return

    if not messagebox.askyesno("Подтверждение", confirmation_message):
        return

    save_state()
    my_df = my_df.drop(index=unique_indexes).reset_index(drop=True)
    update()
    show_plot()
    messagebox.showinfo("Удаление", success_message)


def delete_selected_outliers():
    if not selected_outlier_rows:
        messagebox.showwarning("Выбросы", "Сначала выберите точки или строки в списке выбросов.")
        return

    count = len(selected_outlier_rows)
    delete_rows_from_dataframe(
        selected_outlier_rows,
        f"Удалить {count} выбранных выбросов из столбца '{selected_now}'?",
        f"Удалено строк: {count}.",
    )


def delete_all_outliers():
    if not current_outlier_records:
        messagebox.showinfo("Выбросы", "Для текущего boxplot выбросы не найдены.")
        return

    row_indexes = [record["row_index"] for record in current_outlier_records]
    count = len(row_indexes)
    delete_rows_from_dataframe(
        row_indexes,
        f"Удалить все {count} выбросов из столбца '{selected_now}'?",
        f"Удалено всех выбросов: {count}.",
    )


def draw_histogram(series):
    data = series.dropna()
    ax.hist(
        data,
        bins=get_int_var(plot_bins_var, settings_state["plot_bins"], 5),
        color=settings_state["plot_color"],
        edgecolor="white",
        alpha=0.9,
    )
    if show_mean_var.get():
        mean_ = data.mean()
        ax.axvline(mean_, color="#d62828", linestyle="--", linewidth=2, label=f"Среднее: {mean_:.2f}")
    if show_median_var.get():
        median_ = data.median()
        ax.axvline(median_, color="#f4d35e", linestyle="-", linewidth=2, label=f"Медиана: {median_:.2f}")
    ax.set_title(f"Распределение: {selected_now}")
    ax.set_xlabel(selected_now)
    ax.set_ylabel("Частота")
    if show_mean_var.get() or show_median_var.get():
        ax.legend()
    apply_grid_if_needed()


def draw_boxplot(series):
    global current_outlier_artist

    data = series.dropna()
    records, bounds = compute_outlier_records(data)

    ax.boxplot(
        data.values,
        patch_artist=True,
        showfliers=False,
        widths=0.35,
        boxprops={"facecolor": "#a8dadc", "edgecolor": "#1d3557"},
        medianprops={"color": "#d62828", "linewidth": 2},
        whiskerprops={"color": "#1d3557", "linewidth": 1.5},
        capprops={"color": "#1d3557", "linewidth": 1.5},
    )
    ax.set_title(f"Boxplot: {selected_now}")
    ax.set_ylabel("Значение")
    ax.set_xticks([1])
    ax.set_xticklabels([selected_now])
    apply_grid_if_needed()

    populate_outlier_panel(records, bounds)
    current_outlier_artist = None

    if records:
        if len(records) == 1:
            x_positions = np.array([1.0])
        else:
            x_positions = 1 + np.linspace(-0.06, 0.06, len(records))
        y_values = np.array([record["value"] for record in records], dtype=float)
        current_outlier_artist = ax.scatter(
            x_positions,
            y_values,
            s=get_int_var(marker_size_var, settings_state["marker_size"], 10) + 20,
            c="#ff9f1c",
            edgecolors="#7f5539",
            linewidths=1.2,
            zorder=4,
            picker=5,
        )
        update_outlier_artist_selection()


def draw_line_chart(series):
    data = series.dropna()
    row_numbers = data.index + 1

    ax.plot(row_numbers, data.values, color=settings_state["plot_color"], linewidth=1.8, marker="o", markersize=3)
    ax.set_title(f"Линейный график: {selected_now}")
    ax.set_xlabel("Номер строки")
    ax.set_ylabel(selected_now)
    apply_grid_if_needed()


def draw_scatter_chart(series):
    data = series.dropna()
    row_numbers = data.index + 1

    ax.scatter(
        row_numbers,
        data.values,
        color=settings_state["plot_color"],
        alpha=0.8,
        s=get_int_var(marker_size_var, settings_state["marker_size"], 10),
    )
    ax.set_title(f"Диаграмма рассеяния: {selected_now}")
    ax.set_xlabel("Номер строки")
    ax.set_ylabel(selected_now)
    apply_grid_if_needed()


def draw_bar_chart(series):
    counts = series.dropna().astype(str).value_counts().head(get_int_var(top_n_var, settings_state["top_n"], 3))

    ax.bar(counts.index, counts.values, color=settings_state["plot_color"])
    ax.set_title(f"Топ 10 значений: {selected_now}")
    ax.set_xlabel(selected_now)
    ax.set_ylabel("Количество")
    ax.tick_params(axis="x", rotation=35)
    apply_grid_if_needed()


def draw_pie_chart(series):
    counts = series.dropna().astype(str).value_counts()
    top_n = get_int_var(top_n_var, settings_state["top_n"], 3)
    if len(counts) > top_n:
        top_counts = counts.head(top_n - 1).copy()
        top_counts.loc["Остальные"] = counts.iloc[top_n - 1 :].sum()
    else:
        top_counts = counts

    ax.pie(
        top_counts.values,
        labels=top_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax.set_title(f"Структура значений: {selected_now}")


def open_settings():
    mini_window = tk.Toplevel(root)
    mini_window.title("Настройки")
    mini_window.geometry("720x520")
    mini_window.transient(root)
    mini_window.grab_set()

    notebook = ttk.Notebook(mini_window)
    notebook.pack(fill="both", expand=True, padx=12, pady=12)

    general_tab = ttk.Frame(notebook)
    graph_tab = ttk.Frame(notebook)
    bulk_tab = ttk.Frame(notebook)
    notebook.add(general_tab, text="Общие")
    notebook.add(graph_tab, text="Графики")
    notebook.add(bulk_tab, text="Массовые действия")

    theme_var = tk.StringVar(value=style.theme_use())
    separator_var = tk.StringVar(value=settings_state["separator"])
    encoding_var = tk.StringVar(value=settings_state["encoding"])
    ask_load_var = tk.BooleanVar(value=settings_state["ask_load_options"])

    general_tab.columnconfigure(1, weight=1)
    ttk.Label(general_tab, text="Тема интерфейса:").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
    ttk.Combobox(general_tab, textvariable=theme_var, values=style.theme_names(), state="readonly").grid(row=0, column=1, sticky="ew", padx=12, pady=(12, 6))
    ttk.Label(general_tab, text="Разделитель по умолчанию:").grid(row=1, column=0, sticky="w", padx=12, pady=6)
    ttk.Combobox(general_tab, textvariable=separator_var, values=("Авто", ";", ",", "|", ":", "Tab"), state="readonly").grid(row=1, column=1, sticky="ew", padx=12, pady=6)
    ttk.Label(general_tab, text="Кодировка по умолчанию:").grid(row=2, column=0, sticky="w", padx=12, pady=6)
    ttk.Combobox(general_tab, textvariable=encoding_var, values=("Авто", "utf-8", "cp1251"), state="readonly").grid(row=2, column=1, sticky="ew", padx=12, pady=6)
    ttk.Checkbutton(general_tab, text="Всегда спрашивать параметры при загрузке файла", variable=ask_load_var).grid(
        row=3,
        column=0,
        columnspan=2,
        sticky="w",
        padx=12,
        pady=(10, 6),
    )

    selected_plot_color = {"value": settings_state["plot_color"]}

    def pick_settings_color():
        color = colorchooser.askcolor(color=selected_plot_color["value"], title="Цвет графиков", parent=mini_window)[1]
        if color:
            selected_plot_color["value"] = color
            settings_color_swatch.configure(bg=color)

    graph_tab.columnconfigure(1, weight=1)
    settings_bins_var = tk.IntVar(value=settings_state["plot_bins"])
    settings_top_n_var = tk.IntVar(value=settings_state["top_n"])
    settings_marker_size_var = tk.IntVar(value=settings_state["marker_size"])
    settings_show_grid_var = tk.BooleanVar(value=settings_state["show_grid"])
    settings_show_mean_var = tk.BooleanVar(value=settings_state["show_mean"])
    settings_show_median_var = tk.BooleanVar(value=settings_state["show_median"])
    settings_outlier_var = tk.DoubleVar(value=settings_state["outlier_multiplier"])
    ttk.Label(graph_tab, text="Цвет графиков:").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
    ttk.Button(graph_tab, text="Выбрать", command=pick_settings_color).grid(row=0, column=1, sticky="w", padx=12, pady=(12, 6))
    settings_color_swatch = tk.Label(graph_tab, width=4, bg=selected_plot_color["value"], relief="solid", bd=1)
    settings_color_swatch.grid(row=0, column=2, sticky="w", pady=(12, 6))
    ttk.Label(graph_tab, text="Бины гистограммы:").grid(row=1, column=0, sticky="w", padx=12, pady=6)
    ttk.Spinbox(graph_tab, from_=5, to=100, textvariable=settings_bins_var, width=8).grid(row=1, column=1, sticky="w", padx=12, pady=6)
    ttk.Label(graph_tab, text="Топ категорий:").grid(row=2, column=0, sticky="w", padx=12, pady=6)
    ttk.Spinbox(graph_tab, from_=3, to=30, textvariable=settings_top_n_var, width=8).grid(row=2, column=1, sticky="w", padx=12, pady=6)
    ttk.Label(graph_tab, text="Размер точек:").grid(row=3, column=0, sticky="w", padx=12, pady=6)
    ttk.Spinbox(graph_tab, from_=10, to=150, textvariable=settings_marker_size_var, width=8).grid(row=3, column=1, sticky="w", padx=12, pady=6)
    ttk.Label(graph_tab, text="Множитель IQR для выбросов:").grid(row=4, column=0, sticky="w", padx=12, pady=6)
    ttk.Spinbox(graph_tab, from_=0.5, to=5.0, increment=0.1, textvariable=settings_outlier_var, width=8).grid(row=4, column=1, sticky="w", padx=12, pady=6)
    ttk.Checkbutton(graph_tab, text="Показывать сетку", variable=settings_show_grid_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(10, 4))
    ttk.Checkbutton(graph_tab, text="Показывать линию среднего", variable=settings_show_mean_var).grid(row=6, column=0, columnspan=2, sticky="w", padx=12, pady=4)
    ttk.Checkbutton(graph_tab, text="Показывать линию медианы", variable=settings_show_median_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=12, pady=4)

    bulk_tab.columnconfigure(1, weight=1)
    bulk_numeric_method_var = tk.StringVar(value={"median": "Медиана", "mean": "Среднее", "constant": "Константа"}[settings_state["bulk_numeric_method"]])
    bulk_categorical_method_var = tk.StringVar(value={"mode": "Мода", "constant": "Константа"}[settings_state["bulk_categorical_method"]])
    bulk_numeric_constant_var = tk.StringVar(value=settings_state["bulk_numeric_constant"])
    bulk_categorical_constant_var = tk.StringVar(value=settings_state["bulk_categorical_constant"])
    bulk_fallback_var = tk.BooleanVar(value=settings_state["bulk_fallback_to_global"])
    ttk.Label(bulk_tab, text="Числовые столбцы:").grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))
    ttk.Combobox(bulk_tab, textvariable=bulk_numeric_method_var, values=("Медиана", "Среднее", "Константа"), state="readonly").grid(
        row=0,
        column=1,
        sticky="ew",
        padx=12,
        pady=(12, 6),
    )
    ttk.Label(bulk_tab, text="Числовая константа:").grid(row=1, column=0, sticky="w", padx=12, pady=6)
    ttk.Entry(bulk_tab, textvariable=bulk_numeric_constant_var).grid(row=1, column=1, sticky="ew", padx=12, pady=6)
    ttk.Label(bulk_tab, text="Категориальные столбцы:").grid(row=2, column=0, sticky="w", padx=12, pady=6)
    ttk.Combobox(bulk_tab, textvariable=bulk_categorical_method_var, values=("Мода", "Константа"), state="readonly").grid(
        row=2,
        column=1,
        sticky="ew",
        padx=12,
        pady=6,
    )
    ttk.Label(bulk_tab, text="Категориальная константа:").grid(row=3, column=0, sticky="w", padx=12, pady=6)
    ttk.Entry(bulk_tab, textvariable=bulk_categorical_constant_var).grid(row=3, column=1, sticky="ew", padx=12, pady=6)
    ttk.Checkbutton(bulk_tab, text="Если локальный расчёт не помог, использовать общий по столбцу", variable=bulk_fallback_var).grid(
        row=4,
        column=0,
        columnspan=2,
        sticky="w",
        padx=12,
        pady=(10, 6),
    )

    button_frame = ttk.Frame(mini_window)
    button_frame.pack(fill="x", padx=12, pady=(0, 12))

    def apply_settings():
        global now_sep

        style.theme_use(theme_var.get())
        configure_styles()
        settings_state["separator"] = separator_var.get()
        settings_state["encoding"] = encoding_var.get()
        settings_state["ask_load_options"] = ask_load_var.get()
        settings_state["plot_color"] = selected_plot_color["value"]
        settings_state["plot_bins"] = max(5, settings_bins_var.get())
        settings_state["top_n"] = max(3, settings_top_n_var.get())
        settings_state["marker_size"] = max(10, settings_marker_size_var.get())
        settings_state["show_grid"] = settings_show_grid_var.get()
        settings_state["show_mean"] = settings_show_mean_var.get()
        settings_state["show_median"] = settings_show_median_var.get()
        settings_state["outlier_multiplier"] = max(0.5, float(settings_outlier_var.get()))
        settings_state["bulk_numeric_method"] = METHOD_TO_KEY[bulk_numeric_method_var.get()]
        settings_state["bulk_categorical_method"] = METHOD_TO_KEY[bulk_categorical_method_var.get()]
        settings_state["bulk_numeric_constant"] = bulk_numeric_constant_var.get()
        settings_state["bulk_categorical_constant"] = bulk_categorical_constant_var.get()
        settings_state["bulk_fallback_to_global"] = bulk_fallback_var.get()
        now_sep = settings_state["separator"]

        plot_bins_var.set(settings_state["plot_bins"])
        top_n_var.set(settings_state["top_n"])
        marker_size_var.set(settings_state["marker_size"])
        show_grid_var.set(settings_state["show_grid"])
        show_mean_var.set(settings_state["show_mean"])
        show_median_var.set(settings_state["show_median"])
        update_color_swatch()
        bulk_summary_label.configure(
            text=build_bulk_settings_text()
        )
        update()
        show_plot()
        mini_window.destroy()

    ttk.Button(button_frame, text="Отмена", command=mini_window.destroy).pack(side="right", padx=(8, 0))
    ttk.Button(button_frame, text="Применить", style="Green.TButton", command=apply_settings).pack(side="right")


def show_help():
    help_text = (
        "ИНСТРУКЦИЯ ПО РАБОТЕ:\n\n"
        "1. При открытии файла можно сразу выбрать разделитель и кодировку.\n"
        "2. Слева отображаются столбцы, а ниже — подробное описание выбранного столбца.\n"
        "3. В зоне графика можно менять тип графика, цвет, бины, топ категорий и размер точек.\n"
        "4. Для приближения и перемещения используйте панель Zoom / Pan / Home под графиком.\n"
        "5. Для числовых столбцов доступны гистограмма, boxplot, линия и scatter; для категориальных — bar и pie.\n"
        "6. В режиме boxplot точки за усами можно выбирать мышью или через список снизу, смотреть строку и удалять.\n"
        "7. Во вкладке 'Пропуски' можно заполнять значения по всему столбцу или в разрезе одной/двух группировок.\n"
        "8. Во вкладке 'Типы данных' можно безопасно менять тип столбца с предпросмотром ошибок преобразования.\n"
        "9. В настройках задаются параметры загрузки, графиков и работа кнопки массовых действий.\n"
    )
    messagebox.showinfo("Руководство пользователя", help_text)


def undo_logic():
    global my_df, selected_now

    if not history:
        messagebox.showinfo("Отмена", "Отменять нечего.")
        return

    my_df = history.pop()
    if selected_now not in my_df.columns:
        selected_now = None
    update()
    show_plot()


def load_file():
    global my_df, selected_now, current_file_path, now_sep

    path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if not path:
        return

    load_options = ask_load_options(path) if settings_state["ask_load_options"] else {
        "separator": settings_state["separator"],
        "encoding": settings_state["encoding"],
    }
    if not load_options:
        return

    try:
        my_df = read_csv_with_options(path, load_options["separator"], load_options["encoding"])
    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось прочитать файл: {error}")
        return

    settings_state["separator"] = load_options["separator"]
    settings_state["encoding"] = load_options["encoding"]
    now_sep = settings_state["separator"]
    current_file_path = path
    history.clear()
    selected_now = None
    reset_plot_selector()
    update()
    draw_placeholder("Выберите столбец для анализа.")
    messagebox.showinfo("Ок", "Файл загружен.")


def save_file():
    if my_df is None:
        messagebox.showinfo("Сохранение", "Нет данных для сохранения.")
        return

    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not path:
        return

    try:
        my_df.to_csv(path, index=False)
    except Exception as error:
        messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {error}")


def apply_clean():
    global my_df

    if my_df is None or not selected_now:
        messagebox.showwarning("Выбор столбца", "Сначала выберите столбец.")
        return

    method = method_combo.get()
    if not method:
        messagebox.showwarning("Метод", "Сначала выберите метод обработки.")
        return

    is_numeric = pd.api.types.is_numeric_dtype(my_df[selected_now])

    try:
        if method == "Удалить":
            save_state()
            my_df = my_df.dropna(subset=[selected_now]).reset_index(drop=True)
        else:
            if method in ("Медиана", "Среднее") and not is_numeric:
                raise ValueError(f"{method} используется только для числовых данных.")

            constant_value = None
            if method == "Константа":
                if not constant_value_var.get().strip():
                    raise ValueError("Введите константу для заполнения.")
                constant_value = parse_constant_for_selected_column(constant_value_var.get().strip())

            result = fill_missing_values(
                my_df,
                selected_now,
                METHOD_TO_KEY[method],
                constant_value=constant_value,
                group_columns=get_selected_group_columns(),
                fallback_to_global=group_fill_fallback_var.get(),
            )
            if result.filled_count == 0:
                raise ValueError("Ничего не изменилось: пропусков нет или не хватило данных для заполнения.")

            save_state()
            my_df[selected_now] = result.series

        update()
        show_plot()
    except Exception as error:
        messagebox.showerror("Ошибка", str(error))


def fill_all_with_logica():
    global my_df

    if my_df is None:
        return

    try:
        numeric_constant = None
        categorical_constant = None
        if settings_state["bulk_numeric_method"] == "constant" and settings_state["bulk_numeric_constant"].strip():
            numeric_constant = float(settings_state["bulk_numeric_constant"].strip())
        if settings_state["bulk_categorical_method"] == "constant" and settings_state["bulk_categorical_constant"].strip():
            categorical_constant = settings_state["bulk_categorical_constant"].strip()

        config = BulkFillConfig(
            numeric_method=settings_state["bulk_numeric_method"],
            categorical_method=settings_state["bulk_categorical_method"],
            numeric_constant=numeric_constant,
            categorical_constant=categorical_constant,
            fallback_to_global=settings_state["bulk_fallback_to_global"],
        )
    except ValueError:
        messagebox.showerror("Массовые действия", "Числовая константа в настройках должна быть числом.")
        return

    save_state()
    my_df, report = bulk_fill_missing(my_df, config)
    update()
    show_plot()

    filled_count = sum(report["filled_columns"].values())
    skipped_columns = ", ".join(report["skipped_columns"]) if report["skipped_columns"] else "нет"
    remaining_missing = ", ".join(f"{column}: {count}" for column, count in report["remaining_missing"].items()) or "нет"
    messagebox.showinfo(
        "Массовые действия",
        f"Заполнено значений: {filled_count}.\n"
        f"Пропущенные столбцы: {skipped_columns}.\n"
        f"Оставшиеся пропуски: {remaining_missing}.",
    )


def on_tree_select(event):
    global selected_now

    selection = column_tree.selection()
    if not selection:
        return

    selected_now = column_tree.item(selection[0])["values"][0]
    refresh_grouping_combos()
    refresh_column_details()
    refresh_overview_panel()
    refresh_plot_selector()
    show_plot()


def on_plot_type_change(event=None):
    show_plot()


def show_plot():
    global current_outlier_records, current_outlier_lookup, selected_outlier_rows
    global current_outlier_artist, current_boxplot_bounds

    # Сбрасываем состояние выбросов только если переключаемся с боксплота на другой тип графика
    plot_kind = plot_type_var.get() or DEFAULT_PLOT
    is_boxplot = pd.api.types.is_numeric_dtype(my_df[selected_now]) if my_df is not None and selected_now else False
    is_boxplot = is_boxplot and plot_kind == "Боксплот"

    if not is_boxplot:
        reset_outlier_state(hide_panel=True)

    if my_df is None:
        draw_placeholder("Откройте CSV-файл для начала работы.")
        return

    if not selected_now or selected_now not in my_df.columns:
        refresh_plot_selector()
        draw_placeholder("Выберите столбец для анализа.")
        return

    refresh_plot_selector()
    ax.clear()
    ax.set_axis_on()

    try:
        series = my_df[selected_now]
        if series.dropna().empty:
            draw_placeholder(f"В столбце '{selected_now}' нет данных для построения графика.")
            return

        plot_kind = plot_type_var.get() or DEFAULT_PLOT

        if pd.api.types.is_numeric_dtype(series):
            actual_plot = "Гистограмма" if plot_kind == DEFAULT_PLOT else plot_kind
            if actual_plot == "Гистограмма":
                draw_histogram(series)
            elif actual_plot == "Боксплот":
                draw_boxplot(series)
            elif actual_plot == "Линейный график":
                draw_line_chart(series)
            elif actual_plot == "Диаграмма рассеяния":
                draw_scatter_chart(series)
            else:
                draw_histogram(series)
        else:
            actual_plot = "Столбчатая диаграмма" if plot_kind == DEFAULT_PLOT else plot_kind
            if actual_plot == "Круговая диаграмма":
                draw_pie_chart(series)
            else:
                draw_bar_chart(series)
    except Exception:
        draw_placeholder("Ошибка отрисовки данных.")
        return

    fig.tight_layout()
    schedule.draw_idle()


def update():
    global selected_now

    for item in column_tree.get_children():
        column_tree.delete(item)

    if my_df is None:
        selected_now = None
        status_label.configure(text="Файл не загружен.")
        file_label.configure(text="Файл: не выбран")
        reset_plot_selector()
        refresh_grouping_combos()
        refresh_column_details()
        refresh_overview_panel()
        return

    if selected_now not in my_df.columns:
        selected_now = None

    selected_item_id = None
    for column_name in my_df.columns:
        nans_count = my_df[column_name].isnull().sum()
        dtype = str(my_df[column_name].dtype)
        tags = ("with_NaN",) if nans_count > 0 else ()
        item_id = column_tree.insert("", tk.END, values=(column_name, dtype, nans_count), tags=tags)
        if column_name == selected_now:
            selected_item_id = item_id

    if selected_item_id is not None:
        column_tree.selection_set(selected_item_id)

    total_nans = int(my_df.isnull().sum().sum())
    rows, columns = my_df.shape
    duplicates = int(my_df.duplicated().sum())
    status_label.configure(text=f"Строк: {rows}; Колонок: {columns}; Всего пропусков: {total_nans}; Дубликатов: {duplicates}")
    file_label.configure(text=f"Файл: {os.path.basename(current_file_path) if current_file_path else 'не выбран'}")
    refresh_grouping_combos()
    refresh_column_details()
    refresh_overview_panel()
    refresh_plot_selector()


def save_state():
    if my_df is not None:
        history.append(my_df.copy())


def closing():
    try:
        plt.close("all")
        root.quit()
        root.destroy()
    except Exception:
        pass


# Интерфейс
root = tk.Tk()
root.title("Обработчик табличных данных")
root.geometry("1460x980")
root.minsize(1220, 780)
root.configure(bg="#f7f5ef")

style = ttk.Style()
style.theme_use("clam")
configure_styles()

header_frame = tk.Frame(root, bg="#efeadf", padx=18, pady=14)
header_frame.pack(fill="x")
header_frame.columnconfigure(0, weight=1)
tk.Label(header_frame, text="Обработчик табличных данных", bg="#efeadf", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
tk.Label(
    header_frame,
    text="Графики, описание данных, безопасные типы и настраиваемая очистка.",
    bg="#efeadf",
    fg="#5f5f5f",
    font=("Segoe UI", 10),
).grid(row=1, column=0, sticky="w", pady=(4, 0))

header_actions = tk.Frame(header_frame, bg="#efeadf")
header_actions.grid(row=0, column=1, rowspan=2, sticky="e")
ttk.Button(header_actions, text="Описание", style="Accent.TButton", command=open_dataset_description).pack(side="right", padx=4)
ttk.Button(header_actions, text="Настройки", command=open_settings).pack(side="right", padx=4)
ttk.Button(header_actions, text="Помощь", command=show_help).pack(side="right", padx=4)
ttk.Button(header_actions, text="Отменить", command=undo_logic).pack(side="right", padx=4)
ttk.Button(header_actions, text="Открыть", style="Accent.TButton", command=load_file).pack(side="right", padx=4)
ttk.Button(header_actions, text="Сохранить", style="Green.TButton", command=save_file).pack(side="right", padx=4)

status_label = ttk.Label(root, text="Файл не загружен.", padding=(18, 8))
status_label.pack(fill="x")

main_pw = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_pw.pack(fill="both", expand=True, padx=14, pady=(0, 14))

left_container = ttk.Frame(main_pw)
main_pw.add(left_container, weight=1)
left_container.rowconfigure(0, weight=1)
left_container.columnconfigure(0, weight=1)

left_pw = ttk.PanedWindow(left_container, orient=tk.VERTICAL)
left_pw.grid(row=0, column=0, sticky="nsew")

columns_frame = ttk.LabelFrame(left_pw, text="Столбцы", padding=8)
column_info_frame = ttk.LabelFrame(left_pw, text="Описание выбранного столбца", padding=8)
left_pw.add(columns_frame, weight=3)
left_pw.add(column_info_frame, weight=2)

columns_frame.rowconfigure(2, weight=1)
columns_frame.columnconfigure(0, weight=1)
file_label = ttk.Label(columns_frame, text="Файл: не выбран")
file_label.grid(row=0, column=0, sticky="ew", padx=4, pady=(2, 6))
ttk.Button(columns_frame, text="Открыть описание данных", command=open_dataset_description).grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 6))

column_tree_wrap = ttk.Frame(columns_frame)
column_tree_wrap.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 4))
column_tree_wrap.rowconfigure(0, weight=1)
column_tree_wrap.columnconfigure(0, weight=1)

column_tree = ttk.Treeview(column_tree_wrap, columns=("Name", "Type", "Miss"), show="headings")
column_tree.heading("Name", text="Столбец")
column_tree.heading("Type", text="Тип")
column_tree.heading("Miss", text="Пропуски")
column_tree.column("Name", width=160)
column_tree.column("Type", width=100)
column_tree.column("Miss", width=90)
column_tree.grid(row=0, column=0, sticky="nsew")
column_tree.bind("<<TreeviewSelect>>", on_tree_select)
column_tree.tag_configure("with_NaN", background="#ed4830")

column_tree_scrollbar = ttk.Scrollbar(column_tree_wrap, orient="vertical", command=column_tree.yview)
column_tree_scrollbar.grid(row=0, column=1, sticky="ns")
column_tree.configure(yscrollcommand=column_tree_scrollbar.set)

column_info_frame.rowconfigure(1, weight=1)
column_info_frame.columnconfigure(0, weight=1)
column_type_label = ttk.Label(column_info_frame, text="Текущий тип: -")
column_type_label.grid(row=0, column=0, sticky="w", padx=4, pady=(2, 6))
column_info_text = tk.Text(column_info_frame, wrap="word", height=12, padx=10, pady=10)
column_info_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
column_info_text.configure(state=tk.DISABLED)

right_container = ttk.Frame(main_pw)
main_pw.add(right_container, weight=3)
right_container.rowconfigure(0, weight=1)
right_container.columnconfigure(0, weight=1)

right_pw = ttk.PanedWindow(right_container, orient=tk.VERTICAL)
right_pw.grid(row=0, column=0, sticky="nsew")

schedule_frame = ttk.LabelFrame(right_pw, text="Графики и визуальный анализ", padding=10)
action_frame = ttk.LabelFrame(right_pw, text="Инструменты обработки", padding=10)
right_pw.add(schedule_frame, weight=4)
right_pw.add(action_frame, weight=3)

schedule_frame.rowconfigure(2, weight=1)
schedule_frame.columnconfigure(0, weight=1)

plot_control_frame = ttk.Frame(schedule_frame)
plot_control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
ttk.Label(plot_control_frame, text="Тип графика:").pack(side="left")
plot_type_var = tk.StringVar(value=DEFAULT_PLOT)
plot_type_combo = ttk.Combobox(
    plot_control_frame,
    textvariable=plot_type_var,
    values=(DEFAULT_PLOT,),
    state="disabled",
    width=24,
)
plot_type_combo.pack(side="left", padx=(8, 10))
plot_type_combo.bind("<<ComboboxSelected>>", on_plot_type_change)
ttk.Label(plot_control_frame, text="Цвет:").pack(side="left")
ttk.Button(plot_control_frame, text="Выбрать", command=choose_plot_color).pack(side="left", padx=(8, 4))
color_swatch = tk.Label(plot_control_frame, width=3, bg=settings_state["plot_color"], relief="solid", bd=1)
color_swatch.pack(side="left", padx=(0, 10))
ttk.Button(plot_control_frame, text="Сбросить вид", command=reset_plot_view).pack(side="left", padx=(0, 10))
ttk.Label(plot_control_frame, text="Zoom/Pan доступны на панели ниже.").pack(side="right")

plot_options_frame = ttk.Frame(schedule_frame)
plot_options_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
ttk.Label(plot_options_frame, text="Бины:").pack(side="left")
plot_bins_var = tk.IntVar(value=settings_state["plot_bins"])
plot_bins_spinbox = ttk.Spinbox(plot_options_frame, from_=5, to=100, textvariable=plot_bins_var, width=6, command=show_plot)
plot_bins_spinbox.pack(side="left", padx=(6, 10))
plot_bins_spinbox.bind("<Return>", show_plot)
plot_bins_spinbox.bind("<FocusOut>", show_plot)
ttk.Label(plot_options_frame, text="Топ категорий:").pack(side="left")
top_n_var = tk.IntVar(value=settings_state["top_n"])
top_n_spinbox = ttk.Spinbox(plot_options_frame, from_=3, to=30, textvariable=top_n_var, width=6, command=show_plot)
top_n_spinbox.pack(side="left", padx=(6, 10))
top_n_spinbox.bind("<Return>", show_plot)
top_n_spinbox.bind("<FocusOut>", show_plot)
ttk.Label(plot_options_frame, text="Размер точки:").pack(side="left")
marker_size_var = tk.IntVar(value=settings_state["marker_size"])
marker_size_spinbox = ttk.Spinbox(plot_options_frame, from_=10, to=150, textvariable=marker_size_var, width=6, command=show_plot)
marker_size_spinbox.pack(side="left", padx=(6, 10))
marker_size_spinbox.bind("<Return>", show_plot)
marker_size_spinbox.bind("<FocusOut>", show_plot)
show_grid_var = tk.BooleanVar(value=settings_state["show_grid"])
show_mean_var = tk.BooleanVar(value=settings_state["show_mean"])
show_median_var = tk.BooleanVar(value=settings_state["show_median"])
ttk.Checkbutton(plot_options_frame, text="Сетка", variable=show_grid_var, command=show_plot).pack(side="left", padx=(0, 8))
ttk.Checkbutton(plot_options_frame, text="Среднее", variable=show_mean_var, command=show_plot).pack(side="left", padx=(0, 8))
ttk.Checkbutton(plot_options_frame, text="Медиана", variable=show_median_var, command=show_plot).pack(side="left")

canvas_frame = ttk.Frame(schedule_frame)
canvas_frame.grid(row=2, column=0, sticky="nsew")
canvas_frame.rowconfigure(0, weight=1)
canvas_frame.columnconfigure(0, weight=1)
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
schedule = FigureCanvasTkAgg(fig, master=canvas_frame)
schedule.get_tk_widget().grid(row=0, column=0, sticky="nsew")
schedule.mpl_connect("pick_event", on_plot_pick)

toolbar_frame = ttk.Frame(schedule_frame)
toolbar_frame.grid(row=3, column=0, sticky="ew")
plot_toolbar = NavigationToolbar2Tk(schedule, toolbar_frame, pack_toolbar=False)
plot_toolbar.update()
plot_toolbar.pack(side="left")

outlier_frame = ttk.LabelFrame(schedule_frame, text="Выбросы boxplot", padding=10)
outlier_summary_label = ttk.Label(outlier_frame, text="")
outlier_summary_label.pack(anchor="w")

outlier_tree_container = ttk.Frame(outlier_frame)
outlier_tree_container.pack(fill="x", pady=(8, 8))

outlier_tree = ttk.Treeview(
    outlier_tree_container,
    columns=("row", "value", "side"),
    show="headings",
    height=6,
    selectmode="extended",
)
outlier_tree.heading("row", text="Строка")
outlier_tree.heading("value", text="Значение")
outlier_tree.heading("side", text="Положение")
outlier_tree.column("row", width=80)
outlier_tree.column("value", width=120)
outlier_tree.column("side", width=180)
outlier_tree.pack(side="left", fill="x", expand=True)
outlier_tree.bind("<<TreeviewSelect>>", on_outlier_tree_select)

outlier_scrollbar = ttk.Scrollbar(outlier_tree_container, orient="vertical", command=outlier_tree.yview)
outlier_scrollbar.pack(side="right", fill="y")
outlier_tree.configure(yscrollcommand=outlier_scrollbar.set)

outlier_button_frame = ttk.Frame(outlier_frame)
outlier_button_frame.pack(fill="x", pady=(0, 8))

delete_selected_button = ttk.Button(
    outlier_button_frame,
    text="Удалить выбранные",
    style="Danger.TButton",
    command=delete_selected_outliers,
)
delete_selected_button.pack(side="left", padx=(0, 5))

delete_all_button = ttk.Button(
    outlier_button_frame,
    text="Удалить все выбросы",
    style="Danger.TButton",
    command=delete_all_outliers,
)
delete_all_button.pack(side="left", padx=5)

clear_selection_button = ttk.Button(
    outlier_button_frame,
    text="Снять выделение",
    command=clear_outlier_selection,
)
clear_selection_button.pack(side="left", padx=5)

ttk.Label(outlier_frame, text="Информация о выбранных точках:").pack(anchor="w")
outlier_details = tk.Text(outlier_frame, height=7, wrap="word", padx=10, pady=10)
outlier_details.pack(fill="x", expand=True)
outlier_details.configure(state=tk.DISABLED)

action_notebook = ttk.Notebook(action_frame)
action_notebook.pack(fill="both", expand=True)

missing_tab = ttk.Frame(action_notebook)
types_tab = ttk.Frame(action_notebook)
overview_tab = ttk.Frame(action_notebook)
action_notebook.add(missing_tab, text="Пропуски")
action_notebook.add(types_tab, text="Типы данных")
action_notebook.add(overview_tab, text="Сводка")

for index in range(4):
    missing_tab.columnconfigure(index, weight=1 if index == 3 else 0)

ttk.Label(missing_tab, text="Метод:").grid(row=0, column=0, sticky="w", padx=10, pady=(12, 6))
values_of_methods = ["Медиана", "Среднее", "Мода", "Константа", "Удалить"]
method_combo = ttk.Combobox(missing_tab, values=values_of_methods, state="readonly", width=20)
method_combo.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(12, 6))

ttk.Label(missing_tab, text="Группировать по:").grid(row=1, column=0, sticky="w", padx=10, pady=6)
group_fill_var_1 = tk.StringVar(value=NO_GROUPING)
group_fill_var_2 = tk.StringVar(value=NO_GROUPING)
group_combo_1 = ttk.Combobox(missing_tab, textvariable=group_fill_var_1, values=(NO_GROUPING,), state="disabled", width=20)
group_combo_1.grid(row=1, column=1, sticky="w", padx=(0, 8), pady=6)
group_combo_2 = ttk.Combobox(missing_tab, textvariable=group_fill_var_2, values=(NO_GROUPING,), state="disabled", width=20)
group_combo_2.grid(row=1, column=2, sticky="w", padx=(0, 8), pady=6)

group_fill_fallback_var = tk.BooleanVar(value=True)
ttk.Checkbutton(
    missing_tab,
    text="Если группа не помогла, использовать общий расчёт по столбцу",
    variable=group_fill_fallback_var,
).grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(2, 8))

ttk.Label(missing_tab, text="Константа:").grid(row=3, column=0, sticky="w", padx=10, pady=6)
constant_value_var = tk.StringVar(value="")
ttk.Entry(missing_tab, textvariable=constant_value_var, width=22).grid(row=3, column=1, sticky="w", pady=6)
ttk.Button(
    missing_tab,
    text="Применить к столбцу",
    style="Green.TButton",
    command=apply_clean,
).grid(row=3, column=3, padx=10, sticky="e", pady=6)

ttk.Separator(missing_tab, orient=tk.HORIZONTAL).grid(row=4, column=0, sticky="ew", columnspan=4, padx=10, pady=14)
ttk.Label(missing_tab, text="Массовые действия").grid(row=5, column=0, sticky="w", padx=10)
bulk_summary_label = ttk.Label(missing_tab, text="", wraplength=760, justify="left")
bulk_summary_label.grid(row=6, column=0, columnspan=4, sticky="w", padx=10, pady=(4, 8))
ttk.Button(missing_tab, text="Настроить", command=open_settings).grid(row=7, column=0, sticky="w", padx=10, pady=(0, 10))
ttk.Button(missing_tab, text="Заполнить всё", style="Accent.TButton", command=fill_all_with_logica).grid(row=7, column=3, sticky="e", padx=10, pady=(0, 10))
ttk.Label(
    missing_tab,
    text="Можно заполнять пропуски по всему столбцу или в разрезе одной/двух группировок.",
    wraplength=760,
    justify="left",
).grid(row=8, column=0, columnspan=4, sticky="w", padx=10, pady=(0, 10))

for index in range(3):
    types_tab.columnconfigure(index, weight=1 if index == 2 else 0)
types_tab.rowconfigure(4, weight=1)
type_tab_type_label = ttk.Label(types_tab, text="Текущий тип: -")
type_tab_type_label.grid(row=0, column=0, columnspan=3, sticky="w", padx=10, pady=(12, 8))
ttk.Label(types_tab, text="Новый тип:").grid(row=1, column=0, sticky="w", padx=10, pady=6)
type_combo_var = tk.StringVar(value="Дробное")
ttk.Combobox(types_tab, textvariable=type_combo_var, values=TYPE_OPTIONS, state="readonly", width=18).grid(row=1, column=1, sticky="w", pady=6)
ttk.Button(types_tab, text="Проверить", command=preview_type_conversion).grid(row=1, column=2, sticky="e", padx=10, pady=6)
ttk.Button(types_tab, text="Применить тип", style="Green.TButton", command=apply_type_conversion).grid(row=2, column=2, sticky="e", padx=10, pady=(0, 8))
ttk.Label(
    types_tab,
    text="Перед применением программа покажет, сколько значений потеряются при преобразовании.",
    wraplength=760,
    justify="left",
).grid(row=2, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8))
ttk.Label(types_tab, text="Предпросмотр:").grid(row=3, column=0, sticky="w", padx=10, pady=(4, 6))
type_preview_text = tk.Text(types_tab, wrap="word", padx=10, pady=10)
type_preview_text.grid(row=4, column=0, columnspan=3, sticky="nsew", padx=10, pady=(0, 10))
type_preview_text.configure(state=tk.DISABLED)

overview_tab.rowconfigure(1, weight=1)
overview_tab.columnconfigure(0, weight=1)
overview_control_frame = ttk.Frame(overview_tab)
overview_control_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(12, 8))
ttk.Button(overview_control_frame, text="Обновить", command=refresh_overview_panel).pack(side="left")
ttk.Button(overview_control_frame, text="Расширенное описание", command=open_dataset_description).pack(side="left", padx=8)
overview_text = tk.Text(overview_tab, wrap="word", padx=10, pady=10)
overview_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
overview_text.configure(state=tk.DISABLED)

bulk_summary_label.configure(text=build_bulk_settings_text())
reset_plot_selector()
reset_outlier_state()
refresh_grouping_combos()
refresh_column_details()
refresh_overview_panel()
draw_placeholder("Откройте CSV-файл для начала работы.")

root.protocol("WM_DELETE_WINDOW", closing)
root.mainloop()
