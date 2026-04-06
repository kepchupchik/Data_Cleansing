import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

# Глобальные переменные
my_df = None
# deque с maxlen автоматически удаляет старые записи, если их больше 15
history = deque(maxlen=15)
selected_now = None
# По умолчанию разделитель сам определяется
now_sep = "Авто"


def open_settings():
    global now_sep
    # Настройки
    mini_window = tk.Toplevel(root)
    mini_window.title("Настройка")
    mini_window.geometry("300x300")
    # Выбор темы
    ttk.Label(mini_window, text="Выберите тему:").pack(pady=5, padx=5)
    combo_theme = ttk.Combobox(mini_window, values=style.theme_names(), state="readonly")
    # По умолчанию мы можем ввести в текстовое поле в Combobox любое значение, даже то, которого нет в списке.
    # Передав параметру "state" значение "readonly", мы можем установить для виджета состояние только для чтения
    combo_theme.set(style.theme_use())
    combo_theme.pack(pady=5, padx=5)

    # Выбор разделителя
    ttk.Label(mini_window, text="Разделитель CSV:").pack(pady=5, padx=5)
    combo_sep = ttk.Combobox(mini_window, values=["Авто", ";", ":", "Tab"], state="readonly")
    combo_sep.set(now_sep)
    combo_sep.pack(pady=5, padx=5)

    ttk.Button(mini_window, text="Ок", command=lambda: apply_settings(combo_theme, combo_sep, mini_window)).pack(pady=5,
                                                                                                                 padx=5)
    # lambda тут необходима чтобы запуск команды был только тогда, когда нажали на кнопку. Без лямбды это будет сразу активироваться при прочтении строки


def apply_settings(combo_theme, combo_sep, mini_window):
    global now_sep
    style.configure(combo_theme.get())
    now_sep = combo_sep.get()
    update()
    mini_window.destroy()


def show_help():
    # Помощь
    help_text = (
        "ИНСТРУКЦИЯ ПО РАБОТЕ:\n\n"
        "1. Настройка: В меню 'Настройки' (шестеренка) выберите нужный разделитель CSV "
        "(или оставьте 'Авто') и тему оформления.\n"
        "2. Загрузка: Нажмите 'Открыть' и выберите файл. Программа поддерживает кодировки UTF-8 и CP1251.\n"
        "3. Навигация: Слева в таблице выберите столбец. Красный цвет сигнализирует о наличии пропусков.\n"
        "4. Анализ: Справа отобразится график. Для чисел: пунктир - Среднее, сплошная - Медиана.\n"
        "5. Очистка: Выберите метод в списке и нажмите 'Применить к выбранному'.\n"
        "6. Авто-очистка: Кнопка 'Заполнить всё' обработает все столбцы автоматически.\n"
        "7. Отмена: Кнопка '↩' возвращает данные назад (доступно до 15 последних действий).\n"
        "8. Сохранение: Нажмите 'Сохранить', чтобы записать результат в новый файл.\n\n"
        "Техподдержка: телеграмм @K_e_p_c_h_u_p"
    )
    messagebox.showinfo("Руководство пользователя", help_text)


def undo_logic():
    # Логика отмены действия назад. Возвращает последнее значение из истории df, если такое есть
    global my_df, history
    if len(history) > 0:
        my_df = history.pop()  # Одновременно возвращаем элемент и удаляем
        update()
        show_plot()
    else:
        messagebox.showinfo("Отмена", "Отменять нечего")


def load_file():
    # Загрузка файла CSV и очищение истории. Работает на try, except c выводом ошибки
    global my_df, now_sep
    path = filedialog.askopenfilename(filetypes=[('CSV files', '*.csv')])
    if path:
        # Словарь для сопоставления с реальными символами
        sep_dict = {"Авто": None, ";": ";", ":": ":", "Tab": "\t"}
        this_sep = sep_dict.get(now_sep, None)
        # Если "Авто", то engine="python" сам разберется
        engine = "python" if this_sep is None else None

        try:
            # Пробуем UTF-8
            my_df = pd.read_csv(path, sep=this_sep, engine=engine, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                # Пробуем Windows-1251
                my_df = pd.read_csv(path, sep=this_sep, engine=engine, encoding="cp1251")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось прочитать файл {e}")
                return
    history.clear()
    update()
    messagebox.showinfo("Ок", "Файл загружен")


def save_file():
    # Сохранение DataFrame в CSV файл. Работает на try, except c выводом ошибки
    global my_df
    if my_df is None:
        messagebox.showinfo("Сохранение", "Нет данных для сохранения")
        return

    path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[('CSV files', '*.csv')])

    if path:
        try:
            my_df.to_csv(path, index=False)
            # index=False для того, чтобы не появились новые колонки с индексами при сохранении
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось сохранить файл: {e}")


def apply_clean():
    # Подтверждение очистки на try, except c выводом ошибки
    global my_df, selected_now

    # Проверка того, выбрано ли что-то для того, чтобы заменять
    if my_df is None or not selected_now:
        messagebox.showwarning("Выбор столбца", "Сначала выберите столбец")
        return

    method = method_combo.get()
    is_numeric = pd.api.types.is_numeric_dtype(my_df[selected_now])  # проверка того, что тип данных числовой
    try:
        save_state()
        if method == "Медиана":
            # Медиана подходит только для числовых данных
            if not is_numeric:
                raise ValueError("Медиана используется только для числовых данных")
            my_df[selected_now] = my_df[selected_now].fillna(my_df[selected_now].median())

        elif method == "Среднее":
            # Среднее подходит только для числовых данных
            if not is_numeric:
                raise ValueError("Среднее используется только для числовых данных")
            my_df[selected_now] = my_df[selected_now].fillna(my_df[selected_now].mean())

        elif method == "Мода":
            moda = my_df[selected_now].mode()
            # Мода подходит для всех типов данных, но ее не всегда можно вычислить
            if not moda.empty and pd.notna(moda[0]):
                most_popular = moda[0]
                my_df[selected_now] = my_df[selected_now].fillna(most_popular)
            else:
                # Если моды нет или она NaN(если пустой столбец)
                raise ValueError(
                    f"В столбце {selected_now} нельзя определить моду для замены так как нет подходящих данных")

        elif method == "Константа":
            # Для константы если числовой столбец, то просит число для замены
            val = simpledialog.askstring("Константа", f"Введите значения для замены в {selected_now}:")
            if val is None:
                return
            if is_numeric:
                try:
                    val = float(val)
                except ValueError:
                    raise ValueError(f"Cтолбец {selected_now} числовой. Введите число")
            if str(val).strip() == "":
                raise ValueError("Вы ввели пустую строку. Нужно ввести значение")
            my_df[selected_now] = my_df[selected_now].fillna(val)

        elif method == "Удалить":
            my_df.dropna(subset=[selected_now], inplace=True)
            # inplace=True Модифицирует исходный DataFrame напрямую, вместо создания его копии.
            # Это экономит память, так как изменения
            # происходят «на месте» (in-place), а не через возврат нового объекта
            my_df.reset_index(drop=True, inplace=True)
            # Необходимо для того, чтобы не было дыр(1, 2, 5, 7) после такого большого удаления
        update()
        show_plot()
    except Exception as e:
        messagebox.showerror("Ошибка", str(e))


def fill_all_with_logica():
    # Автоматическое заполнение(по возможности)
    # Для числовых данных используется медиана, для остальных - мода
    # Если с какими-то колонками что-то не получилось, эти колонки записываются в skipped_columns,
    # а в конце пишутся, чтобы пользователь мог сам посмотреть.
    # Такое может произойти, если самым чаще встречающимся значением в столбце оказалось NaN
    global my_df
    if my_df is None:
        return
    save_state()

    skipped_columns = []

    for column in my_df.columns:
        if my_df[column].isnull().sum() > 0:
            # Проверяем есть ли в столбце хотя бы одно не пустое значение, если нет, то берем следующий
            if my_df[column].dropna().empty:
                skipped_columns.append(column)
                continue

            # Медиана для числового типа
            if pd.api.types.is_numeric_dtype(my_df[column]):
                mediana = my_df[column].median()
                my_df[column] = my_df[column].fillna(mediana)
            else:
                # Мода для остальных типов
                moda = my_df[column].mode()
                # Проверка того, что мода не NaN
                if not moda.empty and pd.notna(moda[0]):
                    my_df[column] = my_df[column].fillna(moda[0])

    update()
    # Вывод столбцов, где что-то не получилось
    if skipped_columns:
        columns_str = ", ".join(skipped_columns)
        messagebox.showwarning("Пропуски",
                               f"Все возможные пропуски заполнены\nСледующие столбцы остались пустыми(в них нет данных для расчета): {columns_str}")
    else:
        messagebox.showinfo("Ок", "Все пропуски заполнены")


def on_tree_select(event):
    # (event) означает, что в функции описаны действия или события,
    # которые происходят во время выполнения программы.
    # Простыми словами, это сигналы о том, что мы кликнули на строку
    global selected_now
    selection = column_tree.selection()
    if not selection:
        return

    # Достаем имя столбца из первой колонки выбранной строчки в treeview
    selected_now = column_tree.item(selection[0])['values'][0]
    show_plot()


def show_plot():
    # Функция для отображения графика. Гистограмма для чисел и Столбчатый график для категорий
    if my_df is None or not selected_now:
        return

    # очищение графика
    ax.clear()

    try:
        data = my_df[selected_now].dropna()
        if data.empty:
            ax.text(0.5, 0.5, "Нет данных для отображения")
        elif pd.api.types.is_numeric_dtype(my_df[selected_now]):
            # Отрисовка гистограммы для числовых данных
            ax.hist(data, bins='auto', color="#765a57")
            # Цвет "Блоха, упавшая в обморок"
            ax.set_title(f"Распределение {selected_now}")
            # Считаем значения медианы и среднего и добавляем их на график
            mean_ = data.mean()
            median_ = data.median()
            ax.axvline(mean_, color="red", linestyle="--", linewidth=2, label=f"Среднее: {mean_:.2f}")
            ax.axvline(median_, color="yellow", linestyle="-", linewidth=2, label=f"Медиана: {median_:.2f}")
            ax.legend()
        else:
            data.value_counts().head(10).plot(kind='bar', ax=ax, color="#00cccc")
            # Цвет "Яйца странствующего дрозда"
            ax.set_title(f"Топ 10 значений: {selected_now}")

    except Exception:
        ax.text(0.5, 0.5, "Ошибка отрисовки данных", ha="center", va="center")

    fig.tight_layout()  # Автоматически регулирует параметры подгафиков(чтобы легенда не мешала графику и другое)
    schedule.draw()


def update():
    # Сначала всегда очищаем таблицу от старых записей
    for i in column_tree.get_children():
        column_tree.delete(i)

    # Если данных нет, то просто выходим
    if my_df is None:
        status_label.config(text="Файл не загружен")
        return

    # Заполняем заново
    for column in my_df.columns:
        NaNs = my_df[column].isnull().sum()
        dtype = str(my_df[column].dtype)
        # Если есть пропуски, помечаем строку тегом 'with_NaN' для того, чтобы она подсвечивалась красным
        if NaNs > 0:
            indicator_for_nans = ('with_NaN')
        else:
            indicator_for_nans = ()
        column_tree.insert("", tk.END,
                           values=(column, dtype, NaNs), tags=indicator_for_nans)

    # обновление статуса(который отображается под названием)
    all_na_ns = my_df.isnull().sum().sum()
    rows, columns = my_df.shape
    status_text = f"Строк: {rows}; Колонок: {columns}; Всего пропусков: {all_na_ns}"
    status_label.config(text=status_text)


def save_state():
    # Обновление истории
    if my_df is not None:
        # copy нужен чтобы мы не сохраняли ссылку на тот же объект
        history.append(my_df.copy())


def closing():
    try:
        plt.close("all")
        root.quit()  # Мягко останавливает mainloop
        root.destroy()
    except Exception:
        pass  # Игнор ошибок, если окно уже закрыто


# ИНТЕРФЕЙС
root = tk.Tk()
root.title("Обработчик табличных данных")
root.geometry("1200x850")

# Настройка стилей
style = ttk.Style()
style.theme_use('clam')
style.configure('Green.TButton', background='#3caa3c', foreground='white', font=('Arial', 12, 'bold'))
# Цвет "Влюбленная жаба"

# Создание верхней рамки
# Создание фрейма где вообще лежат кнопки. Это нужно, чтобы их ограничить
top_frame = ttk.Frame(root, padding=15)  # padding это отступ
top_frame.pack(fill="x")  # растягивание по всему окну

# Добавление внутрь фрейма текста
label = ttk.Label(top_frame, text="Обработчик табличных данных")
label.pack(side="left")  # прижимается к левой стороне
# Добавление внутрь фрейма всех кнопок
ttk.Button(top_frame, text="💾Сохранить", command=save_file).pack(side="right", padx=5)
ttk.Button(top_frame, text="📂Открыть", command=load_file).pack(side="right", padx=5)
ttk.Button(top_frame, text="↩Отменить", command=undo_logic).pack(side="right", padx=5)
ttk.Button(top_frame, text="❓", width=3, command=show_help).pack(side="right", padx=5)
ttk.Button(top_frame, text="⚙Настройки", command=open_settings).pack(side="right", padx=5)

status_label = ttk.Label(root, text="Файл не загружен")
status_label.pack(anchor='w')
# anchor: устанавливает позиционирование текста

main_pw = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
main_pw.pack(fill="both", expand=True)  # только эти 2 делают так, чтобы оно прилипало и двигалось так, как окно

# Создание левой панели
left_container = ttk.Frame(main_pw)
# weight=1 означает как 1:3 с контейнером другим
main_pw.add(left_container, weight=1)

column_tree = ttk.Treeview(left_container, columns=("Name", "Type", "Miss"), show="headings")
column_tree.heading("Name", text='Столбец')
column_tree.heading("Type", text='Type')
column_tree.heading("Miss", text='Пропуски')
column_tree.column("Name", width=120)
column_tree.column("Type", width=80)
column_tree.column("Miss", width=80)
column_tree.pack(fill="both", expand=True)

column_tree.bind("<<TreeviewSelect>>", on_tree_select)
column_tree.tag_configure('with_NaN', background="#ed4830")

# Создание правой панели
right_container = ttk.Frame(main_pw)
# weight=3 означает как 1:3 с контейнером другим
main_pw.add(right_container, weight=3)

schedule_frame = ttk.LabelFrame(right_container, text="Анализ распределения", padding=10)
schedule_frame.pack(fill="both", expand=True)

# график
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
schedule = FigureCanvasTkAgg(fig, master=schedule_frame)
schedule.get_tk_widget().pack(fill="both", expand=True)

action_frame = ttk.LabelFrame(right_container, text="Инструменты обработки", padding=10)
action_frame.pack(fill="x", expand=True)

ttk.Label(action_frame, text="Метод:").grid(row=0, column=0, sticky='w', padx=5)
# методы

values_of_methods = ["Медиана", "Среднее", "Мода", "Константа", "Удалить"]
method_combo = ttk.Combobox(action_frame, values=values_of_methods, state="readonly", width=20)
method_combo.grid(row=0, column=1, sticky='w', padx=5)

ttk.Button(action_frame, text="Применить к выбранному", style='Green.TButton', command=apply_clean).grid(row=0,
                                                                                                         column=2,
                                                                                                         padx=10,
                                                                                                         sticky="e")  # sticky="e" прижимает вправо
action_frame.columnconfigure(2, weight=1)
# условно колонка 1 - метод, колонка 2 - выпадающий список, колонка 3 - кнопка применить, а так кнопка и все что справа сдвигается вправо, как на пружине

# авто обработка
ttk.Separator(action_frame, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky='ew', columnspan=5)
ttk.Label(action_frame, text="Массовые действия").grid(row=2, column=0, sticky='w', padx=5)
ttk.Button(action_frame, text="Заполнить всё", command=fill_all_with_logica).grid(row=2, column=1, padx=5)

# делает так, чтобы закрытие шло через функцию
root.protocol("WM_DELETE_WINDOW", closing)
root.mainloop()
