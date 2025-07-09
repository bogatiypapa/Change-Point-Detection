import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from ruptures import Pelt, Binseg, Dynp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from prophet import Prophet
import json
import os
import warnings
from scipy import stats
import random

warnings.filterwarnings('ignore')

# Настройки стиля
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'Times New Roman',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold'
})


class ChangePointApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализ точек изменения")

        # Инициализация данных
        self.x_data = None
        self.y_noisy_data = None
        self.y_clean_data = None

        # Инициализация методов
        self.methods = {}
        self.selected_methods = {}
        self.results = {}

        # Создаем виджеты
        self.create_widgets()

        # Инициализация методов после их определения
        self.methods = {
            "CUSUM": self.detect_cusum,
            "Pettitt": self.detect_pettitt,
            "BOCPD": self.detect_bayesian,
            "PELT": self.detect_pelt,
            "BinSeg": self.detect_binseg,
            "E-Divisive": self.detect_edivisive,
            #"BOCPD": self.detect_bocpd,
            "MOSUM": self.detect_mosum,
            "LSTM": self.detect_lstm,
            "CNN": self.detect_cnn,
            "Prophet": self.detect_prophet,
            "RFA": self.detect_rfa,
            "Dynamic": self.detect_dynp,
            "Ансамбль": self.detect_ensemble
        }

        # Обновляем выбор методов после инициализации
        self.selected_methods = {name: tk.BooleanVar(value=True) for name in self.methods}
        self.update_methods_selection()

    def create_widgets(self):
        # Главный Notebook
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(expand=True, fill=tk.BOTH)

        # Вкладка настройки данных
        self.create_data_settings_tab()

        # Вкладка сравнения методов
        self.comparison_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.comparison_frame, text="Сравнение методов")

    def create_data_settings_tab(self):
        """Создает вкладку для настройки данных"""
        data_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(data_frame, text="Настройка данных")

        # Создаем внешний Notebook для разделения данных и результатов
        outer_notebook = ttk.Notebook(data_frame)
        outer_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Вкладка 1: Параметры данных и управления
        settings_frame = ttk.Frame(outer_notebook)
        outer_notebook.add(settings_frame, text="Управление данными")

        # Вкладка 2: Результаты методов
        results_frame = ttk.Frame(outer_notebook)
        outer_notebook.add(results_frame, text="Результаты анализа")

        # Создаем содержимое для вкладки параметров
        self.create_settings_content(settings_frame)

        # Создаем область для результатов внутри вкладки результатов
        self.create_results_area(results_frame)



    def create_results_area(self, parent_frame):
        """Создает область для отображения результатов методов"""
        # Notebook для результатов методов
        self.method_notebook = ttk.Notebook(parent_frame)
        self.method_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Заглушка при запуске
        placeholder = ttk.Frame(self.method_notebook)
        ttk.Label(placeholder, text="Результаты методов появятся здесь после построения графиков").pack(padx=20,
                                                                                                        pady=20)
        self.method_notebook.add(placeholder, text="Результаты")

    def create_settings_content(self, parent_frame):
        """Создает содержимое вкладки параметров"""
        # Панель управления
        control_frame = ttk.Frame(parent_frame, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Левая часть: параметры данных
        data_params_frame = ttk.LabelFrame(control_frame, text="Параметры данных")
        data_params_frame.grid(row=0, column=0, padx=5, pady=5, sticky=tk.N)

        # Поля для состояния 1
        ttk.Label(data_params_frame, text="Количество точек (Состояние 1):").grid(row=0, column=0, sticky=tk.W)
        self.entry_points_state1 = ttk.Entry(data_params_frame, width=10)
        self.entry_points_state1.grid(row=0, column=1, padx=5, pady=2)
        self.entry_points_state1.insert(0, "20")

        # Поля для состояния 2
        ttk.Label(data_params_frame, text="Количество точек (Состояние 2):").grid(row=1, column=0, sticky=tk.W)
        self.entry_points_state2 = ttk.Entry(data_params_frame, width=10)
        self.entry_points_state2.grid(row=1, column=1, padx=5, pady=2)
        self.entry_points_state2.insert(0, "20")

        # Поля для состояния 3
        ttk.Label(data_params_frame, text="Количество точек (Состояние 3):").grid(row=2, column=0, sticky=tk.W)
        self.entry_points_state3 = ttk.Entry(data_params_frame, width=10)
        self.entry_points_state3.grid(row=2, column=1, padx=5, pady=2)
        self.entry_points_state3.insert(0, "20")

        # Угол наклона для состояния 2
        ttk.Label(data_params_frame, text="Угол наклона (Состояние 2):").grid(row=3, column=0, sticky=tk.W)
        self.entry_angle_state2 = ttk.Entry(data_params_frame, width=10)
        self.entry_angle_state2.grid(row=3, column=1, padx=5, pady=2)
        self.entry_angle_state2.insert(0, "45")

        # Уровень шума (в процентах)
        ttk.Label(data_params_frame, text="Уровень шума (%):").grid(row=4, column=0, sticky=tk.W)
        self.entry_noise_percent = ttk.Entry(data_params_frame, width=10)
        self.entry_noise_percent.grid(row=4, column=1, padx=5, pady=2)
        self.entry_noise_percent.insert(0, "10")

        # Правая часть: выбор методов
        self.methods_frame = ttk.LabelFrame(control_frame, text="Выбор методов")
        self.methods_frame.grid(row=0, column=1, padx=5, pady=5, sticky=tk.N)

        # Обновляем выбор методов
        self.update_methods_selection()

        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)

        ttk.Button(button_frame, text="Построить графики", command=self.generate_and_plot_all).pack(side=tk.LEFT,
                                                                                                    padx=5)
        ttk.Button(button_frame, text="Сохранить данные", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Загрузить данные", command=self.load_settings).pack(side=tk.LEFT, padx=5)
        #ttk.Button(button_frame, text="Построить и сравнить", command=self.build_and_compare).pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Построить график зависимостей", command=self.run_noise_analysis).pack(side=tk.LEFT, padx=5)


        # Область для графиков данных
        data_graph_frame = ttk.Frame(parent_frame)
        data_graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.data_fig, (self.data_ax1, self.data_ax2) = plt.subplots(2, 1, figsize=(10, 6))
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=data_graph_frame)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def update_methods_selection(self):
        """Обновляет выбор методов после их инициализации"""
        # Очищаем предыдущий фрейм выбора методов
        for widget in self.methods_frame.winfo_children():
            widget.destroy()

        # Создаем чекбоксы для выбора методов в 2 столбца
        methods_list = list(self.methods.keys())
        half = len(methods_list) // 2 + len(methods_list) % 2

        for i, method in enumerate(methods_list):
            col = i // half
            row = i % half
            cb = ttk.Checkbutton(
                self.methods_frame,
                text=method,
                variable=self.selected_methods[method],
                onvalue=True,
                offvalue=False
            )
            cb.grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)

    def add_noise(self, y, noise_level_percent, y_range):
        noise_level = (noise_level_percent / 100) * y_range
        noise = np.random.uniform(-noise_level, noise_level, size=len(y))
        return y + noise

    def moving_average(self, y, window_size=5):
        return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

    def generate_and_plot_all(self):
        """Генерирует данные и применяет выбранные методы"""
        try:
            # Генерация данных
            num_points_state1 = int(self.entry_points_state1.get())
            num_points_state2 = int(self.entry_points_state2.get())
            num_points_state3 = int(self.entry_points_state3.get())
            angle_state2 = float(self.entry_angle_state2.get())
            noise_level_percent = float(self.entry_noise_percent.get())

            # Генерация данных
            x_state1 = np.linspace(0, num_points_state1 - 1, num_points_state1)
            y_state1 = np.zeros(num_points_state1)

            if abs(angle_state2 - 90) < 1:
                x_state2 = np.full(num_points_state2, x_state1[-1])
                y_state2 = np.linspace(y_state1[-1], y_state1[-1] + num_points_state2, num_points_state2)
            else:
                x_state2 = np.linspace(num_points_state1, num_points_state1 + num_points_state2 - 1, num_points_state2)
                y_state2 = np.tan(np.radians(angle_state2)) * (x_state2 - x_state2[0]) + y_state1[-1]

            x_state3 = np.linspace(x_state2[-1], x_state2[-1] + num_points_state3 - 1, num_points_state3)
            y_state3 = np.full(num_points_state3, y_state2[-1])

            self.x_data = np.concatenate((x_state1, x_state2, x_state3))
            y = np.concatenate((y_state1, y_state2, y_state3))
            self.y_clean_data = y

            y_range = np.max(y) - np.min(y)
            self.y_noisy_data = self.add_noise(y, noise_level_percent, y_range)

            # Очистка и построение графиков данных
            self.data_ax1.clear()
            self.data_ax2.clear()

            self.data_ax1.plot(self.x_data, self.y_clean_data, 'b-', label='Чистые данные')
            self.data_ax1.set_title('График без шума')
            self.data_ax1.legend()
            self.data_ax1.grid(True)

            self.data_ax2.plot(self.x_data, self.y_noisy_data, 'r-', label='Данные с шумом')
            self.data_ax2.set_title('График с шумом')
            self.data_ax2.legend()
            self.data_ax2.grid(True)

            self.data_canvas.draw()

            # Применение выбранных методов
            self.apply_selected_methods()

            # Показ результатов
            self.show_selected_methods()
            self.show_comparison()

            # ОБНОВЛЯЕМ ГЛАВНОЕ ОКНО
            self.root.update_idletasks()

        except Exception as e:
            print(f"Ошибка: {e}")

    def update_method_plots(self, method_name):
        """Обновляет графики метода"""
        self.method_ax1.clear()
        self.method_ax2.clear()

        # График с шумом
        self.method_ax1.plot(self.x_data, self.y_noisy_data, 'r-', label='Данные с шумом')

        # Добавляем истинные точки изменения
        real_cp1 = int(self.entry_points_state1.get()) - 1
        real_cp2 = real_cp1 + int(self.entry_points_state2.get())

        self.method_ax1.axvline(self.x_data[real_cp1], color='g', linestyle='-', linewidth=1,
                                label='Истинная точка изменения 1')
        self.method_ax1.axvline(self.x_data[real_cp2], color='g', linestyle='-', linewidth=1,
                                label='Истинная точка изменения 2')

        self.method_ax1.set_title(f'{method_name} - Исходные данные с шумом')
        self.method_ax1.legend()
        self.method_ax1.grid(True)

        # График с обработанными данными
        self.method_ax2.plot(self.x_data, self.y_noisy_data, 'r-', alpha=0.3, label='Данные с шумом')

        # Добавляем истинные точки изменения
        self.method_ax2.axvline(self.x_data[real_cp1], color='g', linestyle='-', linewidth=1,
                                label='Истинная точка изменения')
        self.method_ax2.axvline(self.x_data[real_cp2], color='g', linestyle='-', linewidth=1)

        # Добавляем точки изменения метода
        change_points = self.results[method_name]
        for cp in change_points:
            if cp < len(self.x_data):
                self.method_ax2.axvline(self.x_data[cp], color='b', linestyle='--', alpha=0.7, linewidth=1,
                                        label='Обнаруженная точка изменения')

        # Убираем дубликаты в легенде
        handles, labels = self.method_ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.method_ax2.legend(by_label.values(), by_label.keys())

        self.method_ax2.set_title(f'{method_name} - Обнаруженные точки изменения')
        self.method_ax2.grid(True)

        self.method_canvas.draw()

    def start_streaming(self, method_name):
        """Запускает потоковое воспроизведение данных с обнаружением точек изменения"""
        if self.streaming_active:
            self.streaming_active = False
            return

        self.streaming_active = True

        # Создаем новое окно для потокового воспроизведения
        stream_window = tk.Toplevel(self.root)
        stream_window.title(f"Потоковое воспроизведение: {method_name}")
        stream_window.geometry("800x600")

        # Создаем график для потокового отображения
        fig, ax = plt.subplots(figsize=(10, 6))
        canvas = FigureCanvasTkAgg(fig, master=stream_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Кнопка остановки
        stop_button = ttk.Button(
            stream_window,
            text="Остановить воспроизведение",
            command=lambda: self.stop_streaming(stream_window)
        )
        stop_button.pack(pady=10)

        # Инициализация данных для потокового воспроизведения
        x_stream = []
        y_stream = []
        detected_points = []

        # Получаем истинные точки изменения
        real_cp1 = int(self.entry_points_state1.get()) - 1
        real_cp2 = real_cp1 + int(self.entry_points_state2.get())

        # Получаем все точки изменения для этого метода
        all_change_points = sorted(self.results[method_name])

        def update_stream():
            if not self.streaming_active:
                return

            # Добавляем новую точку данных
            if len(x_stream) < len(self.x_data):
                idx = len(x_stream)
                x_stream.append(self.x_data[idx])
                y_stream.append(self.y_noisy_data[idx])

                # Проверяем, является ли текущая точка точкой изменения
                if idx in all_change_points:
                    detected_points.append(idx)

                # Очищаем и перерисовываем график
                ax.clear()

                # Рисуем данные
                ax.plot(x_stream, y_stream, 'b-', label='Данные')

                # Добавляем истинные точки изменения
                if idx >= real_cp1:
                    ax.axvline(self.x_data[real_cp1], color='g', linestyle='-', linewidth=1,
                               label='Истинная точка изменения 1')
                if idx >= real_cp2:
                    ax.axvline(self.x_data[real_cp2], color='g', linestyle='-', linewidth=1,
                               label='Истинная точка изменения 2')

                # Рисуем обнаруженные точки изменения
                for cp in detected_points:
                    if cp < len(x_stream):
                        ax.axvline(x_stream[cp], color='r', linestyle='--', label='Обнаруженная точка изменения')

                # Добавляем легенду только один раз
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys())

                ax.set_title(f"{method_name} - Потоковое воспроизведение")
                ax.grid(True)
                canvas.draw()

                # Запланировать следующее обновление с задержкой 0.3 секунды
                stream_window.after(300, update_stream)
            else:
                # Воспроизведение завершено
                self.streaming_active = False
                ttk.Label(stream_window, text="Воспроизведение завершено").pack()

        # Начинаем воспроизведение
        update_stream()

    def stop_streaming(self, window):
        """Останавливает потоковое воспроизведение и закрывает окно"""
        self.streaming_active = False
        window.destroy()




    def run_noise_analysis(self):
        """Анализ задержки обнаружения (delta T) для 1-30% шума"""
        try:
            original_noise = float(self.entry_noise_percent.get())
            noise_levels = list(range(1, 31))  # 1%..30%
            results = {method: [] for method in self.methods if self.selected_methods[method].get()}

            # Истинные точки изменения
            true_cp1 = int(self.entry_points_state1.get()) - 1
            true_cp2 = true_cp1 + int(self.entry_points_state2.get())

            # Прогресс-бар
            progress_window = tk.Toplevel(self.root)
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=len(noise_levels))
            progress_bar.pack(padx=10, pady=10)
            status_label = ttk.Label(progress_window, text="Анализ...")
            status_label.pack()

            for noise_percent in noise_levels:
                self.entry_noise_percent.delete(0, tk.END)
                self.entry_noise_percent.insert(0, str(noise_percent))
                self.generate_and_plot_all()

                for method in results:
                    points = sorted(self.results.get(method, []))
                    delta_t = None

                    # Ищем первую обнаруженную точку в окрестностях true_cp1 или true_cp2
                    for cp in points:
                        if (true_cp1 - 5 <= cp <= true_cp1 + 5) or (true_cp2 - 5 <= cp <= true_cp2 + 5):
                            delta_t = cp - (true_cp1 if cp <= true_cp1 + 5 else true_cp2)
                            break

                    results[method].append(delta_t if delta_t is not None else np.nan)

                progress_var.set(noise_percent)
                progress_window.update()

            # Восстановление исходных данных
            self.entry_noise_percent.delete(0, tk.END)
            self.entry_noise_percent.insert(0, str(original_noise))
            self.generate_and_plot_all()
            progress_window.destroy()

            self.show_noise_results(noise_levels, results, true_cp1, true_cp2)

        except Exception as e:
            print(f"Ошибка: {e}")

    def show_noise_results(self, noise_levels, results, true_cp1, true_cp2):
        """График зависимости delta T от уровня шума"""
        result_window = tk.Toplevel(self.root)
        result_window.title("Задержка обнаружения (ΔT)")
        result_window.geometry("1000x800")

        fig, ax = plt.subplots(figsize=(10, 6))

        for method, deltas in results.items():
            ax.plot(noise_levels, deltas, 'o-', label=method)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5, label="Идеальное обнаружение")
        ax.set_xlabel("Уровень шума (%)")
        ax.set_ylabel("ΔT")
        ax.set_title(f"Задержка обнаружения")
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        ttk.Button(
            result_window,
            text="Сохранить данные",
            command=lambda: self.save_noise_results(noise_levels, results)
        ).pack(pady=10)

    def show_noise_analysis_results(self, noise_levels, results):
        """Отображает все результаты на одном графике"""
        # Создаем новое окно для результатов
        result_window = tk.Toplevel(self.root)
        result_window.title("Noise Analysis Results (1-30%)")
        result_window.geometry("1000x800")

        # Создаем график
        fig, ax = plt.subplots(figsize=(10, 6))

        # Для каждого метода строим линию на графике
        for method, avg_lags in results.items():
            # Заменяем None на NaN для корректного отображения
            clean_lags = [lag if lag is not None else np.nan for lag in avg_lags]
            ax.plot(noise_levels, clean_lags, 'o-', label=method)

        ax.set_xlabel('Noise Level (%)')
        ax.set_ylabel('Average Detection Lag (points)')
        ax.set_title('Dependence of Detection Lag on Noise Level')
        ax.legend()
        ax.grid(True)

        # Добавляем пояснения
        ax.text(0.5, 0.95,
                "Lower values are better (less lag in detection)",
                transform=ax.transAxes, ha='center', va='top',
                bbox=dict(facecolor='white', alpha=0.8))

        # Встраиваем график в окно
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Кнопка сохранения результатов
        save_button = ttk.Button(
            result_window,
            text="Save Results",
            command=lambda: self.save_noise_results(noise_levels, results)
        )
        save_button.pack(pady=10)

    def save_noise_results(self, noise_levels, results):
        """Сохраняет результаты анализа шума в файл"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Noise Analysis Results"
        )

        if file_path:
            save_data = {
                'parameters': {
                    'points_state1': int(self.entry_points_state1.get()),
                    'points_state2': int(self.entry_points_state2.get()),
                    'points_state3': int(self.entry_points_state3.get()),
                    'angle_state2': float(self.entry_angle_state2.get())
                },
                'noise_levels': noise_levels,
                'results': results
            }

            with open(file_path, 'w') as f:
                json.dump(save_data, f, indent=4)


    def apply_selected_methods(self):
        """Применяет выбранные методы к данным"""
        if self.x_data is None or self.y_noisy_data is None:
            return

        # Очищаем предыдущие результаты
        self.results = {}

        # Применяем только выбранные методы
        for name in self.methods:
            if self.selected_methods[name].get():
                try:
                    self.results[name] = self.methods[name]()
                    print(f"{name} завершен")
                except Exception as e:
                    print(f"Ошибка в {name}: {e}")
                    self.results[name] = []

    def show_selected_methods(self):
        """Создает вкладки для выбранных методов с результатами"""
        # Удаляем все существующие вкладки, кроме первой (заглушки)
        for tab_id in self.method_notebook.tabs()[1:]:
            self.method_notebook.forget(tab_id)

        # Удаляем содержимое первой вкладки (заглушки)
        for child in self.method_notebook.winfo_children():
            child.destroy()

        # Если нет результатов, добавляем заглушку
        if not self.results:
            placeholder = ttk.Label(self.method_notebook,
                                    text="Результаты методов появятся здесь после построения графиков")
            self.method_notebook.add(placeholder, text="Результаты")
            return

        # Создаем новые вкладки для каждого выбранного метода
        for method_name in self.results:
            # Создаем контейнер для вкладки
            tab_frame = ttk.Frame(self.method_notebook)
            self.method_notebook.add(tab_frame, text=method_name)

            # Создаем скроллируемую область
            canvas = tk.Canvas(tab_frame)
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)

            # Настройка скроллинга
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            # Упаковка элементов
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Создаем графики для метода
            self.create_method_tab_content(scrollable_frame, method_name)

        # Выбираем первую вкладку с результатами
        self.method_notebook.select(0)

    # Восстановленные методы
    def save_settings(self):
        """Сохраняет текущие настройки и данные в файл"""
        if self.x_data is None or self.y_noisy_data is None:
            return

        settings = {
            "num_points_state1": int(self.entry_points_state1.get()),
            "num_points_state2": int(self.entry_points_state2.get()),
            "num_points_state3": int(self.entry_points_state3.get()),
            "angle_state2": float(self.entry_angle_state2.get()),
            "noise_level_percent": float(self.entry_noise_percent.get()),
            "x_data": self.x_data.tolist(),
            "y_noisy_data": self.y_noisy_data.tolist(),
            "y_clean_data": self.y_clean_data.tolist(),
            "selected_methods": {name: var.get() for name, var in self.selected_methods.items()}
        }

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                json.dump(settings, f, indent=4)

    def load_settings(self):
        """Загружает настройки и данные из файла"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'r') as f:
                settings = json.load(f)

            # Загружаем настройки
            self.entry_points_state1.delete(0, tk.END)
            self.entry_points_state1.insert(0, str(settings["num_points_state1"]))
            self.entry_points_state2.delete(0, tk.END)
            self.entry_points_state2.insert(0, str(settings["num_points_state2"]))
            self.entry_points_state3.delete(0, tk.END)
            self.entry_points_state3.insert(0, str(settings["num_points_state3"]))
            self.entry_angle_state2.delete(0, tk.END)
            self.entry_angle_state2.insert(0, str(settings["angle_state2"]))
            self.entry_noise_percent.delete(0, tk.END)
            self.entry_noise_percent.insert(0, str(settings["noise_level_percent"]))

            # Загружаем данные
            self.x_data = np.array(settings["x_data"])
            self.y_noisy_data = np.array(settings["y_noisy_data"])
            self.y_clean_data = np.array(settings["y_clean_data"])

            # Обновляем выбор методов
            if "selected_methods" in settings:
                for name, value in settings["selected_methods"].items():
                    if name in self.selected_methods:
                        self.selected_methods[name].set(value)
                self.update_methods_selection()

            # Обновляем графики данных
            self.data_ax1.clear()
            self.data_ax2.clear()

            self.data_ax1.plot(self.x_data, self.y_clean_data, 'b-', label='Чистые данные')
            self.data_ax1.set_title('График без шума')
            self.data_ax1.legend()
            self.data_ax1.grid(True)

            self.data_ax2.plot(self.x_data, self.y_noisy_data, 'r-', label='Данные с шумом')
            self.data_ax2.set_title('График с шумом')
            self.data_ax2.legend()
            self.data_ax2.grid(True)

            self.data_canvas.draw()

            # Применяем методы к загруженным данным
            self.apply_selected_methods()
            self.show_selected_methods()
            self.show_comparison()

    def build_and_compare(self):
        """Строит и сравнивает методы для уровней шума от 1% до 15% с шагом 1%"""
        try:
            # Сохраняем исходные настройки
            original_noise = float(self.entry_noise_percent.get())

            # Создаем окно прогресса
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Выполнение сравнения")
            progress_var = tk.DoubleVar()
            progress_bar = ttk.Progressbar(progress_window, variable=progress_var, maximum=15)
            progress_bar.pack(padx=10, pady=10)
            status_label = ttk.Label(progress_window, text="Подготовка...")
            status_label.pack()
            progress_window.update()

            # Подготовка данных для сохранения
            comparison_results = []

            # Перебираем уровни шума от 1 до 15
            for noise_level in range(1, 16):
                status_label.config(text=f"Обработка шума {noise_level}%...")
                progress_var.set(noise_level)
                progress_window.update()

                try:
                    # Устанавливаем текущий уровень шума
                    self.entry_noise_percent.delete(0, tk.END)
                    self.entry_noise_percent.insert(0, str(noise_level))

                    # Генерируем данные и применяем методы
                    self.generate_and_plot_all()

                    # Получаем реальную точку изменения
                    real_cp = int(self.entry_points_state1.get()) - 1

                    # Собираем результаты для текущего уровня шума
                    noise_results = {
                        "noise_level": noise_level,
                        "methods": {}
                    }

                    for method, points in self.results.items():
                        correct_detections = [cp for cp in points if real_cp <= cp <= real_cp + 3]
                        false_detections = [cp for cp in points if cp < real_cp or cp > real_cp + 3]

                        precision = len(correct_detections) / len(points) if len(points) > 0 else 0
                        recall = 1 if len(correct_detections) > 0 else 0
                        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        lag = np.mean([cp - real_cp for cp in correct_detections]) if correct_detections else 0

                        noise_results["methods"][method] = {
                            "Точность": float(precision),
                            "Полнота": float(recall),
                            "f1_score": float(f1_score),
                            "Delta_t": float(lag),
                            "Ложные обнаружения": len(false_detections)
                        }

                    comparison_results.append(noise_results)

                except Exception as e:
                    print(f"Ошибка при обработке шума {noise_level}%: {e}")
                    continue

            # Восстанавливаем исходный уровень шума
            self.entry_noise_percent.delete(0, tk.END)
            self.entry_noise_percent.insert(0, str(original_noise))
            self.generate_and_plot_all()

            # Генерируем имя файла с диапазоном шума и случайным числом
            filename = f"noise_1-15-1_{random.randint(1000, 9999)}.json"

            # Сохраняем результаты в JSON файл
            file_path = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json")],
                title="Сохранить результаты сравнения",
                initialfile=filename
            )

            if file_path:
                with open(file_path, 'w') as f:
                    json.dump(comparison_results, f, indent=4)

                status_label.config(text=f"Результаты сохранены в:\n{file_path}")
                progress_window.after(3000, progress_window.destroy)
                print(f"Результаты сохранены в файл: {file_path}")
            else:
                progress_window.destroy()

        except Exception as e:
            print(f"Критическая ошибка: {e}")
            if 'progress_window' in locals():
                progress_window.destroy()

    # Методы обнаружения точек изменения
    def detect_cusum(self):
        """Реализация CUSUM метода"""
        n = len(self.y_noisy_data)
        mean = np.mean(self.y_noisy_data)
        std = np.std(self.y_noisy_data)

        # Накопленные суммы
        s = np.zeros(n)
        for t in range(1, n):
            s[t] = s[t - 1] + (self.y_noisy_data[t] - mean) / std

        # Квадраты накопленных сумм
        s2 = s ** 2

        # Нормализация
        s2_norm = (s2 - np.min(s2)) / (np.max(s2) - np.min(s2))

        # Порог для обнаружения
        threshold = 0.8
        changepoints = np.where(s2_norm > threshold)[0]

        # Фильтрация близких точек
        filtered_points = []
        prev_point = -10
        for point in changepoints:
            if point - prev_point > 5:
                filtered_points.append(point)
                prev_point = point

        return filtered_points

    def detect_pettitt(self):
        """Обнаружение одной точки изменения"""
        n = len(self.y_noisy_data)
        if n < 10: return []

        # Нормализуем данные
        y_norm = (self.y_noisy_data - np.mean(self.y_noisy_data)) / np.std(self.y_noisy_data)

        U = np.zeros(n)
        for t in range(1, n):
            U[t] = U[t - 1] + np.sign(y_norm[t] - y_norm[t - 1])

        K = np.max(np.abs(U))
        t0 = np.argmax(np.abs(U))

        # Автоматический порог
        threshold = 1.36 * np.sqrt(n)
        if K > threshold:
            return [t0]
        return []

    def detect_bayesian(self):
        """Обнаружение точек изменения с использованием байесовского подхода"""
        y = self.y_noisy_data
        n = len(y)
        if n < 10:
            return []

        # Параметры алгоритма
        window_size = max(5, n // 20)
        threshold = 0.9
        min_change = 0.5 * np.std(y)

        changepoints = []

        for i in range(window_size, n - window_size):
            # Данные до и после текущей точки
            before = y[i - window_size:i]
            after = y[i:i + window_size]

            # Параметры распределений
            mu_before, std_before = np.mean(before), np.std(before) + 1e-10
            mu_after, std_after = np.mean(after), np.std(after) + 1e-10
            mu_pooled = np.mean(np.concatenate([before, after]))
            std_pooled = np.std(np.concatenate([before, after])) + 1e-10

            # Вычисление логарифмических правдоподобий
            def log_likelihood(data, mu, sigma):
                return np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

            # Гипотеза H0: нет изменения
            log_p_h0 = log_likelihood(before, mu_pooled, std_pooled) + \
                       log_likelihood(after, mu_pooled, std_pooled)

            # Гипотеза H1: есть изменение
            log_p_h1 = log_likelihood(before, mu_before, std_before) + \
                       log_likelihood(after, mu_after, std_after)

            # Нормализация вероятностей
            max_log = max(log_p_h0, log_p_h1)
            log_p_h0 -= max_log
            log_p_h1 -= max_log

            p_h1 = np.exp(log_p_h1) / (np.exp(log_p_h0) + np.exp(log_p_h1))

            # Проверка условий
            if p_h1 > threshold and abs(mu_after - mu_before) > min_change:
                changepoints.append(i)

        # Фильтрация близких точек
        filtered_points = []
        prev_point = -window_size
        for point in sorted(changepoints):
            if point - prev_point >= window_size:
                filtered_points.append(point)
                prev_point = point

        return filtered_points

    def detect_pelt(self):
        """PELT метод для точного обнаружения точек изменения"""
        algo = Pelt(model="rbf", min_size=5).fit(self.y_noisy_data)
        return algo.predict(pen=3)

    def detect_binseg(self):
        """Binary Segmentation метод"""
        algo = Binseg(model="l2", min_size=5).fit(self.y_noisy_data)
        return algo.predict(n_bkps=5)

    def detect_edivisive(self):
        """E-Divisive метод для обнаружения изменений в распределении"""
        n = len(self.y_noisy_data)
        changepoints = []

        for i in range(10, n - 10):
            before = self.y_noisy_data[i - 10:i]
            after = self.y_noisy_data[i:i + 10]

            # Двухвыборочный тест Колмогорова-Смирнова
            stat, p_value = stats.ks_2samp(before, after)
            if p_value < 0.05:
                changepoints.append(i)

        return changepoints

    def detect_bocpd(self):
        """BOCPD метод"""
        y = self.y_noisy_data
        n = len(y)

        if n < 10:
            return []

        # Параметры
        hazard = 1 / 10
        threshold = 0.4
        min_run_length = 3

        changepoints = []
        run_length = 1
        mu = np.mean(y[:5])
        sigma = max(np.std(y[:5]), 0.5)

        for i in range(1, n):
            x = y[i]

            try:
                prob = stats.norm.pdf(x, mu, sigma + 1e-10)
            except:
                prob = 1e-10

            # Байесовское обновление
            prob_change = hazard * prob
            prob_nochange = (1 - hazard) * prob
            total = prob_change + prob_nochange

            if total > 0:
                prob_change /= total
                if prob_change > threshold and run_length >= min_run_length:
                    changepoints.append(i)
                    run_length = 1
                    mu = x
                    sigma = max(np.std(y[max(0, i - 5):i + 1]), 0.5)
                else:
                    run_length += 1
                    alpha = 1.0 / run_length
                    mu = (1 - alpha) * mu + alpha * x
                    sigma = np.sqrt((1 - alpha) * sigma ** 2 + alpha * (x - mu) ** 2)

        # Фильтрация результатов
        final_points = []
        prev = -min_run_length
        for cp in changepoints:
            if cp - prev >= min_run_length:
                final_points.append(cp)
                prev = cp

        return final_points

    def detect_mosum(self):
        """Обнаружение изменений среднего"""
        y = self.y_noisy_data
        n = len(y)
        window = 10
        threshold = 2.0

        mosum = []
        for i in range(window, n - window):
            left = y[i - window:i]
            right = y[i:i + window]
            stat = np.abs(np.mean(left) - np.mean(right))
            mosum.append(stat)

        # Нормализуем статистику
        mosum_norm = (mosum - np.mean(mosum)) / np.std(mosum)
        changepoints = np.where(mosum_norm > threshold)[0] + window

        return list(set(changepoints))

    def detect_lstm(self):
        """LSTM метод для обнаружения сложных изменений"""
        n_features = 1
        n_lookback = 10

        # Подготовка данных
        X, y = [], []
        for i in range(n_lookback, len(self.y_noisy_data)):
            X.append(self.y_noisy_data[i - n_lookback:i].reshape(-1, 1))
            y.append(self.y_noisy_data[i])

        X, y = np.array(X), np.array(y)

        # Архитектура модели LSTM
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(n_lookback, n_features)),
            Dropout(0.2),
            LSTM(64),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        # Прогнозирование и определение аномалий
        predictions = model.predict(X)
        errors = np.abs(predictions.flatten() - y)

        # Определение порога для аномалий
        threshold = np.percentile(errors, 90)
        anomaly_indices = np.where(errors > threshold)[0] + n_lookback

        # Фильтрация точек
        filtered_indices = []
        prev_idx = -100
        for idx in anomaly_indices:
            if idx - prev_idx >= 5:
                filtered_indices.append(idx)
                prev_idx = idx

        return filtered_indices

    def detect_cnn(self):
        """CNN метод для обнаружения локальных паттернов"""
        n_features = 1
        n_lookback = 10

        # Подготовка данных
        X, y = [], []
        for i in range(n_lookback, len(self.y_noisy_data)):
            X.append(self.y_noisy_data[i - n_lookback:i].reshape(-1, 1))
            y.append(self.y_noisy_data[i])

        X, y = np.array(X), np.array(y)

        # Архитектура модели CNN
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_lookback, n_features)),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(0.001), loss='mse')
        model.fit(X, y, epochs=50, batch_size=32, verbose=0)

        # Прогнозирование и определение аномалий
        predictions = model.predict(X)
        errors = np.abs(predictions.flatten() - y)

        # Определение порога для аномалий
        threshold = np.percentile(errors, 90)
        anomaly_indices = np.where(errors > threshold)[0] + n_lookback

        # Фильтрация точек
        filtered_indices = []
        prev_idx = -100
        for idx in anomaly_indices:
            if idx - prev_idx >= 5:
                filtered_indices.append(idx)
                prev_idx = idx

        return filtered_indices

    def detect_prophet(self):
        """Упрощенный аналог Prophet"""
        n = len(self.y_noisy_data)
        changepoints = []
        min_segment = 10

        # Ищем изменения тренда
        for i in range(min_segment, n - min_segment):
            left = self.y_noisy_data[:i]
            right = self.y_noisy_data[i:]

            # Линейная регрессия для обоих сегментов
            x_left = np.arange(len(left))
            slope_left = np.polyfit(x_left, left, 1)[0]

            x_right = np.arange(len(right))
            slope_right = np.polyfit(x_right, right, 1)[0]

            # Если тренды значительно отличаются
            if abs(slope_left - slope_right) > 0.5:
                changepoints.append(i)

        return changepoints

    def detect_rfa(self):
        """Rolling Fourier Analysis метод"""
        window_size = 20
        threshold = 0.5

        n = len(self.y_noisy_data)
        changepoints = []

        for i in range(window_size, n - window_size):
            # Берем два соседних окна
            window1 = self.y_noisy_data[i - window_size:i]
            window2 = self.y_noisy_data[i:i + window_size]

            # Вычисляем спектры Фурье
            fft1 = np.abs(np.fft.fft(window1 - np.mean(window1)))
            fft2 = np.abs(np.fft.fft(window2 - np.mean(window2)))

            # Сравниваем спектры
            diff = np.mean(np.abs(fft1 - fft2))

            if diff > threshold:
                changepoints.append(i)

        return changepoints

    def detect_dynp(self):
        """Dynamic Programming метод"""
        algo = Dynp(model="l1", min_size=5).fit(self.y_noisy_data)
        return algo.predict(n_bkps=5)

    def detect_ensemble(self):
        """Ансамблевый метод, объединяющий несколько подходов"""
        methods_to_combine = ["PELT", "BinSeg", "Dynamic", "LSTM"]
        all_points = []

        for method in methods_to_combine:
            all_points.extend(self.results[method])

        unique_points = sorted(list(set(all_points)))

        # Точка считается значимой, если ее обнаружили хотя бы 2 метода
        consensus_points = []
        for point in unique_points:
            count = sum(1 for method in methods_to_combine if point in self.results[method])
            if count >= 2:
                consensus_points.append(point)

        return consensus_points

    def create_method_tab_content(self, frame, method_name):
        """Создает содержимое вкладки метода (2 графика) с кнопкой потокового воспроизведения"""
        # Создаем контейнер для управления размерами
        container = ttk.Frame(frame)
        container.pack(fill=tk.BOTH, expand=True)

        # Верхняя панель с кнопкой
        control_frame = ttk.Frame(container)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # Кнопка потокового воспроизведения
        self.streaming_active = False
        ttk.Button(
            control_frame,
            text="Запустить потоковое воспроизведение",
            command=lambda: self.start_streaming(method_name)
        ).pack(side=tk.LEFT, padx=5)

        # Используем Figure вместо plt.subplots
        fig = plt.Figure(figsize=(10, 8))
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2)

        # График с шумом
        ax1.plot(self.x_data, self.y_noisy_data, 'r-', label='Данные с шумом')

        # Добавляем истинные точки изменения
        real_cp1 = int(self.entry_points_state1.get()) - 1
        real_cp2 = real_cp1 + int(self.entry_points_state2.get())

        ax1.axvline(self.x_data[real_cp1], color='g', linestyle='-', linewidth=1, label='Истинная точка изменения 1')
        ax1.axvline(self.x_data[real_cp2], color='g', linestyle='-', linewidth=1, label='Истинная точка изменения 2')

        ax1.set_title(f'{method_name} - Исходные данные с шумом')
        ax1.legend()
        ax1.grid(True)

        # График с обработанными данными
        ax2.plot(self.x_data, self.y_noisy_data, 'r-', alpha=0.3, label='Данные с шумом')

        # Добавляем истинные точки изменения
        ax2.axvline(self.x_data[real_cp1], color='g', linestyle='-', linewidth=1, label='Истинная точка изменения')
        ax2.axvline(self.x_data[real_cp2], color='g', linestyle='-', linewidth=1)

        # Добавляем точки изменения метода
        change_points = self.results[method_name]
        for cp in change_points:
            if cp < len(self.x_data):
                ax2.axvline(self.x_data[cp], color='b', linestyle='--', alpha=0.7, linewidth=1,
                            label='Обнаруженная точка изменения')

        # Убираем дубликаты в легенде
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys())

        ax2.set_title(f'{method_name} - Обнаруженные точки изменения')
        ax2.grid(True)

        # Встраиваем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_comparison(self):
        """Обновляет вкладку сравнения методов с акцентом на ΔT"""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        # Реальные точки изменения
        real_cp1 = int(self.entry_points_state1.get()) - 1
        real_cp2 = real_cp1 + int(self.entry_points_state2.get())

        # Создаем таблицу сравнения
        columns = ("Метод", "ΔT", "Правильные", "Ложные")
        tree = ttk.Treeview(self.comparison_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')

        # Подготовка данных для графика
        plot_data = []

        # Заполняем таблицу данными
        for method, points in self.results.items():
            if not points:
                tree.insert("", "end", values=(method, "N/A", 0, 0))
                continue

            # Находим первую обнаруженную точку
            first_point = min(points)

            # Вычисляем ΔT относительно первой истинной точки
            delta_t = first_point - real_cp1

            # Классификация всех точек
            correct_points = []
            false_points = []

            for point in points:
                if point < real_cp1:  # Все точки до первой истинной - ложные
                    false_points.append(point)
                elif real_cp1 <= point <= real_cp2:  # В диапазоне - правильные
                    correct_points.append(point)
                elif point > real_cp2:  # После второй истинной
                    if abs(point - real_cp2) <= 5:  # Близко ко второй точке
                        correct_points.append(point)
                    else:
                        false_points.append(point)

            # Статус для цвета на графике
            point_status = "correct" if (real_cp1 <= first_point <= real_cp2) or (
                    first_point > real_cp2 and abs(first_point - real_cp2) <= 5) else "false"

            plot_data.append({
                'method': method,
                'first_point': first_point,
                'delta_t': delta_t,
                'point_status': point_status,
                'correct_count': len(correct_points),
                'false_count': len(false_points)
            })

            tree.insert("", "end", values=(
                method,
                delta_t,
                len(correct_points),
                len(false_points)
            ))

        tree.pack(fill=tk.BOTH, expand=True)

        # График сравнения
        fig = plt.Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Вертикальные линии и зона
        ax.axvline(real_cp1, color='green', linestyle='-', linewidth=2, label='Истинные точки')
        ax.axvline(real_cp2, color='green', linestyle='-', linewidth=2)
        ax.axvspan(real_cp1, real_cp2, facecolor='lightgreen', alpha=0.2, label='Правильный диапазон')

        # Добавляем данные методов
        for data in plot_data:
            color = 'blue' if data['point_status'] == "correct" else 'red'

            # Отображаем первую точку с подписью ΔT
            ax.scatter(data['first_point'], data['method'], color=color, s=100)
            ax.text(data['first_point'], data['method'],
                    f"ΔT={data['delta_t']}",
                    va='center',
                    ha='left' if data['delta_t'] >= 0 else 'right',
                    fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

        ax.set_xlabel('Время (сек)')
        ax.set_ylabel('Метод')
        ax.set_title('Сравнение методов по ΔT (относительно первой истинной точки)')

        # Легенда
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgreen', label='Правильный диапазон', alpha=0.2),
            Patch(facecolor='green', label='Истинная точка'),
            Patch(facecolor='blue', label='Правильное обнаружение'),
            Patch(facecolor='red', label='Ложное обнаружение')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.grid(True, linestyle='--', alpha=0.6)

        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)






if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = ChangePointApp(root)
    root.mainloop()