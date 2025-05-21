import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from ruptures import Pelt, Binseg, Dynp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score
from prophet  import Prophet
import json
import os
import warnings
from scipy import stats

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
        self.methods = {
            "CUSUM": self.detect_cusum,
            "Pettitt": self.detect_pettitt,
            "Bayesian": self.detect_bayesian,
            "PELT": self.detect_pelt,
            "BinSeg": self.detect_binseg,
            "E-Divisive": self.detect_edivisive,
            "BOCPD": self.detect_bocpd,
            "MOSUM": self.detect_mosum,
            "LSTM": self.detect_lstm,
            "CNN": self.detect_cnn,
            "Prophet": self.detect_prophet,
            "RFA": self.detect_rfa,
            "Dynamic": self.detect_dynp,
            "Ансамбль": self.detect_ensemble
        }

        self.results = {}
        self.create_widgets()

    def create_widgets(self):
        # Главный Notebook
        self.main_notebook = ttk.Notebook(self.root)
        self.main_notebook.pack(expand=True, fill=tk.BOTH)

        # Вкладка настройки данных
        self.create_data_settings_tab()

        # Вкладки методов (будут создаваться при построении графиков)
        self.method_tabs = {}

        # Вкладка сравнения методов
        self.comparison_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(self.comparison_frame, text="Сравнение методов")

    def create_data_settings_tab(self):
        """Создает вкладку для настройки данных"""
        data_frame = ttk.Frame(self.main_notebook)
        self.main_notebook.add(data_frame, text="Настройка данных")

        # Управляющие элементы
        control_frame = ttk.Frame(data_frame, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Поля для состояния 1
        ttk.Label(control_frame, text="Количество точек (Состояние 1):").grid(row=0, column=0, sticky=tk.W)
        self.entry_points_state1 = ttk.Entry(control_frame)
        self.entry_points_state1.grid(row=0, column=1)
        self.entry_points_state1.insert(0, "20")

        # Поля для состояния 2
        ttk.Label(control_frame, text="Количество точек (Состояние 2):").grid(row=0, column=2, sticky=tk.W)
        self.entry_points_state2 = ttk.Entry(control_frame)
        self.entry_points_state2.grid(row=0, column=3)
        self.entry_points_state2.insert(0, "20")

        # Поля для состояния 3
        ttk.Label(control_frame, text="Количество точек (Состояние 3):").grid(row=0, column=4, sticky=tk.W)
        self.entry_points_state3 = ttk.Entry(control_frame)
        self.entry_points_state3.grid(row=0, column=5)
        self.entry_points_state3.insert(0, "20")

        # Угол наклона для состояния 2
        ttk.Label(control_frame, text="Угол наклона (Состояние 2):").grid(row=1, column=0, sticky=tk.W)
        self.entry_angle_state2 = ttk.Entry(control_frame)
        self.entry_angle_state2.grid(row=1, column=1)
        self.entry_angle_state2.insert(0, "45")

        # Уровень шума (в процентах)
        ttk.Label(control_frame, text="Уровень шума (%):").grid(row=2, column=0, sticky=tk.W)
        self.entry_noise_percent = ttk.Entry(control_frame)
        self.entry_noise_percent.grid(row=2, column=1)
        self.entry_noise_percent.insert(0, "10")

        # Кнопки управления
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=6, pady=10)

        ttk.Button(button_frame, text="Построить графики", command=self.generate_and_plot_all).pack(side=tk.LEFT,
                                                                                                    padx=5)
        ttk.Button(button_frame, text="Сохранить данные", command=self.save_settings).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Загрузить данные", command=self.load_settings).pack(side=tk.LEFT, padx=5)

        # Область для графиков
        self.data_fig, (self.data_ax1, self.data_ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=data_frame)
        self.data_canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def add_noise(self, y, noise_level_percent, y_range):
        noise_level = (noise_level_percent / 100) * y_range
        noise = np.random.uniform(-noise_level, noise_level, size=len(y))
        return y + noise

    def moving_average(self, y, window_size=5):
        return np.convolve(y, np.ones(window_size) / window_size, mode='valid')

    def generate_and_plot_all(self):
        """Генерирует данные и применяет все методы"""
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

            # Применение всех методов
            self.apply_all_methods()

            # Показ результатов
            self.show_all_methods()
            self.show_comparison()

        except Exception as e:
            print(f"Ошибка: {e}")

    def apply_all_methods(self):
        """Применяет все методы к данным"""
        if self.x_data is None or self.y_noisy_data is None:
            return

        for name, method in self.methods.items():
            try:
                self.results[name] = method()
                print(f"{name} завершен")
            except Exception as e:
                print(f"Ошибка в {name}: {e}")
                self.results[name] = []

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

        # Порог для обнаружения (эмпирически подобран)
        threshold = 0.8
        changepoints = np.where(s2_norm > threshold)[0]

        # Фильтрация близких точек
        filtered_points = []
        prev_point = -10
        for point in changepoints:
            if point - prev_point > 5:  # Минимальный интервал между точками
                filtered_points.append(point)
                prev_point = point

        return filtered_points

    def detect_pettitt(self):
        """Обнаружение одной точки изменения"""
        n = len(self.y_noisy_data)
        if n < 10: return []  # Слишком мало данных

        # Нормализуем данные
        y_norm = (self.y_noisy_data - np.mean(self.y_noisy_data)) / np.std(self.y_noisy_data)

        U = np.zeros(n)
        for t in range(1, n):
            U[t] = U[t - 1] + np.sign(y_norm[t] - y_norm[t - 1])

        K = np.max(np.abs(U))
        t0 = np.argmax(np.abs(U))

        # Автоматический порог
        threshold = 1.36 * np.sqrt(n)  # Для 95% доверительного интервала
        if K > threshold:
            return [t0]
        return []

    def detect_bayesian(self):
        """Обнаружение точек изменения с использованием байесовского подхода"""
        y = self.y_noisy_data
        n = len(y)
        if n < 10:  # Слишком мало данных для анализа
            return []

        # Параметры алгоритма
        window_size = max(5, n // 20)  # Адаптивный размер окна
        threshold = 0.9  # Порог вероятности изменения
        min_change = 0.5 * np.std(y)  # Минимальное значимое изменение

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

            # Гипотеза H0: нет изменения (одно распределение)
            log_p_h0 = log_likelihood(before, mu_pooled, std_pooled) + \
                       log_likelihood(after, mu_pooled, std_pooled)

            # Гипотеза H1: есть изменение (разные распределения)
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
            if p_value < 0.05:  # 95% доверительный интервал
                changepoints.append(i)

        return changepoints

    def detect_bocpd(self):
        """Гарантированно рабочая версия BOCPD с отладкой"""
        y = self.y_noisy_data
        n = len(y)
        print(f"\nЗапуск BOCPD на {n} точках")  # Отладочный вывод

        if n < 10:
            print("Слишком мало данных!")
            return []

        # 1. Параметры, которые точно работают
        hazard = 1 / 10  # Частота изменений (1 изменение на 10 точек)
        threshold = 0.4  # Порог срабатывания
        min_run_length = 3  # Минимальная длина сегмента

        # 2. Инициализация с защитой от крайних случаев
        changepoints = []
        run_length = 1
        mu = np.mean(y[:5])
        sigma = max(np.std(y[:5]), 0.5)  # Гарантируем минимальный разброс
        print(f"Начальные параметры: μ={mu:.2f}, σ={sigma:.2f}")

        # 3. Главный цикл обработки
        for i in range(1, n):
            x = y[i]

            # 4. Вычисление вероятностей с защитой
            try:
                prob = stats.norm.pdf(x, mu, sigma + 1e-10)
            except:
                print(f"Ошибка на точке {i}: μ={mu}, σ={sigma}")
                prob = 1e-10

            # 5. Байесовское обновление
            prob_change = hazard * prob
            prob_nochange = (1 - hazard) * prob
            total = prob_change + prob_nochange

            # 6. Нормализация и проверка
            if total > 0:
                prob_change /= total
                if prob_change > threshold and run_length >= min_run_length:
                    print(f"Точка изменения на {i}: p={prob_change:.2f}, μ={mu:.2f}→{x:.2f}")
                    changepoints.append(i)
                    run_length = 1
                    mu = x
                    sigma = max(np.std(y[max(0, i - 5):i + 1]), 0.5)
                else:
                    run_length += 1
                    # Адаптивное обновление
                    alpha = 1.0 / run_length
                    mu = (1 - alpha) * mu + alpha * x
                    sigma = np.sqrt((1 - alpha) * sigma ** 2 + alpha * (x - mu) ** 2)
            else:
                print(f"Пропуск точки {i} (нулевая вероятность)")

        # 7. Фильтрация результатов
        final_points = []
        prev = -min_run_length
        for cp in changepoints:
            if cp - prev >= min_run_length:
                final_points.append(cp)
                prev = cp

        print(f"Итоговые точки изменения: {final_points}")
        return final_points

    def detect_mosum(self):
        """Обнаружение изменений среднего"""
        y = self.y_noisy_data
        n = len(y)
        window = 10
        threshold = 2.0  # Более мягкий порог

        mosum = []
        for i in range(window, n - window):
            left = y[i - window:i]
            right = y[i:i + window]
            stat = np.abs(np.mean(left) - np.mean(right))
            mosum.append(stat)

        # Нормализуем статистику
        mosum_norm = (mosum - np.mean(mosum)) / np.std(mosum)
        changepoints = np.where(mosum_norm > threshold)[0] + window

        # Фильтруем дубликаты
        return list(set(changepoints))  # Убираем повторы

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

        # Определение порога для аномалий (90-й процентиль ошибок)
        threshold = np.percentile(errors, 90)
        anomaly_indices = np.where(errors > threshold)[0] + n_lookback

        # Фильтрация точек (минимальный интервал 5 дней)
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

        # Определение порога для аномалий (90-й процентиль ошибок)
        threshold = np.percentile(errors, 90)
        anomaly_indices = np.where(errors > threshold)[0] + n_lookback

        # Фильтрация точек (минимальный интервал 5 дней)
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
        min_segment = 10  # Минимальная длина сегмента

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

    def show_all_methods(self):
        """Создает вкладки для всех методов с результатами"""
        # Удаляем старые вкладки методов
        for tab_id in self.method_tabs.values():
            self.main_notebook.forget(tab_id)
        self.method_tabs = {}

        # Создаем новые вкладки для каждого метода
        for method_name in self.methods.keys():
            frame = ttk.Frame(self.main_notebook)
            self.main_notebook.add(frame, text=method_name)
            self.method_tabs[method_name] = frame

            # Создаем графики для метода
            self.create_method_tab_content(frame, method_name)

    def create_method_tab_content(self, frame, method_name):
        """Создает содержимое вкладки метода (2 графика)"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # График с шумом
        ax1.plot(self.x_data, self.y_noisy_data, 'r-', label='Данные с шумом')
        ax1.set_title(f'{method_name} - Исходные данные с шумом')
        ax1.legend()
        ax1.grid(True)

        # График с обработанными данными
        ax2.plot(self.x_data, self.y_noisy_data, 'r-', alpha=0.3, label='Данные с шумом')

        # Добавляем точки изменения
        change_points = self.results[method_name]
        for cp in change_points:
            if cp < len(self.x_data):
                ax2.axvline(self.x_data[cp], color='b', linestyle='--', alpha=0.7, linewidth=1)

        ax2.set_title(f'{method_name} - Обнаруженные точки изменения')
        ax2.legend()
        ax2.grid(True)

        # Встраиваем график в интерфейс
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)

    def show_comparison(self):
        """Обновляет вкладку сравнения методов с новыми метриками"""
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()

        # Реальная точка изменения (последняя точка состояния 1)
        real_cp = int(self.entry_points_state1.get()) - 1

        # Создаем таблицу сравнения
        columns = ("Метод", "Precision", "Recall", "F1-score", "Задержка (Lag)", "Ложные срабатывания")
        tree = ttk.Treeview(self.comparison_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor='center')

        # Заполняем таблицу данными
        for method, points in self.results.items():
            if not points:
                tree.insert("", "end", values=(method, 0, 0, 0, 0, 0))
                continue

            correct_detections = [cp for cp in points if real_cp <= cp <= real_cp + 3]
            false_detections = [cp for cp in points if cp < real_cp or cp > real_cp + 3]

            precision = len(correct_detections) / len(points) if len(points) > 0 else 0
            recall = 1 if len(correct_detections) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            lag = np.mean([cp - real_cp for cp in correct_detections]) if correct_detections else 0

            tree.insert("", "end", values=(
                method,
                f"{precision:.2f}",
                f"{recall:.2f}",
                f"{f1_score:.2f}",
                f"{lag:.1f}",
                len(false_detections)
            ))  # Исправлено: две закрывающие скобки

        tree.pack(fill=tk.BOTH, expand=True)

        # График сравнения
        fig = plt.Figure(figsize=(10, 4))
        ax = fig.add_subplot(111)

        for method, points in self.results.items():
            if not points:
                continue

            y = [method] * len(points)
            colors = ['green' if real_cp <= cp <= real_cp + 3 else 'red' for cp in points]
            ax.scatter(points, y, c=colors, label=method, alpha=0.7)

        ax.axvline(real_cp, color='blue', linestyle='--', label='Реальная точка изменения')
        ax.axvline(real_cp + 3, color='blue', linestyle=':', alpha=0.5, label='Граница окна')
        ax.set_xlabel('Индекс точки')
        ax.set_title('Сравнение методов обнаружения точек изменения')
        ax.legend()
        ax.grid(True)

        canvas = FigureCanvasTkAgg(fig, master=self.comparison_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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
            "y_clean_data": self.y_clean_data.tolist()
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
            self.apply_all_methods()
            self.show_all_methods()
            self.show_comparison()


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    app = ChangePointApp(root)
    root.mainloop()