import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# === ПАРАМЕТРЫ МОДЕЛИ (можно редактировать) ===
st.sidebar.header("Коэффициенты модели")
f_inf = st.sidebar.number_input("f_∞ (равновесная доля, доля)", 0.01, 0.2, 0.06, 0.01)
k0 = st.sidebar.number_input("k₀", 1e-6, 1.0, 1.12e-4, format="%.2e")
m = st.sidebar.number_input("m (влияние зерна)", 0.0, 5.0, 2.85, 0.05)
n = st.sidebar.number_input("n (показатель Аврами)", 0.1, 3.0, 0.92, 0.05)
Q = st.sidebar.number_input("Q (энергия активации, Дж/моль)", 100000, 300000, 185000, 1000)
R = 8.314

def B_from_G(G):
    return 2 ** ((G - 1) / 2)

def f_model(t, T_C, G):
    T_K = T_C + 273.15
    B = B_from_G(G)
    exponent = -k0 * (B ** m) * (t ** n) * np.exp(-Q / (R * T_K))
    return f_inf * (1 - np.exp(exponent))

def solve_for_T(f_target, t, G):
    if f_target <= 0 or f_target >= f_inf:
        return np.nan
    def equation(T_C):
        return f_model(t, T_C, G) - f_target
    try:
        T_sol = fsolve(equation, 600)[0]
        if 550 <= T_sol <= 900:
            return T_sol
        else:
            return np.nan
    except:
        return np.nan

# === ОСНОВНОЙ ИНТЕРФЕЙС ===
st.title("Калькулятор выделения σ-фазы в стали 12Х18Н12Т")
st.markdown("Модель учитывает: температуру, время, номер зерна (ASTM)")

tab1, tab2 = st.tabs(["Прямой расчёт", "Обратный расчёт"])

with tab1:
    st.subheader("Рассчитать долю σ-фазы")
    col1, col2, col3 = st.columns(3)
    G = col1.number_input("Номер зерна ASTM (G)", 3, 12, 10)
    T = col2.number_input("Температура (°C)", 550, 900, 600)
    t = col3.number_input("Время (ч)", 100, 200000, 10000)
    f_calc = f_model(t, T, G) * 100
    st.success(f"Расчётная доля σ-фазы: **{f_calc:.2f} %**")

with tab2:
    st.subheader("Оценить температуру по доле фазы")
    col1, col2, col3 = st.columns(3)
    G2 = col1.number_input("Номер зерна ASTM (G)", 3, 12, 10, key="G2")
    t2 = col2.number_input("Время эксплуатации (ч)", 100, 200000, 100000, key="t2")
    f2 = col3.number_input("Измеренная доля σ-фазы (%)", 0.1, 20.0, 3.5, key="f2")
    T_est = solve_for_T(f2 / 100.0, t2, G2)
    if np.isnan(T_est):
        st.error("Температура вне диапазона 550–900°C или доля превышает f_∞")
    else:
        st.success(f"Оценка температуры эксплуатации: **{T_est:.1f} °C**")

# === ЗАГРУЗКА ДАННЫХ И ВАЛИДАЦИЯ ===
st.subheader("Валидация модели на экспериментальных данных")
uploaded = st.file_uploader("Загрузите CSV/XLSX с колонками: G, T, t, f_exp (%)", type=["csv", "xlsx"])
if uploaded:
    if uploaded.name.endswith('.csv'):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    df['f_calc (%)'] = df.apply(lambda row: f_model(row['t'], row['T'], row['G']) * 100, axis=1)
    df['Ошибка (%)'] = np.abs(df['f_calc (%)'] - df['f_exp (%)'])
    st.dataframe(df)
    fig, ax = plt.subplots()
    ax.scatter(df['f_exp (%)'], df['f_calc (%)'], alpha=0.7)
    ax.plot([0, 10], [0, 10], 'r--')
    ax.set_xlabel('Эксперимент (%)')
    ax.set_ylabel('Расчёт (%)')
    ax.set_title('Сравнение эксперимента и модели')
    st.pyplot(fig)