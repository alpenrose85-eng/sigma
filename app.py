import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === МОДЕЛЬ (без scipy) ===
def B_from_G(G):
    return 2 ** ((G - 1) / 2)

def f_sigma(t, T_C, G, f_inf=0.06, k0=1.12e-4, m=2.85, n=0.92, Q=185000, R=8.314):
    T_K = T_C + 273.15
    B = B_from_G(G)
    exponent = -k0 * (B ** m) * (t ** n) * np.exp(-Q / (R * T_K))
    return f_inf * (1 - np.exp(exponent))

def solve_T_from_f(f_target, t, G, f_inf=0.06, k0=1.12e-4, m=2.85, n=0.92, Q=185000, R=8.314):
    if f_target <= 0 or f_target >= f_inf:
        return None
    def model_f(T_C):
        return f_sigma(t, T_C, G, f_inf, k0, m, n, Q, R)
    low, high = 550.0, 900.0
    f_low = model_f(low) - f_target
    f_high = model_f(high) - f_target
    if f_low * f_high > 0:
        return None
    for _ in range(60):
        mid = (low + high) / 2.0
        f_mid = model_f(mid) - f_target
        if abs(f_mid) < 1e-9:
            return mid
        if f_low * f_mid < 0:
            high = mid
        else:
            low = mid
    return (low + high) / 2.0

# === ПАРАМЕТРЫ МОДЕЛИ (можно настраивать) ===
st.set_page_config(page_title="σ-фаза: 12Х18Н12Т", layout="wide")
st.title("Калькулятор выделения σ-фазы в стали 12Х18Н12Т")

# Часто используемые номера зерна
COMMON_GRAINS = [3, 5, 8, 9, 10]

with st.sidebar:
    st.header("Параметры модели")
    f_inf = st.number_input("f_∞ (равновесная доля)", 0.01, 0.2, 0.06, 0.01)
    k0 = st.number_input("k₀", 1e-6, 1.0, 1.12e-4, format="%.2e")
    m = st.number_input("m (влияние зерна)", 0.0, 5.0, 2.85, 0.05)
    n = st.number_input("n (показатель Аврами)", 0.1, 3.0, 0.92, 0.05)
    Q = st.number_input("Q (Дж/моль)", 100000, 300000, 185000, 1000)

# === ВКЛАДКИ ===
tab1, tab2, tab3 = st.tabs(["Прямой расчёт", "Обратный расчёт", "Валидация на данных"])

# --- Прямой расчёт ---
with tab1:
    st.subheader("Рассчитать долю σ-фазы")
    col1, col2, col3 = st.columns(3)
    use_preset = col1.checkbox("Использовать стандартные номера зерна", True)
    if use_preset:
        G = col1.selectbox("Номер зерна ASTM (G)", COMMON_GRAINS, index=4)  # по умолчанию 10
    else:
        G = col1.number_input("Номер зерна ASTM (G)", 1, 15, 10)
    T = col2.number_input("Температура (°C)", 550, 900, 600)
    t = col3.number_input("Время (ч)", 100, 200000, 10000)
    f_calc = f_sigma(t, T, G, f_inf, k0, m, n, Q) * 100
    st.success(f"Расчётная доля σ-фазы: **{f_calc:.2f} %**")

# --- Обратный расчёт ---
with tab2:
    st.subheader("Оценить температуру по доле σ-фазы")
    col1, col2, col3 = st.columns(3)
    use_preset2 = col1.checkbox("Использовать стандартные номера зерна", True, key="preset2")
    if use_preset2:
        G2 = col1.selectbox("Номер зерна ASTM (G)", COMMON_GRAINS, index=4, key="G_sel2")
    else:
        G2 = col1.number_input("Номер зерна ASTM (G)", 1, 15, 10, key="G_num2")
    t2 = col2.number_input("Время эксплуатации (ч)", 100, 200000, 100000, key="t2")
    f2 = col3.number_input("Измеренная доля σ-фазы (%)", 0.1, 20.0, 3.5, key="f2")
    T_est = solve_T_from_f(f2 / 100.0, t2, G2, f_inf, k0, m, n, Q)
    if T_est is None:
        st.error("Температура вне диапазона 550–900°C или доля > f_∞")
    else:
        st.success(f"Оценка температуры: **{T_est:.1f} °C**")

# --- Валидация ---
with tab3:
    st.subheader("Загрузите экспериментальные данные")
    st.markdown("Формат: CSV или XLSX с колонками: `G`, `T`, `t`, `f_exp (%)`")
    uploaded = st.file_uploader("Файл данных", type=["csv", "xlsx"])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            required = {'G', 'T', 't', 'f_exp (%)'}
            if not required <= set(df.columns):
                st.error(f"Нужны колонки: {required}")
            else:
                df['f_calc (%)'] = df.apply(
                    lambda row: f_sigma(row['t'], row['T'], row['G'], f_inf, k0, m, n, Q) * 100,
                    axis=1
                )
                df['Ошибка (%)'] = np.abs(df['f_calc (%)'] - df['f_exp (%)'])
                st.dataframe(df.round(3))

                fig, ax = plt.subplots()
                ax.scatter(df['f_exp (%)'], df['f_calc (%)'], alpha=0.7)
                ax.plot([0, df['f_exp (%)'].max()*1.1], [0, df['f_exp (%)'].max()*1.1], 'r--')
                ax.set_xlabel('Эксперимент (%)')
                ax.set_ylabel('Расчёт (%)')
                ax.set_title('Сравнение')
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Ошибка: {e}")
