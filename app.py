import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# === МОДЕЛЬ ПО ГОСТ 5639–82 ===
def B_from_G(G):
    a = {
        3: 0.0156,
        5: 0.00390,
        8: 0.00049,
        9: 0.000244,
        10: 0.000122,
    }
    if G not in a:
        return 1.0
    a_ref = 0.000122
    return np.sqrt(a_ref / a[G])

def f_sigma(t, T_C, G, f_inf=0.062, k0=2.05e-3, m=2.31, n=0.80, Q=104000, R=8.314):
    T_K = T_C + 273.15
    B = B_from_G(G)
    exponent = -k0 * (B ** m) * (t ** n) * np.exp(-Q / (R * T_K))
    return f_inf * (1 - np.exp(exponent))

def solve_T_from_f(f_target, t, G, f_inf=0.062, k0=2.05e-3, m=2.31, n=0.80, Q=104000, R=8.314):
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

# === ОБРАБОТКА ФАЙЛА С БЛОКАМИ "8 НОМЕР" ===
def parse_custom_excel(df_raw):
    """Преобразует файл с блоками '8 номер', '9 номер' в плоскую таблицу с колонкой G"""
    rows = []
    current_G = None
    for idx, row in df_raw.iterrows():
        # Ищем строку вида "8 номер"
        first_cell = str(row.iloc[0]).strip()
        if re.match(r'\d+\s+номер', first_cell):
            current_G = int(re.search(r'\d+', first_cell).group())
            continue
        # Ищем заголовки
        if 'Температура' in first_cell or 'G' in first_cell:
            continue
        # Данные
        try:
            T = float(row.iloc[0])
            t = float(row.iloc[1])
            f = float(row.iloc[2])
            if current_G is not None:
                rows.append({"G": current_G, "T": T, "t": t, "f_exp (%)": f})
        except (ValueError, TypeError):
            continue
    return pd.DataFrame(rows)

# === ИНТЕРФЕЙС ===
st.set_page_config(page_title="σ-фаза: 12Х18Н12Т", layout="wide")
st.title("Калькулятор выделения σ-фазы в стали 12Х18Н12Т")

if 'df_current' not in st.session_state:
    st.session_state.df_current = pd.DataFrame()

with st.sidebar:
    st.header("Параметры модели")
    f_inf = st.number_input("f_∞ (равновесная доля)", 0.01, 0.2, 0.062, 0.001)
    k0 = st.number_input("k₀", 1e-6, 1.0, 2.05e-3, format="%.2e")
    m = st.number_input("m (влияние зерна)", 0.0, 5.0, 2.31, 0.01)
    n = st.number_input("n (показатель Аврами)", 0.1, 3.0, 0.80, 0.01)
    Q = st.number_input("Q (Дж/моль)", 100000, 300000, 104000, 1000)

tab1, tab2, tab3 = st.tabs(["Прямой расчёт", "Обратный расчёт", "Валидация"])

# --- Прямой расчёт ---
with tab1:
    st.subheader("Рассчитать долю σ-фазы")
    col1, col2, col3 = st.columns(3)
    G = col1.selectbox("Номер зерна по ГОСТ 5639–82", [3, 5, 8, 9, 10], index=4)
    T = col2.number_input("Температура (°C)", 550, 900, 600)
    t = col3.number_input("Время (ч)", 100, 200000, 10000)
    f_calc = f_sigma(t, T, G, f_inf, k0, m, n, Q) * 100
    st.success(f"Расчётная доля σ-фазы: **{f_calc:.2f} %**")

# --- Обратный расчёт ---
with tab2:
    st.subheader("Оценить температуру по доле σ-фазы")
    col1, col2, col3 = st.columns(3)
    G2 = col1.selectbox("Номер зерна по ГОСТ 5639–82", [3, 5, 8, 9, 10], index=4, key="G2")
    t2 = col2.number_input("Время эксплуатации (ч)", 100, 200000, 100000, key="t2")
    f2 = col3.number_input("Измеренная доля σ-фазы (%)", 0.01, 20.0, 3.5, key="f2")
    T_est = solve_T_from_f(f2 / 100.0, t2, G2, f_inf, k0, m, n, Q)
    if T_est is None:
        st.error("Температура вне диапазона 550–900°C или доля > f_∞")
    else:
        st.success(f"Оценка температуры: **{T_est:.1f} °C**")

# --- Валидация ---
with tab3:
    st.subheader("Валидация модели на экспериментальных данных")
    
    col_load, col_save = st.columns(2)
    with col_load:
        project_file = st.file_uploader("Загрузить проект (.json)", type=["json"])
    with col_save:
        if st.button("Сохранить проект (.json)"):
            if not st.session_state.df_current.empty:
                project_data = {
                    "model_params": {
                        "f_inf": f_inf,
                        "k0": k0,
                        "m": m,
                        "n": n,
                        "Q": Q
                    },
                    "data": st.session_state.df_current.to_dict(orient="records")
                }
                project_json = pd.io.json.dumps(project_data, indent=2)
                st.download_button(
                    label="Скачать проект.json",
                    data=project_json,
                    file_name="sigma_project.json",
                    mime="application/json"
                )
            else:
                st.warning("Нет данных для сохранения.")
    
    input_method = st.radio(
        "Способ ввода данных:",
        ("Загрузить файл (CSV/XLSX)", "Ввести вручную"),
        horizontal=True
    )
    
    df = None
    
    if project_file:
        try:
            project = pd.io.json.loads(project_file.read())
            f_inf = project["model_params"]["f_inf"]
            k0 = project["model_params"]["k0"]
            m = project["model_params"]["m"]
            n = project["model_params"]["n"]
            Q = project["model_params"]["Q"]
            df = pd.DataFrame(project["data"])
            st.success("Проект загружен!")
        except Exception as e:
            st.error(f"Ошибка при загрузке проекта: {e}")
    
    if df is None:
        if input_method == "Загрузить файл (CSV/XLSX)":
            uploaded = st.file_uploader("Файл данных", type=["csv", "xlsx"], key="file_uploader")
            if uploaded:
                try:
                    if uploaded.name.endswith('.csv'):
                        df_raw = pd.read_csv(uploaded, header=None)
                    else:
                        df_raw = pd.read_excel(uploaded, header=None)
                    df = parse_custom_excel(df_raw)
                    if df.empty:
                        # Попытка стандартного чтения
                        if uploaded.name.endswith('.csv'):
                            df = pd.read_csv(uploaded)
                        else:
                            df = pd.read_excel(uploaded, header=1)
                        df = df.dropna(how='all').reset_index(drop=True)
                    if "Исключить" not in df.columns:
                        df["Исключить"] = False
                except Exception as e:
                    st.error(f"Ошибка при чтении файла: {e}")
        else:
            st.markdown("Введите экспериментальные данные:")
            example_data = pd.DataFrame([
                {"G": 10, "T": 600, "t": 2000, "f_exp (%)": 1.26, "Исключить": False},
                {"G": 8, "T": 600, "t": 4000, "f_exp (%)": 0.68, "Исключить": True},
            ])
            df = st.data_editor(
                example_data,
                num_rows="dynamic",
                column_config={
                    "G": st.column_config.NumberColumn("Номер зерна (ГОСТ)", min_value=1, max_value=15, step=1),
                    "T": st.column_config.NumberColumn("Температура (°C)", min_value=500, max_value=950, step=10),
                    "t": st.column_config.NumberColumn("Время (ч)", min_value=100, max_value=200000, step=100),
                    "f_exp (%)": st.column_config.NumberColumn(
                        "Доля σ-фазы (%)",
                        min_value=0.0,
                        max_value=20.0,
                        step=0.01,
                        format="%.2f"
                    ),
                    "Исключить": st.column_config.CheckboxColumn("Исключить", default=False),
                },
                use_container_width=True,
                hide_index=True
            )
    
    if df is not None and not df.empty:
        st.session_state.df_current = df.copy()
        required = {'G', 'T', 't', 'f_exp (%)'}
        if not required <= set(df.columns):
            st.error(f"Таблица должна содержать колонки: {required}. Текущие: {list(df.columns)}")
        else:
            df['G'] = pd.to_numeric(df['G'], errors='coerce').astype('Int64')
            df['T'] = pd.to_numeric(df['T'], errors='coerce')
            df['t'] = pd.to_numeric(df['t'], errors='coerce')
            df['f_exp (%)'] = pd.to_numeric(df['f_exp (%)'], errors='coerce')
            df = df.dropna(subset=['G', 'T', 't', 'f_exp (%)'])
            
            if df.empty:
                st.warning("Не удалось преобразовать данные.")
            else:
                df_filtered = df[df["Исключить"] == False].copy()
                if df_filtered.empty:
                    st.warning("Все строки исключены.")
                else:
                    df_filtered['f_calc (%)'] = df_filtered.apply(
                        lambda row: f_sigma(row['t'], row['T'], row['G'], f_inf, k0, m, n, Q) * 100,
                        axis=1
                    )
                    df_filtered['Ошибка (%)'] = np.abs(df_filtered['f_calc (%)'] - df_filtered['f_exp (%)'])
                    st.dataframe(df_filtered.round(3))
                    
                    fig, ax = plt.subplots()
                    ax.scatter(df_filtered['f_exp (%)'], df_filtered['f_calc (%)'], alpha=0.7, s=60)
                    ax.plot([0, df_filtered['f_exp (%)'].max()*1.1], [0, df_filtered['f_exp (%)'].max()*1.1], 'r--')
                    ax.set_xlabel('Эксперимент (%)')
                    ax.set_ylabel('Расчёт (%)')
                    ax.set_title('Сравнение (исключённые точки не учитываются)')
                    ax.grid(True, linestyle='--', alpha=0.5)
                    st.pyplot(fig)

st.markdown("---")
st.caption("""
Модель основана на уравнении Аврами с учётом плотности границ зёрен по ГОСТ 5639–82.  
Коэффициенты подобраны по экспериментальным данным для стали 12Х18Н12Т (G=8,9,10).  
Диапазон: 550–900 °C, время до 200 000 ч.
""")
