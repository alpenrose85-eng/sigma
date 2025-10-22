import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Streamlit UI
st.title("Калькулятор σ-фазы")
tab1, tab2 = st.tabs(["Прямой", "Обратный"])
with tab1:
    G = st.number_input("G", 1, 15, 10)
    T = st.number_input("T (°C)", 550, 900, 600)
    t = st.number_input("t (ч)", 100, 200000, 10000)
    f = f_sigma(t, T, G) * 100
    st.write(f"Доля σ-фазы: {f:.2f} %")
with tab2:
    G = st.number_input("G", 1, 15, 10, key="G2")
    t = st.number_input("t (ч)", 100, 200000, 100000, key="t2")
    f_in = st.number_input("f (%)", 0.1, 20.0, 3.5, key="f2")
    T_est = solve_T_from_f(f_in / 100.0, t, G)
    if T_est:
        st.write(f"Температура: {T_est:.1f} °C")
    else:
        st.write("Температура вне диапазона")
