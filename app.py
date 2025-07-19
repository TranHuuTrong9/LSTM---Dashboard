# app.py

import streamlit as st
from LSTM_trending_core import forecast_lstm
import pandas as pd

st.set_page_config(page_title="LSTM Stock Forecast Dashboard", layout="wide")

st.title("📈 LSTM Multi-Stock Forecast Dashboard")
st.write(
    "Dự báo giá đóng cửa 10 ngày tới cho nhiều cổ phiếu Việt Nam bằng mô hình LSTM. "
    "Kết quả trực quan trên biểu đồ và có thể tải về bảng dự báo."
)

TICKERS = [
    'VNINDEX','CTR','BID','HPG','FPT','SGP','TRC','MWG','DPM','DCM','DBD','MBB','STB','DBC','VCG','SSI','EVF','DHC'
]

ticker = st.selectbox("Chọn mã cổ phiếu", TICKERS, index=3)
predict_days = st.slider("Số ngày dự báo", min_value=5, max_value=30, value=10, step=1)
run = st.button("Dự báo!")

if run:
    with st.spinner(f"Đang dự báo cho {ticker}..."):
        forecast_df, fig = forecast_lstm(ticker, predict_days)
        if forecast_df is None:
            st.error("Dữ liệu không đủ để dự báo.")
        else:
            st.pyplot(fig)
            st.subheader("Bảng dự báo chi tiết")
            st.dataframe(forecast_df, hide_index=True, use_container_width=True)
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải về kết quả (CSV)",
                data=csv,
                file_name=f'forecast_{ticker}.csv',
                mime='text/csv'
            )
