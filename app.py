import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="低空无人机风险预警", layout="wide")
st.title("🚁 低空无人机飞行风险预警系统")
st.subheader("输入飞行环境参数，实时预测飞行风险等级（0=安全 / 1=低风险 / 2=高风险）")

# ===== 固定加载随机森林（已移除XGBoost选项）=====
model_data = joblib.load('低空风险模型.pkl')
model = model_data['rf']

# ===== 三套天气 数字→名称 映射表（与训练数据一致）=====
weather_day = {0:'多云',1:'晴',2:'阴',3:'大雨',4:'小雨',5:'雷阵雨',6:'中雨',7:'暴雨',8:'阵雨',9:'小到中雨',10:'大到暴雨',11:'中到大雨'}
weather_night = {0:'晴',1:'多云',2:'阴',3:'阵雨',4:'大雨',5:'小雨',6:'雷阵雨',7:'中雨',8:'暴雨',9:'雾',10:'小到中雨',11:'大暴雨',12:'大到暴雨'}
weather_main = {0:'多云',1:'晴',2:'阴',3:'大雨',4:'小雨',5:'雷阵雨',6:'中雨',7:'暴雨',8:'阵雨',9:'小到中雨',10:'大到暴雨',11:'中到大雨'}

def name_to_code(weather_dict, name):
    for k, v in weather_dict.items():
        if v == name:
            return k

st.header("输入环境参数")
col1, col2, col3 = st.columns(3)

with col1:
    wind_level = st.number_input("风力等级", min_value=0.0, max_value=12.0, value=2.0, step=1.0)
    max_temp = st.number_input("最高温(℃)", min_value=-20.0, max_value=50.0, value=20.0)
    min_temp = st.number_input("最低温(℃)", min_value=-20.0, max_value=50.0, value=10.0)

with col2:
    avg_temp = st.number_input("平均温(℃)", min_value=-20.0, max_value=50.0, value=15.0)
    month = st.number_input("月份", min_value=1, max_value=12, value=1, step=1)
    day_weather_text = st.selectbox("白天天气", list(weather_day.values()))

with col3:
    night_weather_text = st.selectbox("夜间天气", list(weather_night.values()))
    main_weather_text = st.selectbox("主要天气", list(weather_main.values()))

# 温差自动计算
temp_diff = max_temp - min_temp
st.info(f"温差(自动计算) = {max_temp:.1f} - {min_temp:.1f} = {temp_diff:.1f} ℃")

# 天气文字转回训练时的数字编码
day_weather = name_to_code(weather_day, day_weather_text)
night_weather = name_to_code(weather_night, night_weather_text)
main_weather = name_to_code(weather_main, main_weather_text)

if st.button("🔴 开始风险预测"):
    input_data = pd.DataFrame([{
        "风力等级": wind_level,
        "最高温": max_temp,
        "最低温": min_temp,
        "温差": temp_diff,
        "平均温": avg_temp,
        "月份": month,
        "白天天气": day_weather,
        "夜间天气": night_weather,
        "主要天气": main_weather
    }])
    pred = model.predict(input_data)[0]
    pred_proba = model.predict_proba(input_data)[0]

    st.divider()
    st.subheader("预测结果")
    if pred == 0:
        st.success("✅ 安全")
    elif pred == 1:
        st.warning("⚠️ 低风险")
    else:
        st.error("🔴 高风险")
    st.write(f"各类别概率：安全{pred_proba[0]:.2f}，低风险{pred_proba[1]:.2f}，高风险{pred_proba[2]:.2f}")
