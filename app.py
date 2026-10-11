import streamlit as st
import joblib
import pandas as pd

# 加载训练好的模型（路径改成你桌面上的实际路径）
model_data = joblib.load('低空风险模型.pkl')
rf_model = model_data['rf']
xgb_model = model_data['xgb']
class_names = model_data['class_names']  # {0:'低风险',1:'中风险',2:'高风险'}

st.set_page_config(page_title="深圳低空风险预警系统", page_icon="🚁", layout="centered")

st.title("🚁 深圳低空经济风险预警系统")
st.markdown("输入飞行环境参数，实时预测无人机飞行风险等级（绿=低 / 黄=中 / 红=高）")

# 选择模型
model_choice = st.radio("选择预测模型", ["随机森林", "XGBoost"])

st.subheader("📊 输入环境参数")
col1, col2, col3 = st.columns(3)
with col1:
    wind = st.number_input("风力等级", min_value=0.0, max_value=12.0, value=2.0, step=1.0)
    high_t = st.number_input("最高温(℃)", min_value=-20.0, max_value=50.0, value=20.0)
    low_t = st.number_input("最低温(℃)", min_value=-20.0, max_value=50.0, value=10.0)
with col2:
    avg_t = st.number_input("平均温(℃)", min_value=-20.0, max_value=50.0, value=15.0)
    month = st.number_input("月份", min_value=1, max_value=12, value=1, step=1)
    day_weather = st.selectbox("白天天气(0/1/2)", [0, 1, 2])
with col3:
    night_weather = st.selectbox("夜间天气(0/1/2)", [0, 1, 2])
    main_weather = st.selectbox("主要天气(0/1/2)", [0, 1, 2])

# 温差自动计算 = 最高温 - 最低温（和训练数据一致）
temp_diff = high_t - low_t
st.caption(f"温差(自动计算) = {high_t} - {low_t} = {temp_diff} ℃")

# 按训练时的特征顺序组装
features = [wind, high_t, low_t, temp_diff, avg_t, month, day_weather, night_weather, main_weather]
feature_names = ['风力等级', '最高温', '最低温', '温差', '平均温', '月份', '白天天气', '夜间天气', '主要天气']

if st.button("🔴 开始风险预测"):
    X_input = pd.DataFrame([features], columns=feature_names)

    if model_choice == "随机森林":
        pred = rf_model.predict(X_input)[0]
    else:
        pred = xgb_model.predict(X_input)[0]

    name = class_names[pred]
    # 展示红绿灯结果
    if pred == 0:
        st.success(f"🟢 低风险：当前环境适合飞行")
    elif pred == 1:
        st.warning(f"🟡 中等风险：建议谨慎飞行")
    else:
        st.error(f"🔴 高风险：禁止飞行！")
    st.metric("预测风险等级", name)
