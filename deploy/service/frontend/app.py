import streamlit as st
import requests
import time
import json
import os
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="LAVAD - 视频异常检测",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 LAVAD - 视频异常检测系统")
st.markdown("**L**anguage-based **A**nomaly **V**AD - 利用大语言模型实现无需训练的视频异常检测")

st.sidebar.title("配置")
cloud_url = st.sidebar.text_input("云端服务器地址", value=os.environ.get("API_URL", "http://8.147.71.33:8000"))
st.sidebar.markdown("---")
st.sidebar.info("""
**使用说明：**
1. 点击下方按钮上传视频文件
2. 点击"开始检测"提交任务
3. 等待处理完成后查看结果
4. 可下载可视化结果视频
""")

if "result" not in st.session_state:
    st.session_state["result"] = None
if "task_id" not in st.session_state:
    st.session_state["task_id"] = None

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📤 视频上传")
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=["mp4", "avi"],
        help="支持 MP4 和 AVI 格式"
    )

    if uploaded_file is not None:
        st.success(f"已选择: {uploaded_file.name}")
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"文件大小: {file_size:.2f} MB")

    if st.button("🚀 开始检测", type="primary", disabled=uploaded_file is None):
        if uploaded_file is None:
            st.error("请先上传视频文件")
        else:
            with st.spinner("正在上传并处理视频..."):
                try:
                    files = {"video": (uploaded_file.name, uploaded_file.getvalue(), "video/mp4")}
                    response = requests.post(
                        f"{cloud_url}/predict",
                        files=files,
                        timeout=3600
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.session_state["task_id"] = result["task_id"]
                        st.session_state["result"] = None
                        st.success("视频上传成功！")
                        st.info(f"任务ID: {result['task_id']}")
                        st.rerun()
                    else:
                        st.error(f"上传失败: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"无法连接到云端服务器 {cloud_url}")
                except Exception as e:
                    st.error(f"发生错误: {str(e)}")

with col2:
    st.subheader("📊 检测结果")

    if st.session_state["task_id"] and not st.session_state["result"]:
        task_id = st.session_state["task_id"]
        progress_bar = st.progress(0)
        status_text = st.empty()

        while True:
            try:
                progress_response = requests.get(f"{cloud_url}/progress/{task_id}", timeout=10)
                if progress_response.status_code == 200:
                    progress_data = progress_response.json()
                    status = progress_data.get("status", "unknown")
                    percent = progress_data.get("percent", 0)
                    message = progress_data.get("message", "")

                    progress_bar.progress(percent / 100)
                    status_text.text(f"状态: {message}")

                    if status == "completed":
                        st.success("✅ 处理完成！")
                        progress_bar.progress(100)
                        status_text.text("处理完成！")

                        result_response = requests.get(f"{cloud_url}/result/{task_id}", timeout=30)
                        if result_response.status_code == 200:
                            result_data = result_response.json()
                            st.session_state["result"] = result_data.get("result")
                        break
                    elif status == "error":
                        st.error(f"❌ 处理失败: {message}")
                        break
                    elif status == "not_found":
                        st.warning("⚠️ 任务不存在")
                        break

                time.sleep(2)

            except Exception as e:
                st.warning(f"获取进度失败: {str(e)}, 继续等待...")
                time.sleep(2)

        st.rerun()

    if st.session_state["result"]:
        result = st.session_state["result"]
        task_id = st.session_state["task_id"]

        has_anomaly = result.get("has_anomaly", False)
        anomaly_count = result.get("anomaly_count", 0)
        anomalous_frames = result.get("anomalous_frames", [])
        anomaly_details = result.get("anomaly_details", [])
        summary = result.get("summary", {})
        all_scores = result.get("all_scores", {})

        if has_anomaly:
            st.error(f"🚨 检测到异常！共 {anomaly_count} 个异常帧")
        else:
            st.success("✅ 未检测到异常 - 视频内容正常")

        st.markdown("---")
        st.subheader("📋 检测概览")

        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("总帧数", summary.get("total_frames", 0))
        with col_info2:
            st.metric("异常帧数", anomaly_count)
        with col_info3:
            ratio = summary.get("anomaly_ratio", 0)
            st.metric("异常比例", f"{ratio}%")
        with col_info4:
            st.metric("检测状态", "有异常" if has_anomaly else "正常")

        st.markdown("---")

        if anomalous_frames:
            st.subheader("🔍 异常帧画面")
            st.warning(f"以下 {len(anomaly_details)} 个帧检测到异常，点击展开查看详情")

            for i, frame_info in enumerate(anomaly_details[:5]):
                frame_num = frame_info['frame']
                time_sec = frame_info['time']
                caption = frame_info['caption']
                anomaly_types = frame_info['anomaly_types']
                score = frame_info['score']

                with st.expander(f"⚠️ 帧 {frame_num} | 时间: {time_sec}s | 类型: {', '.join(anomaly_types)} | 置信度: {score}"):
                    col_img, col_info = st.columns([1, 1])
                    with col_img:
                        try:
                            frame_url = f"{cloud_url}/frame/{task_id}/{frame_num}"
                            st.image(frame_url, caption=f"帧 {frame_num}", width=300)
                        except Exception as e:
                            st.warning(f"无法加载帧图片")
                    with col_info:
                        st.markdown(f"**帧号:** {frame_num}")
                        st.markdown(f"**时间:** {time_sec} 秒")
                        st.markdown(f"**异常类型:** {', '.join(anomaly_types)}")
                        st.markdown(f"**置信度:** {score}")
                        st.markdown(f"**Caption:** {caption}")

            st.markdown("---")

        st.subheader("📈 异常分数曲线")

        if all_scores:
            scores_df = pd.DataFrame([
                {"帧号": int(k), "异常分数": v} for k, v in all_scores.items()
            ])
            st.bar_chart(scores_df.set_index("帧号"))
            st.info(f"共有 {len(all_scores)} 帧检测到异常关键词")
        else:
            st.info("✅ 所有帧 Caption 已生成，未检测到异常关键词")
            st.success("视频内容正常，无异常行为")

        st.markdown("---")
        st.subheader("📄 检测摘要")
        st.write(f"视频共 **{summary.get('total_frames', 0)}** 帧，检测到 **{anomaly_count}** 个异常帧，异常比例为 **{summary.get('anomaly_ratio', 0)}%**")

    elif st.session_state["task_id"]:
        st.info("👆 请等待检测完成...")
    else:
        st.info("👈 请先上传视频并开始检测")
        st.markdown("""
        ### 项目介绍

        **LAVAD** 是一种基于大语言模型的**无需训练**视频异常检测方法。

        **核心流程：**
        1. 使用 **BLIP-2** 生成视频帧的文本描述
        2. 使用 **ImageBind** 构建多模态索引
        3. 使用 **LLaMA 2** 生成时序摘要并评估异常分数
        4. 通过跨模态相似度 refinement 异常分数

        **支持的异常类型：**
        - 暴力行为 (fight, punch, attack...)
        - 摔倒 (fall, collapse, slip...)
        - 奔跑/追逐 (run, chase, flee...)
        - 异常活动 (scream, fire, weapon...)
        - 事故 (crash, collision, explosion...)
        """)