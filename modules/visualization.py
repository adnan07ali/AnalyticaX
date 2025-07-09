import streamlit as st
import plotly.express as px
import pandas as pd

def run_visualization(df: pd.DataFrame):
    st.subheader("📊 Visualize Your Data")

    # --- Detect Columns ---
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    st.markdown(f"**🧠 Detected {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.**")

    # --- Chart Type ---
    chart_type = st.selectbox("📈 Select Chart Type", [
        "Bar Chart", "Pie Chart", "Line Chart", "Box Plot", "Histogram", "Scatter Plot"
    ])

    color = st.color_picker("🎨 Pick a plot color", "#FF4B4B")

    if chart_type == "Bar Chart":
        col_x = st.selectbox("X-axis (Categorical)", categorical_cols)
        col_y = st.selectbox("Y-axis (Numeric)", numeric_cols)
        agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
        if col_x and col_y:
            plot_df = df.groupby(col_x)[col_y].agg(agg_func).reset_index()
            fig = px.bar(plot_df, x=col_x, y=col_y, color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        col_cat = st.selectbox("Category Column", categorical_cols)
        col_val = st.selectbox("Values Column", numeric_cols)
        if col_cat and col_val:
            pie_df = df.groupby(col_cat)[col_val].sum().reset_index()
            fig = px.pie(pie_df, names=col_cat, values=col_val, color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        col_x = st.selectbox("X-axis", numeric_cols)
        col_y = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        if col_x and col_y:
            fig = px.line(df, x=col_x, y=col_y, line_shape="linear", color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        col_cat = st.selectbox("Categorical Column (X)", categorical_cols)
        col_num = st.selectbox("Numeric Column (Y)", numeric_cols)
        if col_cat and col_num:
            fig = px.box(df, x=col_cat, y=col_num, color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Histogram":
        col = st.selectbox("Numeric Column", numeric_cols)
        bins = st.slider("Number of bins", 5, 100, 20)
        if col:
            fig = px.histogram(df, x=col, nbins=bins, color_discrete_sequence=[color])
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
        hue_col = st.selectbox("Color by (Optional)", [None] + categorical_cols)
        if x_col and y_col:
            fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, color_discrete_sequence=px.colors.qualitative.Safe)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("⚠️ Select valid columns to generate plot.")
