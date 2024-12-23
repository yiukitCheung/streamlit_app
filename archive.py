def plot_support_resistance_histogram(filtered_batch_alerts):
    # Clear the previous figure
    histogram_fig = go.Figure()  # This ensures each time it's called, a fresh figure is created

    if 'support' in filtered_batch_alerts.columns or 'resistance' in filtered_batch_alerts.columns:

        if 'support' in filtered_batch_alerts.columns:
            histogram_fig.add_trace(go.Histogram(
                x=filtered_batch_alerts['support'],
                name='Support',
                marker_color='blue',
                opacity=0.5,
                xbins=dict(size=0.5)
            ))

        if 'resistance' in filtered_batch_alerts.columns:
            histogram_fig.add_trace(go.Histogram(
                x=filtered_batch_alerts['resistance'],
                name='Resistance',
                marker_color='red',
                opacity=0.5,
                xbins=dict(size=0.5)
            ))

        histogram_fig.update_layout(
            barmode='overlay',
            xaxis_title='Price',
            yaxis_title='Count',
            showlegend=False,
            title={
                'text': 'Support/Resistance Levels Strength',
                'x': 0.5,  # Center the title
                'xanchor': 'center',  # Anchor the title at the center
                'yanchor': 'top',  # Anchor the title at the top
                'font': {
                    'size': 18,  # Set the font size
                    'color': 'black',  # Set the font color
                    'family': 'Arial, sans-serif'  # Set the font family
                }
            },
            xaxis=dict(
                showticklabels=True,
                title='Price'
            ),
            yaxis=dict(
                title='Strength'
            )
        )
        st.plotly_chart(histogram_fig, use_container_width=True, showlegend=False, key=f"chart_{time.time()}")
    else:
        st.warning("No support/resistance data available for the selected interval and symbol.")

# Display the latest alerts
if not filtered_live_alerts.empty:
    col1, col2 = st.columns(2)

    with col1:

        only_time = filtered_live_alerts['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        only_time = only_time.head(1).values[0]

        st.markdown(f"""
        <div style="
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: #4CAF50; 
            border: 2px solid #4CAF50; 
            padding: 10px; 
            border-radius: 10px;
            background-color: #f9f9f9;
        ">
        {only_time}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        alert_type = filtered_live_alerts['alert_type'].values[0]

        # Determine the color and message based on the alert type
        if alert_type == 'bullish_engulfer' or alert_type == 'bullish_382':
            alert_message = f"{alert_type.replace('_', ' ').capitalize()} &#x1F7E2;"  # Green Circle Emoji
            alert_color = "#4CAF50"  # Green color
        elif alert_type == 'bearish_engulfer':
            alert_message = f"{alert_type.replace('_', ' ').capitalize()} &#x1F534;"  # Red Circle Emoji
            alert_color = "#FF5733"  # Red color
        st.markdown(f"""
        <div style="
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: {alert_color}; 
            border: 2px solid {alert_color}; 
            padding: 10px; 
            border-radius: 10px;
            background-color: #f9f9f9;
        ">
        {alert_message}
        </div>
        """, unsafe_allow_html=True)
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: #808080; 
            border: 2px solid #808080; 
            padding: 10px; 
            border-radius: 10px;
            background-color: #f9f9f9;
        ">
        {_("No Alert Time Available")}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            text-align: center; 
            font-size: 36px; 
            font-weight: bold; 
            color: #808080;
            border: 2px solid #808080; 
            padding: 10px; 
            border-radius: 10px;
            background-color: #f9f9f9;
        ">
        {_("No Alert Yet")} 
        </div>
        """, unsafe_allow_html=True)