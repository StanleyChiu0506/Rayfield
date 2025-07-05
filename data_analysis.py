import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from fpdf import FPDF
import tempfile
import os

# Try importing ARIMA with error handling
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

# Set page config
st.set_page_config(layout="wide", page_title="Energy Data Analytics Suite")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'user_params' not in st.session_state:
    st.session_state.user_params = {}

# File upload section
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload energy data file", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        st.session_state.df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        st.session_state.df = pd.read_excel(uploaded_file)
    
    st.sidebar.success("Data loaded successfully!")
    st.session_state.user_params = {}  # Reset params when new data loads

# Main analysis interface
if st.session_state.df is not None:
    st.header("Energy Data Analytics Dashboard")
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Data Overview", 
        "Time Analysis", 
        "Efficiency Metrics",
        "Cost Analysis",
        "Emissions Calculator",
        "Forecasting",
        "Energy Mix",
        "Report Generator"
    ])

    with tab1:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head(10))
        
        st.subheader("Data Summary")
        st.write(st.session_state.df.describe())
        
        st.subheader("Interactive Data Visualization")
        
        # Let user select columns to plot
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        date_cols = [col for col in st.session_state.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if len(date_cols) > 0 and len(numeric_cols) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Auto-select first date column
                x_axis = st.selectbox("X-axis (time)", date_cols, index=0)
            
            with col2:
                y_axis = st.selectbox("Y-axis (value)", numeric_cols)
            
            # Fuel type filtering
            fuel_filter = None
            if 'fuel_type' in st.session_state.df.columns:
                fuel_options = ['All Fuel Types'] + list(st.session_state.df['fuel_type'].unique())
                selected_fuel = st.selectbox("Filter by fuel type", fuel_options)
                if selected_fuel != 'All Fuel Types':
                    fuel_filter = selected_fuel
            
            # Apply filters and sort by x-axis
            plot_data = st.session_state.df.copy()
            if fuel_filter:
                plot_data = plot_data[plot_data['fuel_type'] == fuel_filter]
            plot_data = plot_data.sort_values(x_axis)
            
            # Convert x-axis to proper datetime if needed
            try:
                x_values = pd.to_datetime(plot_data[x_axis])
                is_datetime = True
            except:
                x_values = plot_data[x_axis]
                is_datetime = False
            
            # Plot the data
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot individual points and connecting lines
            if 'fuel_type' in plot_data.columns:
                for fuel_type in plot_data['fuel_type'].unique():
                    subset = plot_data[plot_data['fuel_type'] == fuel_type].sort_values(x_axis)
                    if is_datetime:
                        x_plot = pd.to_datetime(subset[x_axis])
                    else:
                        x_plot = subset[x_axis]
                    # Plot connecting line first (behind points)
                    ax.plot(x_plot, subset[y_axis], '-', alpha=0.3, label=f'_nolegend_')
                    # Then plot points on top
                    ax.scatter(x_plot, subset[y_axis], label=fuel_type, alpha=0.8)
            else:
                # Plot single connecting line and points
                ax.plot(x_values, plot_data[y_axis], '-', color='blue', alpha=0.3)
                ax.scatter(x_values, plot_data[y_axis], alpha=0.8)
            
            # Formatting
            if is_datetime:
                ax.set_xlabel(x_axis)
                fig.autofmt_xdate()
                ax.xaxis.set_major_locator(plt.MaxNLocator(8))
            else:
                ax.set_xlabel(x_axis)
                ax.xaxis.set_major_locator(plt.MaxNLocator(10))
                plt.xticks(rotation=45)
            
            ax.set_ylabel(y_axis)
            title = f"{y_axis} Timeline"
            if fuel_filter:
                title += f" ({fuel_filter})"
            ax.set_title(title)
            ax.grid(True)
            
            if 'fuel_type' in plot_data.columns and len(plot_data['fuel_type'].unique()) > 1:
                ax.legend()
            
            st.pyplot(fig)
            
            # Add statistics
            if x_axis in numeric_cols:
                correlation = plot_data[[x_axis, y_axis]].corr().iloc[0,1]
                st.write(f"Correlation between {x_axis} and {y_axis}: {correlation:.2f}")
        else:
            st.warning("Need at least one date/time column and one numeric column for plotting")

    # 1. Time Period Analysis
    with tab2:
        st.subheader("Time-Based Analysis")
        
        # Let user select date column
        date_cols = [col for col in st.session_state.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        selected_date_col = st.selectbox("Select date/time column", date_cols if date_cols else st.session_state.df.columns)
        
        # Let user select value column
        value_col = st.selectbox("Select metric to analyze", st.session_state.df.select_dtypes(include=np.number).columns)
        
        try:
            st.session_state.df[selected_date_col] = pd.to_datetime(st.session_state.df[selected_date_col])
            min_date = st.session_state.df[selected_date_col].min()
            max_date = st.session_state.df[selected_date_col].max()
            
            date_range = st.date_input("Select analysis period", [min_date, max_date])
            
            if len(date_range) == 2:
                mask = (st.session_state.df[selected_date_col] >= pd.to_datetime(date_range[0])) & \
                       (st.session_state.df[selected_date_col] <= pd.to_datetime(date_range[1]))
                filtered_data = st.session_state.df.loc[mask]
                
                col1, col2 = st.columns(2)
                with col1:
                    total_val = filtered_data[value_col].sum()
                    overall_total = st.session_state.df[value_col].sum()
                    st.metric(f"Total {value_col}", 
                             f"{total_val:,.2f}",
                             delta=f"{(total_val/overall_total-1)*100:.1f}% vs total")
                
                with col2:
                    avg_val = filtered_data[value_col].mean()
                    overall_avg = st.session_state.df[value_col].mean()
                    st.metric(f"Average {value_col}",
                             f"{avg_val:,.2f}",
                             delta=f"{(avg_val/overall_avg-1)*100:.1f}% vs average")
                
                st.line_chart(filtered_data.set_index(selected_date_col)[value_col])
        except Exception as e:
            st.error(f"Date processing error: {str(e)}")

    # 2. Efficiency Metrics
    with tab3:
        st.subheader("Energy Efficiency Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            input_col = st.selectbox("Select INPUT energy column", 
                                   st.session_state.df.select_dtypes(include=np.number).columns)
        with col2:
            output_col = st.selectbox("Select OUTPUT energy column", 
                                    st.session_state.df.select_dtypes(include=np.number).columns)
        
        if st.button("Calculate Efficiency"):
            st.session_state.df['efficiency'] = (st.session_state.df[output_col] / st.session_state.df[input_col]) * 100
            
            st.write("### Efficiency Distribution")
            fig, ax = plt.subplots()
            st.session_state.df['efficiency'].hist(ax=ax, bins=20)
            st.pyplot(fig)
            
            if 'fuel_type' in st.session_state.df.columns:
                st.write("### Efficiency by Fuel Type")
                efficiency_by_fuel = st.session_state.df.groupby('fuel_type')['efficiency'].mean().sort_values()
                st.bar_chart(efficiency_by_fuel)
            
            st.write("### Least Efficient Operations")
            st.dataframe(st.session_state.df.nsmallest(5, 'efficiency'))

    # 3. Cost Analysis
    with tab4:
        st.subheader("Cost Analysis")
        
        cost_col = st.selectbox("Select cost column", 
                              st.session_state.df.select_dtypes(include=np.number).columns)
        output_col = st.selectbox("Select output column for cost analysis",
                                st.session_state.df.select_dtypes(include=np.number).columns)
        
        cost_by_fuel = st.session_state.df.groupby('fuel_type')[cost_col].sum() if 'fuel_type' in st.session_state.df.columns else None
        cost_per_unit = st.session_state.df.groupby('fuel_type').apply(
            lambda x: x[cost_col].sum() / x[output_col].sum()) if 'fuel_type' in st.session_state.df.columns else None
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Total Cost Breakdown")
            if cost_by_fuel is not None:
                st.bar_chart(cost_by_fuel)
            else:
                st.write(f"Total Cost: {st.session_state.df[cost_col].sum():,.2f}")
        
        with col2:
            st.write("### Cost per Unit Output")
            if cost_per_unit is not None:
                st.bar_chart(cost_per_unit)
            else:
                st.write(f"Average Cost per Unit: {st.session_state.df[cost_col].sum()/st.session_state.df[output_col].sum():,.2f}")

    # 4. Emissions Calculator
    with tab5:
        st.subheader("Emissions Calculator")
        
        # User provides emission factors
        st.write("### Enter Emission Factors (CO2 per unit output)")
        if 'fuel_type' in st.session_state.df.columns:
            fuels = st.session_state.df['fuel_type'].unique()
            emission_factors = {}
            
            for fuel in fuels:
                emission_factors[fuel] = st.number_input(
                    f"{fuel} (kgCO2/unit)", 
                    min_value=0.0, 
                    value=500.0,
                    key=f"em_factor_{fuel}")
            
            if st.button("Calculate Emissions"):
                output_col = st.selectbox("Select output column", st.session_state.df.select_dtypes(include=np.number).columns)
                st.session_state.df['emissions'] = st.session_state.df.apply(
                    lambda row: row[output_col] * emission_factors.get(row['fuel_type'], 0), axis=1)
                
                total_emissions = st.session_state.df['emissions'].sum()
                st.metric("Total CO2 Emissions", f"{total_emissions/1000:,.1f} tonnes")
                
                emissions_by_fuel = st.session_state.df.groupby('fuel_type')['emissions'].sum()
                st.bar_chart(emissions_by_fuel)
        else:
            emission_factor = st.number_input("Emission factor (kgCO2/unit)", min_value=0.0, value=500.0)
            output_col = st.selectbox("Select output column", st.session_state.df.select_dtypes(include=np.number).columns)
            if st.button("Calculate Emissions"):
                st.session_state.df['emissions'] = st.session_state.df[output_col] * emission_factor
                total_emissions = st.session_state.df['emissions'].sum()
                st.metric("Total CO2 Emissions", f"{total_emissions/1000:,.1f} tonnes")

                # 5. Multi-Fuel Fourier Forecasting
    with tab6:
        st.subheader("Fuel-Specific Energy Forecasting")
        
        # Auto-detect datetime column with unique key
        date_col = next((col for col in st.session_state.df.columns 
                        if 'date' in col.lower() or 'time' in col.lower()), None)
        
        if date_col and 'fuel_type' in st.session_state.df.columns:
            # Value selection with unique key
            value_col = st.selectbox("Select energy metric", 
                                   st.session_state.df.select_dtypes(include=np.number).columns,
                                   key="energy_metric_select")
            
            # Get unique fuel types
            fuel_types = st.session_state.df['fuel_type'].unique()
            
            # Let user select which fuel types to include
            selected_fuels = st.multiselect("Select fuel types to forecast",
                                          options=fuel_types,
                                          default=fuel_types[:min(3, len(fuel_types))],  # Show first 3 by default
                                          key="fuel_type_multiselect")
            
            # Forecast duration with unique key
            forecast_days = st.slider("Forecast duration (days)", 
                                    1, 7, 3,
                                    key="forecast_days_slider")
            
            if st.button("Generate Fuel-Specific Forecasts", key="multi_forecast_button"):
                with st.spinner("Analyzing fuel patterns..."):
                    try:
                        from scipy.optimize import curve_fit  # Import moved here to ensure it's available
                        
                        # Prepare figure
                        fig, ax = plt.subplots(figsize=(14, 7))
                        
                        # Dictionary to store forecasts
                        forecasts = {}
                        
                        # Process each selected fuel type
                        for fuel_type in selected_fuels:
                            # Prepare data
                            ts_data = (st.session_state.df[st.session_state.df['fuel_type'] == fuel_type]
                                      .set_index(date_col)[value_col]
                                      .resample('H').mean()
                                      .ffill()
                                      .clip(lower=0.01))
                            
                            # Only proceed if we have enough data
                            if len(ts_data) > 24:  # At least 1 day of data
                                hours = np.arange(len(ts_data))
                                y_values = ts_data.values
                                
                                # Fourier model with 3 harmonics
                                def fourier_model(x, a0, a1, b1, a2, b2, a3, b3):
                                    return (a0 + 
                                            a1 * np.cos(2 * np.pi * x / 24) + b1 * np.sin(2 * np.pi * x / 24) +
                                            a2 * np.cos(4 * np.pi * x / 24) + b2 * np.sin(4 * np.pi * x / 24) +
                                            a3 * np.cos(6 * np.pi * x / 24) + b3 * np.sin(6 * np.pi * x / 24))
                                
                                # Fit model
                                try:
                                    p0 = [np.mean(y_values), 0, 0, 0, 0, 0, 0]
                                    params, _ = curve_fit(fourier_model, hours, y_values, p0=p0, maxfev=10000)
                                    
                                    # Generate forecast
                                    forecast_hours = forecast_days * 24
                                    extended_hours = np.arange(len(hours) + forecast_hours)
                                    full_fit = fourier_model(extended_hours, *params).clip(min=0)
                                    
                                    # Get dates for plotting
                                    extended_dates = pd.date_range(
                                        start=ts_data.index[0],
                                        periods=len(extended_hours),
                                        freq='H'
                                    )
                                    
                                    # Split into historical and forecast periods
                                    historical_dates = extended_dates[:len(hours)]
                                    forecast_dates = extended_dates[len(hours):]
                                    historical_fit = full_fit[:len(hours)]
                                    forecast_fit = full_fit[len(hours):]
                                    
                                    # Plot actual data (last 3 days for clarity)
                                    plot_days = 3
                                    plot_start = ts_data.index[-1] - pd.Timedelta(days=plot_days)
                                    plot_data = ts_data[ts_data.index >= plot_start]
                                    ax.plot(plot_data.index, plot_data.values, 
                                           label=f"{fuel_type} Actual", alpha=0.7, linestyle='-')
                                    
                                    # Plot forecast
                                    ax.plot(forecast_dates, forecast_fit, 
                                           label=f"{fuel_type} Forecast", linestyle='--')
                                    
                                    # Store forecast for display
                                    forecasts[fuel_type] = pd.DataFrame({
                                        'timestamp': forecast_dates,
                                        'fourier_forecast': forecast_fit
                                    })
                                    
                                except Exception as e:
                                    st.warning(f"Could not generate forecast for {fuel_type}: {str(e)}")
                                    continue
                        
                        # Only proceed if we have at least one successful forecast
                        if forecasts:
                            # Formatting
                            ax.set_xlabel('Timestamp')
                            ax.set_ylabel(f'{value_col}')
                            ax.set_title(f'Energy Output: Actual vs Fourier Forecast by Fuel Type')
                            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(True, alpha=0.3)
                            
                            # Highlight forecast period
                            forecast_start = st.session_state.df[date_col].max()
                            ax.axvline(x=forecast_start, color='gray', linestyle=':')
                            ax.axvspan(forecast_start, extended_dates[-1], color='yellow', alpha=0.1)
                            
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Show forecast data tables
                            st.subheader("Forecast Details")
                            for fuel_type, forecast_df in forecasts.items():
                                with st.expander(f"{fuel_type} Forecast Data"):
                                    st.dataframe(forecast_df.set_index('timestamp').style.format("{:.2f}"))
                        else:
                            st.error("No successful forecasts generated for selected fuel types")
                            
                    except Exception as e:
                        st.error(f"Forecasting system error: {str(e)}")
        elif not date_col:
            st.warning("No datetime column found for forecasting")
        else:
            st.warning("No fuel_type column found - cannot generate fuel-specific forecasts")

        # 8. Energy Mix
    with tab7:
        st.subheader("Energy Mix Analysis")
        
        if 'fuel_type' in st.session_state.df.columns:
            mix_col = st.selectbox("Select metric for energy mix", 
                                 st.session_state.df.select_dtypes(include=np.number).columns)
            
            energy_mix = st.session_state.df.groupby('fuel_type')[mix_col].sum()
            
            # Filter out negative values and zero values
            energy_mix = energy_mix[energy_mix > 0]
            
            if not energy_mix.empty:
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Energy Mix by Volume")
                    fig1, ax1 = plt.subplots()
                    ax1.pie(energy_mix, labels=energy_mix.index, autopct='%1.1f%%')
                    st.pyplot(fig1)
                
                with col2:
                    if 'cost' in st.session_state.df.columns:
                        cost_mix = st.session_state.df.groupby('fuel_type')['cost'].sum()
                        # Filter out negative values and zero values for cost mix
                        cost_mix = cost_mix[cost_mix > 0]
                        
                        if not cost_mix.empty:
                            st.write("### Energy Mix by Cost")
                            fig2, ax2 = plt.subplots()
                            ax2.pie(cost_mix, labels=cost_mix.index, autopct='%1.1f%%')
                            st.pyplot(fig2)
                        else:
                            st.warning("No positive cost values available for selected fuel types")
                    else:
                        st.warning("No 'cost' column found for cost analysis")
            else:
                st.warning(f"No positive {mix_col} values available for selected fuel types")
        else:
            st.warning("No 'fuel_type' column found for energy mix analysis")

    # 10. Report Generator
    with tab8:
        st.subheader("Custom Report Generator")
        
        report_name = st.text_input("Report Name", "Energy Analysis Report")
        report_sections = st.multiselect("Select sections to include",
                                       ["Executive Summary", 
                                        "Time Period Analysis",
                                        "Efficiency Metrics",
                                        "Cost Breakdown",
                                        "Emissions Analysis",
                                        "Demand Forecast",
                                        "Energy Mix"])
        
        if st.button("Generate PDF Report"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                
                # Title
                pdf.cell(200, 10, txt=report_name, ln=1, align='C')
                pdf.ln(10)
                
                # Add selected sections
                if "Executive Summary" in report_sections:
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Executive Summary", ln=1, align='L')
                    pdf.set_font("Arial", size=12)
                    pdf.multi_cell(0, 10, txt="Key findings from the energy data analysis.")
                    pdf.ln(5)
                
                # Add other sections similarly...
                
                pdf.output(tmpfile.name)
                
                with open(tmpfile.name, "rb") as f:
                    st.download_button(
                        label="Download Report",
                        data=f,
                        file_name=f"{report_name.replace(' ', '_')}.pdf",
                        mime="application/pdf"
                    )
                
                os.unlink(tmpfile.name)
else:
    st.info("Please upload an energy data file to begin analysis")


