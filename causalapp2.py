import streamlit as st
import pandas as pd
from causalimpact import CausalImpact
import pandas as pd
import numpy as np
import warnings
import io
import contextlib
import re
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")

# Load the data
data = pd.read_csv('Data/final_merged_dataset.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Function to get unique non-zero values from a column
def get_unique_nonzero(data, column):
    return [x for x in data[column].dropna().unique() if x != 0 and x != '0' and x == x]

# Function to clean and get unique values for POV Article, considering case and whitespace
def get_unique_pov(data, column):
    unique_vals = set(val.strip().lower() for val in data[column].dropna().unique() if val.strip().lower() != '0')
    return list(unique_vals)


def perform_causal_impact_analysis(data, event_date_str, pre_days, post_days, selected_data_type):
    """
    Perform a Causal Impact analysis on provided data.

    Parameters:
    data (pd.DataFrame): The dataset containing the time series data and milestones.
    event_date_str (str): The specific event date in 'YYYY-MM-DD' format.
    pre_days (int): Number of days to include in the pre-intervention period.
    post_days (int): Number of days to include in the post-intervention period.

    Returns:
    ci (CausalImpact): The CausalImpact object after fitting the model.
    post_intervention_milestones (pd.DataFrame): Milestones during the post-intervention period.
    """

    # Convert the event date from string to datetime
    event_date = pd.to_datetime(event_date_str)

    empty_milestones_df = pd.DataFrame(columns=['Date', 'Event and related notes', 'Brief description'])


    # Define the pre-intervention and post-intervention periods
    pre_period_start = event_date - pd.Timedelta(days=pre_days)
    post_period_end = event_date + pd.Timedelta(days=post_days)

    if data[selected_data_type].isnull().all():
        return None, "Input response cannot have just Null values."    

    # Aggregate the sentiment data by date
    data_agg = data.groupby('Date').agg({
        'Positive': 'mean',
        'Negative': 'mean',
        'Neutral': 'mean',
        'Count': 'mean',
        'hpv_Positive': 'mean',
        'hpv_Neutral': 'mean',
        'hpv_Negative': 'mean',
        'hpv_Count': 'mean',
        'hpv_neu_neg': 'mean',
        'mmr_neu_neg': 'mean',
        'mmr_Positive': 'mean',
        'mmr_Neutral': 'mean',
        'mmr_Negative': 'mean',
        'mmr_Count': 'mean',
    }).reset_index()
    st.write("Aggregated Data:", data_agg)
    # Extract the data for the analysis period
    analysis_period_data = data_agg[
        (data_agg['Date'] >= pre_period_start) & 
        (data_agg['Date'] <= post_period_end)
    ].set_index('Date').asfreq('D').fillna(method='ffill')

    st.write("Analysis Period Data:", analysis_period_data)

    if analysis_period_data.empty or analysis_period_data[selected_data_type].isnull().all():
        return None, empty_milestones_df, "Not enough data for analysis in the selected period."

    # Prepare the data for CausalImpact
    analysis_data = analysis_period_data[selected_data_type].to_frame('y')

    # Define the pre and post intervention periods for CausalImpact
    pre_period = [str(pre_period_start.date()), str((event_date - pd.Timedelta(days=1)).date())]
    post_period = [str(event_date.date()), str(post_period_end.date())]

    try:
        ci = CausalImpact(analysis_data, pre_period, post_period)
            # Display the summary of the analysis
        print(ci.summary())
        print(ci.summary(output='report'))

        # Plot the results
        ci.plot()

        # Check for any significant milestones in the post-intervention days
        post_intervention_milestones = data[
            (data['Date'] > event_date) & 
            (data['Date'] <= post_period_end) &
            (data['Event and related notes'].notnull() | 
            data['Brief description'].notnull())
        ][['Date', 'Event and related notes', 'Brief description']]

        print(post_intervention_milestones)

        return ci, post_intervention_milestones, None
       # return ci, None
    except Exception as e:
        return None, empty_milestones_df, str(e)

    # Perform the CausalImpact analysis
    #ci = CausalImpact(analysis_data, pre_period, post_period)



# Function to capture the output of another function
def capture_output(function, *args, **kwargs):
    with io.StringIO() as buf, contextlib.redirect_stdout(buf):
        function(*args, **kwargs)
        return buf.getvalue()
    
def display_significance(p_value):
    if p_value <= 0.05:
            # Statistically significant
        color = "green"
        message = "Statistically significant (p <= 0.05)"
    else:
            # Not statistically significant
        color = "red"
        message = "Not statistically significant (p > 0.05)"

        # Display the message in color
    st.markdown(f"<span style='color: {color};'>{message}</span>", unsafe_allow_html=True)


def display_significance2(p_value):
    if p_value <= 0.05:
        st.success("Statistically significant (p <= 0.05)")
    else:
        st.error("Not statistically significant (p > 0.05)")


def extract_p_value_and_posterior_prob(result_str):
    pattern = r"Posterior tail-area probability p:\s*([0-9\.]+)\s*Posterior prob\. of a causal effect:\s*([0-9\.]+)%"
    match = re.search(pattern, result_str, re.DOTALL)  # re.DOTALL to match across multiple lines
    if match:
        p_value = float(match.group(1))
        posterior_prob = float(match.group(2))
        return p_value, posterior_prob
    else:
        return None, None
    

# Function to inject custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Inject your CSS with this line (assuming you have a CSS file)
local_css("style.css")

# Your main app
st.title("📚📱 Social Media and Public Milestone Causal Association Analysis App")


# Sidebar Filters
st.sidebar.header('🔍 Filters')

# Step 1: Select News Type
news_type = st.sidebar.radio("🗞️ Select News Type", ("News Articles", "Public Health Agencies"))

# Define columns to be displayed for each news type
public_health_columns = [
    'Date', 'Event and related notes', 'Vaccine ', 'Event type ', 'PH_Agency',
    'Our vaccine categorization', 'Population ', 'Expected public interest ', 
    'Impact on sentiment ', 'Impact on uptake_2'
]

news_article_columns = [
    'Date', 'Keyword', 'Source name', 'Original Source', 'URL', 
    'Lead (title)', 'Brief description', 'POV Article', 'POV Public Health', 
    'POV Anti-vax', 'Impact on Uptake', 'News Category', 'Date of Publication', 
    'Vaccine Type', 'Normalized Vaccine Lead Title', 'Region', 'Country', 'Comments'
]

# Filtering based on News Type and subsequent choices
if news_type == "News Articles":
    filtered_data = data[data['Lead (title)'].notna()]
    
    # Subsequent filters for News Articles
    vaccine_types = ['All'] + get_unique_nonzero(filtered_data, 'Normalized Vaccine Lead Title')
    selected_vaccine = st.sidebar.selectbox('💉 Select Vaccine Type', vaccine_types)

    news_categories = ['All'] + get_unique_nonzero(filtered_data, 'News Category')
    selected_news_category = st.sidebar.selectbox('📰 Select News Category', news_categories)

    pov_articles = ['All'] + get_unique_pov(filtered_data, 'POV Public Health')
    selected_pov_article = st.sidebar.selectbox('🏥 Select POV Public Health', pov_articles)

    # Apply filters
    if selected_vaccine != 'All':
        filtered_data = filtered_data[filtered_data['Normalized Vaccine Lead Title'] == selected_vaccine]
    if selected_news_category != 'All':
        filtered_data = filtered_data[filtered_data['News Category'] == selected_news_category]
    if selected_pov_article != 'All':
        filtered_data = filtered_data[filtered_data['POV Public Health'].str.strip().str.lower() == selected_pov_article.strip().lower()]

    # Select specific columns for News Articles
    filtered_data = filtered_data[news_article_columns]

elif news_type == "Public Health Agencies":
    filtered_data = data[data['Event and related notes'].notna()]
    
    # Subsequent filters for Public Health Agencies
    vax_types = ['All'] + list(filtered_data['Vaccine '].dropna().unique())
    selected_vaccine_type = st.sidebar.selectbox('💉 Select Vaccine Type', vax_types)

    event_types = ['All'] + list(filtered_data['Event type '].dropna().unique())
    selected_event_type = st.sidebar.selectbox('🚨 Select Event Type', event_types)

    PH_types = ['All'] + list(filtered_data['PH_Agency'].dropna().unique())
    selected_PH_type = st.sidebar.selectbox('🛡️ Select Public Health Agency', PH_types)

    populations = ['All'] + list(filtered_data['Population '].dropna().unique())
    selected_population = st.sidebar.selectbox('👥 Select Population', populations)

    public_interests = ['All'] + list(filtered_data['Expected public interest '].dropna().unique())
    selected_public_interest = st.sidebar.selectbox('🔍 Select Expected Public Interest', public_interests)

    # Apply filters
    if selected_vaccine_type != 'All':
        filtered_data = filtered_data[filtered_data['Vaccine '] == selected_vaccine_type]
    if selected_event_type != 'All':
        filtered_data = filtered_data[filtered_data['Event type '] == selected_event_type]
    if selected_PH_type != 'All':
        filtered_data = filtered_data[filtered_data['PH_Agency'] == selected_PH_type]
    if selected_population != 'All':
        filtered_data = filtered_data[filtered_data['Population '] == selected_population]
    if selected_public_interest != 'All':
        filtered_data = filtered_data[filtered_data['Expected public interest '] == selected_public_interest]

    # Select specific columns for Public Health Agencies
    filtered_data = filtered_data[public_health_columns]
    

# Display Filtered Data
st.write('📝 Filtered Data:')
st.dataframe(filtered_data, height=200)


if filtered_data.empty:
    st.error("No data available for the selected filters.")
else:

    event_column = 'Event and related notes' if news_type == "Public Health Agencies" else 'Lead (title)'

    st.markdown("📅  **Select an Event for Analysis**")
    selected_event = st.selectbox('Select an Event:', filtered_data[event_column].unique(), label_visibility='collapsed')


    #selected_event = st.selectbox('Select an Event for Analysis', filtered_data[event_column].unique())

    # Mapping of technical names to user-friendly descriptions
    social_media_data_mapping = {
        'Positive': 'Overall Positive Sentiment',
        'Negative': 'Overall Negative Sentiment',
        'Neutral': 'Overall Neutral Sentiment',
        'Count': 'Overall Sentiment Count',
        'hpv_Positive': 'HPV Vaccine Positive Sentiment',
        'hpv_Neutral': 'HPV Vaccine Neutral Sentiment',
        'hpv_Negative': 'HPV Vaccine Negative Sentiment',
        'hpv_Count': 'HPV Vaccine Sentiment Count',
        'hpv_neu_neg': 'HPV Vaccine Neutral and Negative Sentiment',
        'mmr_neu_neg': 'MMR Vaccine Neutral and Negative Sentiment',
        'mmr_Positive': 'MMR Vaccine Positive Sentiment',
        'mmr_Neutral': 'MMR Vaccine Neutral Sentiment',
        'mmr_Negative': 'MMR Vaccine Negative Sentiment',
        'mmr_Count': 'MMR Vaccine Sentiment Count'
    }

    # Convert this mapping to a list of tuples for the selectbox
    social_media_data_types = list(social_media_data_mapping.items())

    # Use the selectbox with the updated options
   # selected_data_type = st.selectbox('Select Social Media Data Type for Analysis', social_media_data_types, format_func=lambda x: x[1])

    # Access the selected data type key for analysis
    
    if selected_event:

        #selected_data_type = st.selectbox('Select Social Media Data Type for Analysis', social_media_data_types)
        st.markdown("**📈 Select Social Media Data Type for Analysis**")
        selected_data_type = st.selectbox('', social_media_data_types, format_func=lambda x: x[1])
        selected_data_type = selected_data_type[0]
        # Retrieve the event date
        event_date = pd.to_datetime(filtered_data[filtered_data[event_column] == selected_event]['Date'].iloc[0])

        pre_days = st.sidebar.slider('⏮️ Select number of Pre-Event Days', min_value=0, max_value=30, value=7)
        post_days = st.sidebar.slider('⏭️ Select number of Post-Event Days', min_value=0, max_value=30, value=7)

        

        earliest_date = data['Date'].min()
        latest_date = data['Date'].max()

        if event_date - pd.Timedelta(days=pre_days) < earliest_date:
            st.error("⚠️ Not enough pre-event data available for the selected event date. Please select a different event or reduce the number of pre-event days.")
        elif event_date + pd.Timedelta(days=post_days) > latest_date:
            st.error("⚠️ Not enough post-event data available for the selected event date. Please select a different event or reduce the number of post-event days.")
        else:
            st.write(f"Performing analysis for event on {event_date}, with {pre_days} pre days and {post_days} post days")

        ci, post_intervention_milestones, error_message = perform_causal_impact_analysis(data, event_date.strftime('%Y-%m-%d'), pre_days, post_days, selected_data_type)

        if error_message:
            st.error(f"Error: {error_message}")
        else:
            # Proceed with displaying the analysis results
            st.success("✅ Analysis Completed Successfully")
            

            st.header("📊🔍 Analysis Results")
            # Display Causal Impact Analysis Report
            st.write("📊 Causal Impact Analysis Result:")
            summary_str = ci.summary()
            #st.text(summary_str)
            #summary_str = ci.summary()
                    # Extract and display p-value and posterior probability
            p_value, posterior_prob = extract_p_value_and_posterior_prob(summary_str)
            if p_value is not None:
                display_significance2(p_value)
            else:
                st.write("P-value not available")
            # if p_value is not None:
            #     display_significance(p_value)
            # else:
            #     st.write("P-value not available.")


            import pandas as pd
            import re

            # Assume `summary_str` is your CausalImpact summary string

            # Parse the summary string for the actual and predicted averages
            actual_avg = re.search(r"Actual\s+([\d\.]+)", summary_str).group(1)
            predicted_avg = re.search(r"Prediction \(s\.d\.\)\s+([\d\.]+)", summary_str).group(1)
            p_value = re.search(r"Posterior tail-area probability p:\s+([\d\.]+)", summary_str).group(1)
            causal_effect_prob = re.search(r"Posterior prob\. of a causal effect:\s+([\d\.]+)%", summary_str).group(1)

            # Make a decision based on p-value
            statistical_decision = "Causal effect is Significant" if float(p_value) <= 0.05 else "Not Significant"

            # Create a DataFrame
            df1 = pd.DataFrame({
                "Date": event_date,  # Example date, replace with actual date
                "Milestone": selected_event,  # Example event, replace with actual event name
                "Actual Average": [actual_avg],
                "Predicted Average": [predicted_avg],
                "Statistical Decision": [statistical_decision]
            })

            print(df1)
            
            html1 = df1.to_html(escape=False, index=False)

            # Display the HTML in Streamlit
            st.markdown(html1, unsafe_allow_html=True)  
            st.markdown('<sub>Actual Average: actual average of sentiment counts for post-intervention period.</sub>'
                '<br><sub>Predicted Average: predicted average of sentiment counts if no milestone(s) occurred.</sub>', unsafe_allow_html=True)



            st.write("")
            st.header("🔖 Post Intervention Milestones")
        # Optional: Display Post Intervention Milestones
            st.write("Post Intervention Milestones:")
            post_intervention_milestones = post_intervention_milestones.replace([0, '0', np.nan], 'N/A')

    # Now, display the DataFrame without truncation
    # Using st.dataframe to display as an interactive table
            #st.dataframe(post_intervention_milestones)

            post_intervention_milestones = post_intervention_milestones.rename(columns={
                'Event and related notes': 'Public Health Agency Vax News',
                'Brief description': 'Public Media Vax News Articles'
            })

            # Convert the renamed DataFrame to HTML
            html = post_intervention_milestones.to_html(escape=False, index=False)

            # Display the HTML in Streamlit
            st.markdown(html, unsafe_allow_html=True)        
            
        #  html = post_intervention_milestones.to_html(escape=False, index=False)
        #  st.markdown(html, unsafe_allow_html=True)

import plotly.graph_objects as go
import pandas as pd
import plotly.express as px

# Assuming 'data' and 'post_intervention_milestones' are already loaded
# 'selected_data_type', 'selected_event', and 'event_date' are set based on user selection
if 'event_date' in locals() or 'event_date' in globals():
    start_date = event_date - pd.Timedelta(days=15)
    end_date = event_date + pd.Timedelta(days=15)
    filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

    if filtered_data.empty:
        st.write("No data available for the selected period.")
    else:
            # Create the plot
        y_range = [filtered_data[selected_data_type].min(), filtered_data[selected_data_type].max()]

            # Create the plot
        fig = px.line(filtered_data, x='Date', y=selected_data_type, title=f'Trend for {selected_data_type} Sentiments Around the Milestone',
                        labels={selected_data_type: 'Value', 'Date': 'Date'}, markers=True)

            # Add a vertical line for the selected event with a dummy trace for the legend
        fig.add_trace(go.Scatter(x=[event_date, event_date], y=y_range, mode="lines", name="Selected Milestone",
                                    line=dict(color="red", width=2, dash="dash")))

            # Add an annotation for the selected event
    # Add an annotation for the selected event with increased font size and italicized text
        fig.add_annotation(
            x=event_date, 
            y=filtered_data[filtered_data['Date'] == event_date][selected_data_type].max(),
            text=f"<i>{selected_event}</i>",  # Italicize text with <i> HTML tag
            showarrow=True, 
            arrowhead=5,
            arrowsize=2,
            ax=-40, 
            ay=-40,
            font=dict(
                size=14,  # Increase font size as needed
                color="black",  # You can specify font color if needed
                family="Arial, sans-serif"  # Specify font family if desired
            )
        )


        # Add markers for post-intervention milestones with a dummy trace for legend
        added_post_intervention_legend = False
        post_intervention_dates = []
        for index, row in post_intervention_milestones.iterrows():
            milestone_date = row['Date']
            if row['Public Health Agency Vax News'] != 'N/A' or row['Public Media Vax News Articles'] != 'N/A':
                # Only plot if within the filtered date range
                if start_date <= milestone_date <= end_date:
                    post_intervention_dates.append(milestone_date)
                    if not added_post_intervention_legend:  # Add legend entry once
                        fig.add_trace(go.Scatter(x=[milestone_date], y=[filtered_data[selected_data_type].max()], mode="lines", name="Post-intervention Milestone",
                                                line=dict(color="limegreen", width=1, dash="dot")))
                        added_post_intervention_legend = True

        # Add the post-intervention vertical lines outside the loop
        for date in post_intervention_dates:
            fig.add_vline(x=date, line_width=1, line_dash="dot", line_color="limegreen")

        # Set plot layout
        fig.update_layout(
            title=f'Trend for {selected_data_type} Sentiments Around the Milestone',
            xaxis_title='Date',
            yaxis_title='Count',
            legend_title='Legend',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(211,211,211,0.5)'),  # Lighter grid lines with lower alpha
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.write("")
        st.header("📈 Sentiment Trend Plot Around Milestone")
        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.header('Causal Impact Analysis Report')
        st.write('📈 Causal Impact Analysis Report:')

        if error_message:
            st.error(f"Error: {error_message}")
        else:

            st.text(summary_str)
            st.text(ci.summary(output='report'))



################################################################
import pandas as pd
import matplotlib.pyplot as plt # required later
from causalimpact import CausalImpact

import streamlit as st

df = pd.DataFrame({'date': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06'],

                   'value': [1539, 1696, 1427, 1341, 1426, 1471]})

@st.experimental_memo
def get_pre_post(data, change):  
    pre_start = min(data.index)
    pre_end = int(data[data['date'] == change].index.values)
    post_start = pre_end + 1
    post_end = max(data.index)
    pre_period = [pre_start, pre_end]
    post_period = [post_start, post_end]

    return pre_period, post_period

change = '2021-01-05'

pre_period, post_period = get_pre_post(df, change)

ci = CausalImpact(df.drop(['date'], axis=1), pre_period, post_period, prior_level_sd=None)

ci.plot()

fig = plt.gcf() # to get current figure
ax = plt.gca() # to get current axes

st.pyplot(fig)
