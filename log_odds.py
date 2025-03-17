import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import math

# Set page title and configuration
st.set_page_config(
    page_title="Cross-Language Log Odds Ratio Analysis",
    layout="wide"
)

# Function to calculate log odds ratio
def log_odds_ratio(p1, p2):
    epsilon = 0.0001
    p1 = max(epsilon, min(1-epsilon, p1))
    p2 = max(epsilon, min(1-epsilon, p2))
    return math.log(p1/(1-p1)) - math.log(p2/(1-p2))

# Create DataFrame directly from the data
data = [
    {"figure": "Charles de Gaulle", "language": "English", "topic": 0.6409044428, "sentiment": 0.5079147604, "english_entail": 0, "other_entail": 0.4333333333},
    {"figure": "Charles de Gaulle", "language": "French", "topic": 0.6287010653, "sentiment": 0.5197385677, "english_entail": 0, "other_entail": 0.5421666747},
    {"figure": "Che Guevara", "language": "English", "topic": 0.6459901542, "sentiment": 0.4931009415, "english_entail": 0.7666666667, "other_entail": 0},
    {"figure": "Che Guevara", "language": "Spanish", "topic": 0.6375195871, "sentiment": 0.4956889751, "english_entail": 0.7012987013, "other_entail": 0},
    {"figure": "Dwight Eisenhower", "language": "English", "topic": 0.576226824, "sentiment": 0.4646259933, "english_entail": None, "other_entail": None},
    {"figure": "Dwight Eisenhower", "language": "Russian", "topic": 0.5807404201, "sentiment": 0.5329799311, "english_entail": None, "other_entail": None},
    {"figure": "Fidel Castro", "language": "English", "topic": 0.5462618511, "sentiment": 0.5204083783, "english_entail": 0.6923076923, "other_entail": 0},
    {"figure": "Fidel Castro", "language": "Spanish", "topic": 0.5565106652, "sentiment": 0.4830995693, "english_entail": 0.6621621622, "other_entail": 0},
    {"figure": "Fulgencio Batista", "language": "English", "topic": 0.4729977154, "sentiment": 0.5148144753, "english_entail": 0.7710843373, "other_entail": 0},
    {"figure": "Fulgencio Batista", "language": "Spanish", "topic": 0.4902343397, "sentiment": 0.4580564687, "english_entail": 0.7323943662, "other_entail": 0},
    {"figure": "George Walker Bush", "language": "English", "topic": 0.6074363488, "sentiment": 0.5015480862, "english_entail": 0, "other_entail": 0},
    {"figure": "George Walker Bush", "language": "Spanish", "topic": 0.6062106784, "sentiment": 0.5111581227, "english_entail": 0, "other_entail": 0},
    {"figure": "JFK", "language": "English", "topic": 0.5328000448, "sentiment": 0.5148158514, "english_entail": 0, "other_entail": 0},
    {"figure": "JFK", "language": "Spanish", "topic": 0.5402872697, "sentiment": 0.5127989922, "english_entail": 0, "other_entail": 0},
    {"figure": "Joseph Stalin", "language": "English", "topic": 0.5800237365, "sentiment": 0.4753391545, "english_entail": 0, "other_entail": 0},
    {"figure": "Joseph Stalin", "language": "Russian", "topic": 0.5781851775, "sentiment": 0.4901529838, "english_entail": 0, "other_entail": 0},
    {"figure": "Lord Mountbatten", "language": "English", "topic": 0.5570490065, "sentiment": 0.5024207747, "english_entail": 0, "other_entail": 0},
    {"figure": "Lord Mountbatten", "language": "Hindi", "topic": 0.5541955168, "sentiment": 0.5038154085, "english_entail": 0, "other_entail": 0},
    {"figure": "Mahatma Gandhi", "language": "English", "topic": 0.5540970387, "sentiment": 0.4585445758, "english_entail": 0, "other_entail": 0},
    {"figure": "Mahatma Gandhi", "language": "Hindi", "topic": 0.5529752957, "sentiment": 0.4617332425, "english_entail": 0, "other_entail": 0},
    {"figure": "Mikhail Gorbachev", "language": "English", "topic": 0.5357400151, "sentiment": 0.4906770666, "english_entail": 0.6355140187, "other_entail": 0},
    {"figure": "Mikhail Gorbachev", "language": "Russian", "topic": 0.5292633243, "sentiment": 0.4922896589, "english_entail": 0.3977272727, "other_entail": 0},
    {"figure": "Napoleon Bonaparte", "language": "English", "topic": 0.4966125304, "sentiment": 0.4841329648, "english_entail": 0, "other_entail": 0},
    {"figure": "Napoleon Bonaparte", "language": "French", "topic": 0.5130937114, "sentiment": 0.4875206548, "english_entail": 0, "other_entail": 0},
    {"figure": "Nikita Khrushchev", "language": "English", "topic": 0.58490626, "sentiment": 0.460537951, "english_entail": 0, "other_entail": 0},
    {"figure": "Nikita Khrushchev", "language": "Russian", "topic": 0.5397152168, "sentiment": 0.489257677, "english_entail": 0, "other_entail": 0},
    {"figure": "Queen Victoria", "language": "English", "topic": 0.6264298717, "sentiment": 0.5510719637, "english_entail": 0, "other_entail": 0},
    {"figure": "Queen Victoria", "language": "Hindi", "topic": 0.6247159288, "sentiment": 0.4986571912, "english_entail": 0, "other_entail": 0},
    {"figure": "Richard Nixon", "language": "English", "topic": 0.6197548697, "sentiment": 0.4907644977, "english_entail": 0, "other_entail": 0.4513274336},
    {"figure": "Richard Nixon", "language": "Russian", "topic": 0.601643626, "sentiment": 0.5224370406, "english_entail": 0, "other_entail": 0.4594594595},
    {"figure": "Ronald Reagan", "language": "English", "topic": 0.5605031612, "sentiment": 0.5081455297, "english_entail": 0, "other_entail": 0},
    {"figure": "Ronald Reagan", "language": "Russian", "topic": 0.5582441939, "sentiment": 0.4814028049, "english_entail": 0, "other_entail": 0},
    {"figure": "Theodore Roosevelt", "language": "English", "topic": 0.6234051355, "sentiment": 0.5126769827, "english_entail": 0, "other_entail": 0.4186046512},
    {"figure": "Theodore Roosevelt", "language": "Spanish", "topic": 0.6158898381, "sentiment": 0.5360350107, "english_entail": 0, "other_entail": 0.4117647059},
    {"figure": "Vladimir Lenin", "language": "English", "topic": 0.5204199936, "sentiment": 0.499483545, "english_entail": 0.6857142857, "other_entail": 0},
    {"figure": "Vladimir Lenin", "language": "Russian", "topic": 0.4977741441, "sentiment": 0.5016730542, "english_entail": 0.5909090909, "other_entail": 0},
    {"figure": "Vladimir Putin", "language": "English", "topic": 0.5498654858, "sentiment": 0.4581484264, "english_entail": 0, "other_entail": 0},
    {"figure": "Vladimir Putin", "language": "Russian", "topic": 0.5340000101, "sentiment": 0.4498313837, "english_entail": 0, "other_entail": 0},
    {"figure": "Winston Churchill", "language": "English", "topic": 0.407112711, "sentiment": 0.4776186783, "english_entail": None, "other_entail": 0.5210084034},
    {"figure": "Winston Churchill", "language": "French", "topic": 0.4284260142, "sentiment": 0.4718842171, "english_entail": None, "other_entail": 0.5652173913}
]

df = pd.DataFrame(data)

# Get unique languages and calculate language pairs
unique_languages = df['language'].unique()

# Generate all language pairs
language_pairs = []
for i in range(len(unique_languages)):
    for j in range(i + 1, len(unique_languages)):
        language_pairs.append(f"{unique_languages[i]}-{unique_languages[j]}")

# Filter out French-Spanish since it has no data
language_pairs = [pair for pair in language_pairs if pair != "French-Spanish"]
language_pairs = language_pairs[:4]  # Take only the first 4 pairs

# Header
st.title("Cross-Language Log Odds Ratio Analysis")
st.markdown("""
    Visualizations showing log odds ratios for topic and sentiment scores across language pairs.
    Positive values (blue) indicate higher values in the first language, negative values (green) in the second language.
""")

# Language pair selector
selected_pair = st.radio(
    "Select Language Pair:",
    options=language_pairs,
    horizontal=True
)

# Split the selected pair
lang1, lang2 = selected_pair.split('-')

# Calculate log odds ratios for the selected pair
topic_results = []
sentiment_results = []

# Get unique figures
unique_figures = df['figure'].unique()

# Calculate log odds for each figure in the selected language pair
for figure in unique_figures:
    lang1_data = df[(df['figure'] == figure) & (df['language'] == lang1)]
    lang2_data = df[(df['figure'] == figure) & (df['language'] == lang2)]
    
    if not lang1_data.empty and not lang2_data.empty:
        # Calculate topic log odds
        topic_log_odds = log_odds_ratio(lang1_data['topic'].iloc[0], lang2_data['topic'].iloc[0])
        topic_results.append({
            'figure': figure,
            'log_odds': topic_log_odds
        })
        
        # Calculate sentiment log odds
        sentiment_log_odds = log_odds_ratio(lang1_data['sentiment'].iloc[0], lang2_data['sentiment'].iloc[0])
        sentiment_results.append({
            'figure': figure,
            'log_odds': sentiment_log_odds
        })

# Convert to DataFrames
topic_df = pd.DataFrame(topic_results)
sentiment_df = pd.DataFrame(sentiment_results)

# Calculate statistics
if not topic_df.empty:
    topic_mean = topic_df['log_odds'].mean()
    topic_median = topic_df['log_odds'].median()
    topic_min = topic_df['log_odds'].min()
    topic_max = topic_df['log_odds'].max()
else:
    topic_mean = topic_median = topic_min = topic_max = 0

if not sentiment_df.empty:
    sent_mean = sentiment_df['log_odds'].mean()
    sent_median = sentiment_df['log_odds'].median()
    sent_min = sentiment_df['log_odds'].min()
    sent_max = sentiment_df['log_odds'].max()
else:
    sent_mean = sent_median = sent_min = sent_max = 0

# Create tabs
tab1, tab2 = st.tabs(["Topic Score", "Sentiment Score"])

with tab1:
    st.subheader(f"Topic Score Log Odds Ratio: {lang1} vs {lang2}")
    
    st.markdown(f"""
    **Statistics**:
    - Mean: {topic_mean:.3f}
    - Median: {topic_median:.3f}
    - Range: [{topic_min:.3f}, {topic_max:.3f}]
    """)
    
    # Sort by log odds value
    topic_df_sorted = topic_df.sort_values('log_odds')
    
    # Create violin-style plot
    fig1 = go.Figure()
    
    # Add scatter points
    fig1.add_trace(go.Scatter(
        x=topic_df_sorted['log_odds'],
        y=topic_df_sorted['figure'],
        mode='markers',
        marker=dict(
            size=12,
            color=topic_df_sorted['log_odds'],
            colorscale=[[0, 'green'], [0.5, 'lightgray'], [1, 'blue']],
            colorbar=dict(title="Log Odds Ratio"),
            cmin=-max(abs(topic_min), abs(topic_max)),
            cmid=0,
            cmax=max(abs(topic_min), abs(topic_max))
        ),
        text=topic_df_sorted['figure'],
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    
    # Add vertical line at x=0
    fig1.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(topic_df_sorted) - 0.5,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig1.update_layout(
        height=max(400, len(topic_df_sorted) * 40),  # Adjust height based on number of points
        xaxis_title="Log Odds Ratio",
        yaxis_title="Historical Figure",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            zeroline=False,
            range=[min(topic_min * 1.1, -0.1), max(topic_max * 1.1, 0.1)]
        )
    )
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Add annotations
    st.markdown(f"""
    - **Positive values (blue)**: Higher topic scores in {lang1}
    - **Negative values (green)**: Higher topic scores in {lang2}
    - **Values near zero**: Similar topic scores in both languages
    """)
    
    # Display data table
    with st.expander("Show Data Table"):
        st.dataframe(
            topic_df.sort_values('log_odds', ascending=False)
            .rename(columns={'figure': 'Historical Figure', 'log_odds': 'Log Odds Ratio'})
            .reset_index(drop=True),
            hide_index=True
        )

with tab2:
    st.subheader(f"Sentiment Score Log Odds Ratio: {lang1} vs {lang2}")
    
    st.markdown(f"""
    **Statistics**:
    - Mean: {sent_mean:.3f}
    - Median: {sent_median:.3f}
    - Range: [{sent_min:.3f}, {sent_max:.3f}]
    """)
    
    # Sort by log odds value
    sentiment_df_sorted = sentiment_df.sort_values('log_odds')
    
    # Create violin-style plot
    fig2 = go.Figure()
    
    # Add scatter points
    fig2.add_trace(go.Scatter(
        x=sentiment_df_sorted['log_odds'],
        y=sentiment_df_sorted['figure'],
        mode='markers',
        marker=dict(
            size=12,
            color=sentiment_df_sorted['log_odds'],
            colorscale=[[0, 'green'], [0.5, 'lightgray'], [1, 'blue']],
            colorbar=dict(title="Log Odds Ratio"),
            cmin=-max(abs(sent_min), abs(sent_max)),
            cmid=0,
            cmax=max(abs(sent_min), abs(sent_max))
        ),
        text=sentiment_df_sorted['figure'],
        hovertemplate='%{y}: %{x:.3f}<extra></extra>'
    ))
    
    # Add vertical line at x=0
    fig2.add_shape(
        type="line",
        x0=0, y0=-0.5,
        x1=0, y1=len(sentiment_df_sorted) - 0.5,
        line=dict(color="black", width=1, dash="dash")
    )
    
    # Update layout
    fig2.update_layout(
        height=max(400, len(sentiment_df_sorted) * 40),  # Adjust height based on number of points
        xaxis_title="Log Odds Ratio",
        yaxis_title="Historical Figure",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            zeroline=False,
            range=[min(sent_min * 1.1, -0.1), max(sent_max * 1.1, 0.1)]
        )
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Add annotations
    st.markdown(f"""
    - **Positive values (blue)**: More positive sentiment in {lang1}
    - **Negative values (green)**: More positive sentiment in {lang2}
    - **Values near zero**: Similar sentiment in both languages
    """)
    
    # Display data table
    with st.expander("Show Data Table"):
        st.dataframe(
            sentiment_df.sort_values('log_odds', ascending=False)
            .rename(columns={'figure': 'Historical Figure', 'log_odds': 'Log Odds Ratio'})
            .reset_index(drop=True),
            hide_index=True
        )

# Add explanation in sidebar
with st.sidebar:
    st.title("About Log Odds Ratios")
    st.markdown("""
    ### What are Log Odds Ratios?
    
    The log odds ratio measures the relative difference in probabilities between two languages.
    
    ### Calculation
    
    ```
    logOddsRatio(p1, p2) = log(p1/(1-p1)) - log(p2/(1-p2))
    ```
    
    Where:
    - p1 is the proportion/score in the first language
    - p2 is the proportion/score in the second language
    
    ### Interpretation
    
    - **Positive values**: Higher proportion in the first language
    - **Negative values**: Higher proportion in the second language
    - **Values near zero**: Similar proportions in both languages
    
    ### Use Cases
    
    - **Topic Scores**: Measures differences in coverage or prominence
    - **Sentiment Scores**: Measures differences in positive/negative portrayal
    """)
