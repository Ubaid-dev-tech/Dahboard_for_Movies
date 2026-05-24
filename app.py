import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast

plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")
st.title("🎬 Movie Analytics Dashboard")
st.markdown("""
**Explore TMDB 5000 movies dataset with interactive filters.**
Analyze ratings, revenue, genres, trends, budget, popularity from 1916 to 2017.
Use the sidebar to filter by release date, rating, and genre.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['revenue'] = df['revenue'].replace(0,pd.NA)
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['budget'] = df['budget'].replace(0,pd.NA)
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notna(x) else [])
    df = df.dropna(subset=['vote_average', 'revenue', 'release_year', 'budget', 'popularity', 'runtime', 'release_date'])
    return df

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Filters")

default_start = df['release_date'].min().to_pydatetime()
default_end = df['release_date'].max().to_pydatetime()

# TWO SEPARATE DATE INPUTS - no tuple problem
start_date_input = st.sidebar.date_input("Start Date", value=default_start, min_value=default_start, max_value=default_end)
end_date_input = st.sidebar.date_input("End Date", value=default_end, min_value=default_start, max_value=default_end)

start_ts = pd.Timestamp(start_date_input)
end_ts = pd.Timestamp(end_date_input)

min_rating = st.sidebar.slider("Min Rating", 0.0, 10.0, 0.0, 0.1)
all_genres = sorted(list(set([g for sublist in df['genres'] for g in sublist])))
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

if st.sidebar.button("Reset Filters"):
    st.rerun()

# Apply Filters
df2 = df[
    (df['release_date'] >= start_ts) &
    (df['release_date'] <= end_ts) &
    (df['vote_average'] >= min_rating)
].copy()

if selected_genres:
    df2 = df2[df2['genres'].apply(lambda x: any(g in x for g in selected_genres))]

# KPIs
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies", len(df2))
col2.metric("Avg Rating", f"{df2['vote_average'].mean():.2f}")
col3.metric("Total Revenue", f"${df2['revenue'].sum()/1e9:.2f}B")
col4.metric("Avg Runtime", f"{df2['runtime'].mean():.0f} min")

st.divider()

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Top 10 Genres")
    genre_counts = df2.explode('genres')['genres'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, palette='viridis')
    ax.set_xlabel("Count")
    st.pyplot(fig); plt.close()

with col2:
    st.subheader("2. Rating Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df2['vote_average'], bins=20, kde=True, ax=ax, color='orange')
    ax.set_xlabel("Vote Average")
    st.pyplot(fig); plt.close()

col3, col4 = st.columns(2)
with col3:
    st.subheader("3. Revenue vs Rating")
    sample_df = df2.sample(min(500, len(df2)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample_df, x='vote_average', y=sample_df['revenue']/1e6, ax=ax, alpha=0.6)
    ax.set_xlabel("Vote Average"); ax.set_ylabel("Revenue (Million $)")
    st.pyplot(fig); plt.close()

with col4:
    st.subheader("4. Runtime by Rating")
    df2['rating_bin'] = pd.cut(df2['vote_average'], bins=[0, 5, 7, 8.5, 10], labels=['Low', 'Med', 'High', 'Top'])
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(data=df2, x='rating_bin', y='runtime', ax=ax, palette='coolwarm')
    ax.set_xlabel("Rating Range"); ax.set_ylabel("Runtime (min)")
    st.pyplot(fig); plt.close()

col5, col6 = st.columns(2)
with col5:
    st.subheader("5. Movies per Year")
    year_counts = df2['release_year'].value_counts().sort_index().tail(30)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(x=year_counts.index, y=year_counts.values, ax=ax, marker='o')
    ax.set_xlabel("Year"); ax.set_ylabel("Count")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close()

with col6:
    st.subheader("6. Budget vs Revenue")
    temp = df2[df2['budget'] > 0]
    sample_df = temp.sample(min(500, len(temp)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample_df, x=sample_df['budget']/1e6, y=sample_df['revenue']/1e6, ax=ax, alpha=0.6)
    ax.set_xlabel("Budget (Million $)"); ax.set_ylabel("Revenue (Million $)")
    st.pyplot(fig); plt.close()

col7, col8 = st.columns(2)
with col7:
    st.subheader("7. Popularity vs Rating")
    sample_df = df2.sample(min(500, len(df2)))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample_df, x='vote_average', y='popularity', ax=ax, alpha=0.6)
    ax.set_xlabel("Vote Average"); ax.set_ylabel("Popularity")
    st.pyplot(fig); plt.close()

with col8:
    st.subheader("8. Vote Count Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df2['vote_count'], bins=30, ax=ax, color='cyan')
    xlim_max = df2['vote_count'].quantile(0.95)
    if pd.isna(xlim_max):
        xlim_max = 1000
    ax.set_xlabel("Vote Count")
    ax.set_xlim(0, xlim_max)
    st.pyplot(fig)
    plt.close()

col9, col10 = st.columns(2)
with col9:
    st.subheader("9. Top 10 Languages")
    lang_counts = df2['original_language'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=lang_counts.values, y=lang_counts.index, ax=ax, palette='magma')
    ax.set_xlabel("Count")
    st.pyplot(fig); plt.close()

with col10:
    st.subheader("10. Revenue by Year")
    yearly_revenue = df2.groupby('release_year')['revenue'].sum().tail(30)/1e9
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=yearly_revenue.index, y=yearly_revenue.values, ax=ax, palette='rocket')
    ax.set_xlabel("Year"); ax.set_ylabel("Revenue (Billion $)")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close()

st.divider()
st.subheader("Top 10 Highest Rated Movies")
st.dataframe(df2.nlargest(10, 'vote_average')[['title','vote_average','revenue','release_year']], use_container_width=True)