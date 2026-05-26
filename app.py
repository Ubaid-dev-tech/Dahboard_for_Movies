import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ast
from datetime import datetime

plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

st.set_page_config(page_title="Movie Analytics Dashboard", layout="wide")
st.title("🎬 Movie Analytics Dashboard")
st.markdown("""
**Explore TMDB 5000 movies dataset with interactive filters.**
Analyze ratings, revenue, genres, trends, budget, popularity from 1916 to 2017.
Use the sidebar to filter by release date, rating, genre, budget, revenue, popularity, and keyword.
""")

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb_5000_movies.csv")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['vote_count'] = pd.to_numeric(df['vote_count'], errors='coerce')
    df['genres'] = df['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notna(x) else [])
    df['keywords_list'] = df['keywords'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notna(x) else []
    ) if 'keywords' in df.columns else [[] for _ in range(len(df))]
    df = df.dropna(subset=['vote_average', 'revenue', 'release_year', 'budget', 'popularity', 'runtime', 'release_date'])
    return df

df = load_data()

# ─────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
st.sidebar.markdown("All filters apply to **all charts simultaneously**.")

# 1. DATE / TIME RANGE FILTER
st.sidebar.subheader("📅 Date Range")
default_start = df['release_date'].min().to_pydatetime()
default_end   = df['release_date'].max().to_pydatetime()
start_date_input = st.sidebar.date_input("Start Date", value=default_start, min_value=default_start, max_value=default_end)
end_date_input   = st.sidebar.date_input("End Date",   value=default_end,   min_value=default_start, max_value=default_end)
start_ts = pd.Timestamp(start_date_input)
end_ts   = pd.Timestamp(end_date_input)

# 2. SEARCH / TEXT FILTER
st.sidebar.subheader("🔎 Search / Text Filter")
search_query = st.sidebar.text_input("Search by movie title or keyword", placeholder="e.g. Batman, love, space...")

# 3. CATEGORY FILTER — original language dropdown
st.sidebar.subheader("🌐 Category Filter")
LANGUAGE_NAMES = {
    'af': 'Afrikaans', 'ar': 'Arabic', 'bn': 'Bengali', 'ca': 'Catalan',
    'cn': 'Cantonese', 'cs': 'Czech', 'cy': 'Welsh', 'da': 'Danish',
    'de': 'German', 'el': 'Greek', 'en': 'English', 'eo': 'Esperanto',
    'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque', 'fa': 'Persian (Farsi)',
    'fi': 'Finnish', 'fr': 'French', 'gl': 'Galician', 'gu': 'Gujarati',
    'he': 'Hebrew', 'hi': 'Hindi', 'hr': 'Croatian', 'hu': 'Hungarian',
    'hy': 'Armenian', 'id': 'Indonesian', 'is': 'Icelandic', 'it': 'Italian',
    'ja': 'Japanese', 'ka': 'Georgian', 'kn': 'Kannada', 'ko': 'Korean',
    'ku': 'Kurdish', 'lb': 'Luxembourgish', 'lt': 'Lithuanian', 'lv': 'Latvian',
    'mk': 'Macedonian', 'ml': 'Malayalam', 'mn': 'Mongolian', 'mr': 'Marathi',
    'ms': 'Malay', 'nb': 'Norwegian', 'nl': 'Dutch', 'no': 'Norwegian',
    'pa': 'Punjabi', 'pl': 'Polish', 'ps': 'Pashto', 'pt': 'Portuguese',
    'ro': 'Romanian', 'ru': 'Russian', 'sh': 'Serbo-Croatian', 'sk': 'Slovak',
    'sl': 'Slovenian', 'sq': 'Albanian', 'sr': 'Serbian', 'sv': 'Swedish',
    'sw': 'Swahili', 'ta': 'Tamil', 'te': 'Telugu', 'th': 'Thai',
    'tl': 'Filipino', 'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu',
    'vi': 'Vietnamese', 'wo': 'Wolof', 'xx': 'Unknown', 'yo': 'Yoruba',
    'zh': 'Chinese (Mandarin)', 'zu': 'Zulu',
}
all_lang_codes = sorted(df['original_language'].dropna().unique().tolist())
lang_display_options = {
    LANGUAGE_NAMES.get(code, code.upper()) + f" ({code})": code
    for code in all_lang_codes
}
selected_lang_labels = st.sidebar.multiselect(
    "Filter by Original Language",
    options=sorted(lang_display_options.keys()),
    default=[],
    help="Select one or more languages"
)
selected_languages = [lang_display_options[label] for label in selected_lang_labels]

# 4. MULTI-SELECT FILTER — genres
st.sidebar.subheader("🎭 Genre Multi-Select")
all_genres = sorted(list(set([g for sublist in df['genres'] for g in sublist])))
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

# 5. NUMERICAL RANGE — typed number inputs
budget_max_raw   = float(df['budget'].max())
revenue_max_raw  = float(df['revenue'].max())
pop_min_raw      = float(df['popularity'].min())
pop_max_raw      = float(df['popularity'].max())
runtime_min_raw  = float(df['runtime'].min())
runtime_max_raw  = float(df['runtime'].max())

st.sidebar.subheader("⭐ Rating Range")
r_col1, r_col2 = st.sidebar.columns(2)
min_rating = r_col1.number_input("Min Rating", min_value=0.0, max_value=10.0, value=0.0, step=0.1, format="%.1f")
max_rating = r_col2.number_input("Max Rating", min_value=0.0, max_value=10.0, value=10.0, step=0.1, format="%.1f")

st.sidebar.subheader("💰 Budget (Million $)")
b_col1, b_col2 = st.sidebar.columns(2)
budget_min_input = b_col1.number_input("Min Budget", min_value=0.0, value=0.0, step=1.0, format="%.1f")
budget_max_input = b_col2.number_input("Max Budget", min_value=0.0, value=round(budget_max_raw / 1e6, 1), step=1.0, format="%.1f")
budget_range = (budget_min_input, budget_max_input)

st.sidebar.subheader("🎯 Revenue (Million $)")
rv_col1, rv_col2 = st.sidebar.columns(2)
revenue_min_input = rv_col1.number_input("Min Revenue", min_value=0.0, value=0.0, step=1.0, format="%.1f")
revenue_max_input = rv_col2.number_input("Max Revenue", min_value=0.0, value=round(revenue_max_raw / 1e6, 1), step=1.0, format="%.1f")
revenue_range = (revenue_min_input, revenue_max_input)

st.sidebar.subheader("🔥 Popularity")
p_col1, p_col2 = st.sidebar.columns(2)
pop_min_input = p_col1.number_input("Min Pop", min_value=0.0, value=round(pop_min_raw, 1), step=1.0, format="%.1f")
pop_max_input = p_col2.number_input("Max Pop", min_value=0.0, value=round(pop_max_raw, 1), step=1.0, format="%.1f")
popularity_range = (pop_min_input, pop_max_input)

st.sidebar.subheader("⏱️ Runtime (min)")
rt_col1, rt_col2 = st.sidebar.columns(2)
runtime_min_input = rt_col1.number_input("Min Min", min_value=0, value=int(runtime_min_raw), step=1)
runtime_max_input = rt_col2.number_input("Max Min", min_value=0, value=int(runtime_max_raw), step=1)
runtime_range = (runtime_min_input, runtime_max_input)

# 6. RESET / CLEAR FILTERS
st.sidebar.divider()
if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
    st.rerun()

# ─────────────────────────────────────────────
# APPLY ALL FILTERS — connected to ALL charts
# ─────────────────────────────────────────────
df2 = df[
    (df['release_date'] >= start_ts) &
    (df['release_date'] <= end_ts) &
    (df['vote_average'] >= min_rating) &
    (df['vote_average'] <= max_rating) &
    (df['budget']     >= budget_range[0]   * 1e6) &
    (df['budget']     <= budget_range[1]   * 1e6) &
    (df['revenue']    >= revenue_range[0]  * 1e6) &
    (df['revenue']    <= revenue_range[1]  * 1e6) &
    (df['popularity'] >= popularity_range[0]) &
    (df['popularity'] <= popularity_range[1]) &
    (df['runtime']    >= runtime_range[0]) &
    (df['runtime']    <= runtime_range[1])
].copy()

# Genre multi-select
if selected_genres:
    df2 = df2[df2['genres'].apply(lambda x: any(g in x for g in selected_genres))]

# Language category filter
if selected_languages:
    df2 = df2[df2['original_language'].isin(selected_languages)]

# Search / text filter — matches title OR keywords
if search_query.strip():
    q = search_query.strip().lower()
    title_mask = df2['title'].str.lower().str.contains(q, na=False)
    kw_mask = df2['keywords_list'].apply(lambda kws: any(q in kw.lower() for kw in kws))
    df2 = df2[title_mask | kw_mask]

# ─────────────────────────────────────────────
# ACTIVE FILTERS SUMMARY BADGE
# ─────────────────────────────────────────────
active_filters = []
if search_query.strip():
    active_filters.append(f"🔎 \"{search_query.strip()}\"")
if selected_languages:
    active_filters.append(f"🌐 {', '.join(selected_languages)}")
if selected_genres:
    active_filters.append(f"🎭 {', '.join(selected_genres)}")
if min_rating > 0.0 or max_rating < 10.0:
    active_filters.append(f"⭐ {min_rating}–{max_rating}")
if budget_range != (0.0, round(budget_max_raw / 1e6, 1)):
    active_filters.append(f"💰 ${budget_range[0]}M–${budget_range[1]}M")
if revenue_range != (0.0, round(revenue_max_raw / 1e6, 1)):
    active_filters.append(f"🎯 ${revenue_range[0]}M–${revenue_range[1]}M")

if active_filters:
    st.info("**Active Filters:** " + " | ".join(active_filters) + f"  →  **{len(df2):,} movies**")
else:
    st.success(f"No filters active — showing all **{len(df2):,} movies**")

# ─────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Movies",  f"{len(df2):,}")
col2.metric("Avg Rating",    f"{df2['vote_average'].mean():.2f}" if len(df2) else "N/A")
col3.metric("Total Revenue", f"${df2['revenue'].sum()/1e9:.2f}B" if len(df2) else "N/A")
col4.metric("Avg Runtime",   f"{df2['runtime'].mean():.0f} min" if len(df2) else "N/A")

st.divider()

# Guard: show a message if no data
if df2.empty:
    st.warning("⚠️ No movies match the current filters. Please adjust the sidebar filters.")
    st.stop()

# ─────────────────────────────────────────────
# CHARTS — all driven by df2
# ─────────────────────────────────────────────
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
    sample_df = df2.sample(min(500, len(df2)), random_state=42)
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
    budget_df = df2[df2['budget'] > 0]
    sample_df = budget_df.sample(min(500, len(budget_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=sample_df, x=sample_df['budget']/1e6, y=sample_df['revenue']/1e6, ax=ax, alpha=0.6)
    ax.set_xlabel("Budget (Million $)"); ax.set_ylabel("Revenue (Million $)")
    st.pyplot(fig); plt.close()

col7, col8 = st.columns(2)
with col7:
    st.subheader("7. Popularity vs Rating")
    sample_df = df2.sample(min(500, len(df2)), random_state=42)
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
    st.pyplot(fig); plt.close()

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
    yearly_revenue = df2.groupby('release_year')['revenue'].sum().tail(30) / 1e9
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=yearly_revenue.index, y=yearly_revenue.values, ax=ax, palette='rocket')
    ax.set_xlabel("Year"); ax.set_ylabel("Revenue (Billion $)")
    plt.xticks(rotation=45)
    st.pyplot(fig); plt.close()

st.divider()

# ─────────────────────────────────────────────
# FILTERED DATA TABLE
# ─────────────────────────────────────────────
st.subheader("🏆 Top 10 Highest Rated Movies (filtered)")
display_cols = ['title', 'vote_average', 'revenue', 'budget', 'popularity', 'runtime',
                'release_year', 'original_language']
display_cols = [c for c in display_cols if c in df2.columns]
st.dataframe(
    df2.nlargest(10, 'vote_average')[display_cols].reset_index(drop=True),
    use_container_width=True
)

st.subheader("📋 Full Filtered Dataset")
st.caption(f"Showing {len(df2):,} movies matching all current filters")
st.dataframe(
    df2[display_cols].reset_index(drop=True),
    use_container_width=True,
    height=300
)