import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
from collections import Counter
from itertools import combinations

# Title
st.title("🎲 EuroMillions Analysis Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload your EuroMillions CSV", type="csv")

# Game selection
game = st.selectbox("Select Lottery Game:", ["EuroMillions", "National Lottery"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Data", df.head())

    if game == "EuroMillions":
        main_cols = ['N1', 'N2', 'N3', 'N4', 'N5']
        star_cols = ['S1', 'S2']
        main_count = 5
        star_count = 2
        main_label = "main numbers"
        star_label = "Lucky Stars"
    else:  # National Lottery
        main_cols = ['N1', 'N2', 'N3', 'N4', 'N5', 'N6']
        star_cols = ['BN']  # Optional, can be []
        main_count = 6
        star_count = 1  # Set to 0 if you don't want to use Bonus
        main_label = "main numbers"
        star_label = "Bonus Ball"

    # Frequency Analysis
    st.subheader("📊 Frequency of Main Numbers")
    main_numbers = pd.concat([df[col] for col in main_cols])
    main_freq = main_numbers.value_counts().sort_index()
    fig, ax = plt.subplots()
    main_freq.plot(kind='bar', color='cornflowerblue', ax=ax)
    st.pyplot(fig)

    # Least Frequently Drawn Numbers
    st.subheader("⬇️ Least Frequently Drawn Numbers")
    least_freq = main_freq.sort_values().head(10)
    st.write(least_freq)
    fig_least, ax_least = plt.subplots()
    least_freq.plot(kind='bar', color='lightcoral', ax=ax_least)
    ax_least.set_title('Least Frequently Drawn Numbers')
    st.pyplot(fig_least)

    # Lucky Star Frequency
    st.subheader("⭐ Frequency of Lucky Stars")
    stars = pd.concat([df[col] for col in star_cols])
    star_freq = stars.value_counts().sort_index()
    fig2, ax2 = plt.subplots()
    star_freq.plot(kind='bar', color='gold', ax=ax2)
    st.pyplot(fig2)

    # Consecutive Number Check
    def has_consecutive(row):
        nums = sorted([row[col] for col in main_cols])
        return any(nums[i] + 1 == nums[i+1] for i in range(main_count - 1))

    df['Consecutive'] = df.apply(has_consecutive, axis=1)

    st.subheader("📋 Draws with Consecutive Numbers")
    st.write(df['Consecutive'].value_counts().rename({True: "Has Consecutive", False: "No Consecutive"}))

    # Visualize Consecutive Numbers
    fig3, ax3 = plt.subplots()
    df['Consecutive'].value_counts().plot(kind='bar', color=['mediumseagreen', 'salmon'], ax=ax3)
    ax3.set_xticklabels(['No Consecutive', 'Has Consecutive'], rotation=0)
    ax3.set_title('Draws with Consecutive Numbers')
    ax3.set_xlabel('Consecutive Numbers')
    ax3.set_ylabel('Number of Draws')
    st.pyplot(fig3)

    # Odd/Even Analysis
    def odd_even_ratio(row):
        nums = [row[col] for col in main_cols]
        odds = sum(1 for n in nums if n % 2 != 0)
        evens = main_count - odds
        return f"{odds} Odd / {evens} Even"

    df['OddEven'] = df.apply(odd_even_ratio, axis=1)

    st.subheader("🔢 Odd/Even Combinations per Draw")
    st.write(df['OddEven'].value_counts())

    # Plot Odd/Even Distribution
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    df['OddEven'].value_counts().plot(kind='bar', color='mediumseagreen', ax=ax4)
    ax4.set_title('Odd/Even Combinations per Draw')
    ax4.set_xlabel('Combination')
    ax4.set_ylabel('Number of Draws')
    ax4.grid(axis='y')
    st.pyplot(fig4)

    # Calculate overdue numbers for use in AI-Style suggestions
    all_numbers = pd.concat([
        df[[col]] for col in main_cols
    ])
    all_numbers = all_numbers.reset_index(drop=True)
    all_numbers = all_numbers.rename(columns={all_numbers.columns[0]: 'Number'})
    all_numbers['DrawIndex'] = all_numbers.index
    overdue = all_numbers.groupby('Number')['DrawIndex'].max().sort_values()

    # AI-Like Number Suggestion
    st.subheader("🎰 AI-Style Number Selection")

    strategy = st.selectbox(
        "Choose your number selection strategy:",
        ["Most Frequent", "Least Frequent", "Most Overdue", "Random"]
    )

    top_n = st.slider("Select top N numbers to pick from:", 5, 25, 10)

    if strategy == "Most Frequent":
        pool = main_freq.sort_values(ascending=False).head(top_n).index.tolist()
    elif strategy == "Least Frequent":
        pool = main_freq.sort_values(ascending=True).head(top_n).index.tolist()
    elif strategy == "Most Overdue":
        pool = list(overdue.index[-top_n:])
    else:  # Random
        pool = list(main_freq.index)

    selected_numbers = random.sample(pool, main_count)
    selected_stars = random.sample(list(star_freq.index), star_count)

    st.write("🎟️ **Suggested Numbers:**", selected_numbers)
    st.write("🌟 **Suggested Lucky Stars:**", selected_stars)

    # --- K-NN Analysis ---
    st.subheader("🔎 K-Nearest Neighbors (K-NN) Draw Finder")

    # User input for numbers
    user_numbers = st.multiselect(
        "Select 5 main numbers to search for similar draws:",
        options=sorted(main_numbers.unique()),
        default=sorted(main_numbers.unique())[:5]
    )

    user_stars = st.multiselect(
        "Select 2 Lucky Stars to search for similar draws:",
        options=sorted(stars.unique()),
        default=sorted(stars.unique())[:2]
    )

    k = st.slider("Number of nearest draws to find (k):", 1, 10, 3)

    if len(user_numbers) == 5 and len(user_stars) == 2:
        # Prepare data for K-NN
        X = df[main_cols + star_cols].values
        user_input = np.array(sorted(user_numbers) + sorted(user_stars)).reshape(1, -1)

        # Fit and find neighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, indices = nn.kneighbors(user_input)

        st.write(f"Top {k} most similar past draws:")
        st.dataframe(df.iloc[indices[0]])
    else:
        st.info("Please select exactly 5 main numbers and 2 Lucky Stars.")

    # After main_freq is calculated

    st.subheader("⏳ Most Overdue Numbers")
    # Combine all main numbers into a single column and track their draw index
    all_numbers = pd.concat([
        df[[col]] for col in main_cols
    ])
    all_numbers = all_numbers.reset_index(drop=True)
    all_numbers = all_numbers.rename(columns={all_numbers.columns[0]: 'Number'})
    all_numbers['DrawIndex'] = all_numbers.index
    overdue = all_numbers.groupby('Number')['DrawIndex'].max().sort_values()
    most_overdue = overdue.index[-10:]
    st.write("Most overdue numbers (not drawn recently):", list(most_overdue))

    st.subheader("🤝 Commonly Drawn Number Pairs")
    pairs = []
    for _, row in df.iterrows():
        nums = sorted([row[col] for col in main_cols])
        pairs.extend(combinations(nums, 2))
    pair_counts = Counter(pairs)
    common_pairs = pair_counts.most_common(10)
    st.write(pd.DataFrame(common_pairs, columns=['Pair', 'Count']))

    st.subheader("🔗 Commonly Drawn Number Triplets")
    triplets = []
    for _, row in df.iterrows():
        nums = sorted([row[col] for col in main_cols])
        triplets.extend(combinations(nums, 3))
    triplet_counts = Counter(triplets)
    common_triplets = triplet_counts.most_common(10)
    st.write(pd.DataFrame(common_triplets, columns=['Triplet', 'Count']))

    # --- Check User Numbers Against Past Draws ---
    st.subheader("🔍 Check Your Numbers Against Past Draws")

    user_main = st.multiselect(
        f"Enter your {main_count} {main_label}:",
        options=sorted(main_numbers.unique()),
        default=[]
    )
    if star_count > 0:
        user_star = st.multiselect(
            f"Enter your {star_count} {star_label}:",
            options=sorted(pd.concat([df[col] for col in star_cols]).unique()),
            default=[]
        )
    else:
        user_star = []

    if len(user_main) == 5 and len(user_star) == 2:
        def match_count(row):
            main_match = len(set(user_main) & {row[col] for col in main_cols})
            star_match = len(set(user_star) & {row[col] for col in star_cols})
            return pd.Series({'Main Matches': main_match, 'Star Matches': star_match})

        matches = df.apply(match_count, axis=1)
        result = pd.concat([df, matches], axis=1)
        st.write("Draws with at least 2 main numbers or 1 star matched:")
        st.dataframe(result[(result['Main Matches'] >= 2) | (result['Star Matches'] >= 1)])
    else:
        st.info("Please select exactly 5 main numbers and 2 Lucky Stars to check against past draws.")

st.info("Note: Lottery draws are random. Analyses and suggestions are for entertainment only and do not improve your chances of winning.")
