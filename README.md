# 🎲 Lottery Analysis Dashboard

A Streamlit dashboard for analyzing EuroMillions and UK National Lottery (Lotto) draws.  
Upload your CSV file and explore statistics, patterns, and AI-style number suggestions!

---

## Features

- **Supports EuroMillions and National Lottery (Lotto)**
- Frequency analysis of main and star/bonus numbers
- Least frequently drawn numbers
- Most overdue numbers
- Odd/Even and consecutive number pattern analysis
- Commonly drawn pairs and triplets
- K-Nearest Neighbors (K-NN) for similar draw search
- AI-style number suggestions (most frequent, least frequent, most overdue, random)
- Check your numbers against all past draws

---

## How to Use

1. **Install requirements:**
    ```
    pip install streamlit pandas matplotlib scikit-learn
    ```

2. **Prepare your CSV file:**
    - For **EuroMillions**: Columns should be `N1`, `N2`, `N3`, `N4`, `N5`, `S1`, `S2`
    - For **National Lottery**: Columns should be `N1`, `N2`, `N3`, `N4`, `N5`, `N6`, `BN`
    - Each row = one draw

3. **Run the app:**
    ```
    streamlit run test.py
    ```

4. **Upload your CSV** and select the game type.

5. **Explore the dashboard!**

---

## Notes

- All analyses are for entertainment only. Lottery draws are random and past results do not influence future draws.
- For best results, use the correct CSV format for your chosen game.

---

## Example CSV Format

**EuroMillions:**
| N1 | N2 | N3 | N4 | N5 | S1 | S2 |
|----|----|----|----|----|----|----|
|  1 |  5 | 12 | 23 | 45 |  2 |  8 |

**National Lottery:**
| N1 | N2 | N3 | N4 | N5 | N6 | BN |
|----|----|----|----|----|----|----|
|  3 | 11 | 22 | 34 | 40 | 49 | 17 |

---

## Disclaimer

> This tool is for statistical exploration and fun only.  
> It does **not** improve your chances of winning the lottery.

---

## License

MIT License