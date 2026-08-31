# Bitcoin Quantile Model

This notebook explores the power laws and Quantile regressions to model Bitcoin pricing. This is purely an analytical excercise. Not financial advice. 

We use data from Kaggle which is updated regularly. Techniques explored here include

- Power Laws
- Regressions
- Quantile Regression
- Machine Learning
- Random Forest models
- Predictions

## Interactive quantile chart

`docs/index.html` is a self-contained interactive chart of all 99 fitted quantile
bands. Hover any date — historical or projected — to read the price at every
quantile, and see where the actual close sat in the distribution.

Rebuild it with fresh Kaggle data:

```bash
python build_chart.py
```

Options: `--csv PATH` to skip the Kaggle download, `--out PATH` to write elsewhere.
There's no server and no build step; open the output directly from disk. It does
need network access on first load to fetch the Plotly.js charting library from its
CDN.