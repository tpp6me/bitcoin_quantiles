# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a Bitcoin Quantile Power Law Analysis project that explores the relationship between Bitcoin's price and time using quantile regression techniques. The analysis uses historical Bitcoin price data from Kaggle to model price movements and make predictions.

## Key Components

- **bitcoin_quantile_studies_daily_kaggle_data.ipynb**: Main analysis notebook containing the complete Bitcoin power law and quantile regression study
- **requirements.txt**: Python dependencies for the project

## Development Environment

This is a Python-based data science project that uses Jupyter notebooks for analysis and visualization.

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### Running the Analysis
```bash
jupyter notebook bitcoin_quantile_studies_daily_kaggle_data.ipynb
```

## Data Source and Processing

- Uses Bitcoin historical data from Kaggle (mczielinski/bitcoin-historical-data dataset)
- Data is automatically downloaded using `kagglehub` library
- Minute-level data is aggregated to daily prices for analysis
- Bitcoin genesis date is set to 2010-01-03 for time-based calculations

## Analysis Framework

### Core Techniques
- **Power Law Analysis**: Uses log-log plots to identify linear relationships in Bitcoin price growth
- **Quantile Regression**: Models different percentiles (1st to 99th) of price distributions using statsmodels
- **Machine Learning**: Random Forest classifiers predict 5% price movements in 7-day windows
- **Halving Cycle Analysis**: Incorporates Bitcoin halving dates to model cyclical behavior

### Key Features
- Computes days since Bitcoin genesis and last halving event
- Maps each price observation to its closest quantile position
- Generates price predictions for different quantile levels
- Creates binary classification targets for significant price movements (±5% in next 7 days)

### Visualization
- Log-log plots showing power law relationships
- Quantile regression lines overlaid on price data
- Both logarithmic and normal scale visualizations

## Model Architecture

The predictive models use three main features:
- Current quantile position (0.01-0.99)
- Days since last halving event
- 7-day moving average of quantile positions

Random Forest classifiers are trained to predict:
- Probability of 5%+ price increase in next 7 days
- Probability of 5%+ price decrease in next 7 days

## Key Libraries

- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib**: Visualization and plotting
- **statsmodels**: Quantile regression analysis
- **scikit-learn**: Machine learning models and preprocessing
- **kagglehub**: Automated data downloading from Kaggle