# Import of the libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


# plotting the index, moving average, difference and percent_change
def data_plot(df_index):
    """

    :param df_index:
    :return:
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 6))
    df_index.plot(kind="line", color="red", ax=ax1)
    df_index.diff(axis=0, periods=1).add_suffix("_change").plot(kind="line", color="blue", ax=ax2)
    df_index.pct_change().add_suffix("_%change").plot(kind="line", color="green", ax=ax3)
    df_index.rolling(window=6).mean().add_suffix("_6M_ma").plot(kind="line", color="blue", ax=ax1)
    plt.show()


def correlation(df_features, df_index):
    """
    Correlation heatmap
    :param df_features:
    :param df_index:
    :return:
    """
    df = pd.concat([df_features, df_index], axis=1)
    corr_matrix = df.corr()

    plt.figure(figsize=(16, 16))
    heatmap = sn.heatmap(corr_matrix, vmin=-1, vmax=1, annot=False, cmap="RdBu")
    heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 18}, pad=12)
    plt.show()

    def correl(X_train, threes):
        """
        display highly correlated features
        :param X_train:
        :param threes:
        :return:
        """
        cor = X_train.corr()
        correlation_matrix = np.corrcoef(X_train.transpose())
        corr = correlation_matrix - np.diagflat(correlation_matrix.diagonal())
        print("max corr:", corr.max(), ", min corr: ", corr.min())
        c1 = cor.stack().sort_values(ascending=False).drop_duplicates()
        high_correlation = c1[c1.values != 1]
        print(high_correlation[high_correlation > threes])
        return high_correlation[high_correlation > threes]

    high_cor = correl(df, 0.96)
    return high_cor


# Feature Selection
def boruta_feature_selection(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    # initialize Boruta
    forest = RandomForestRegressor(n_jobs=-1, max_depth=5)
    boruta = BorutaPy(estimator=forest, n_estimators="auto", max_iter=100)  # number of trials to perform

    # fit Boruta (it accepts np.array, not pd.DataFrame)
    boruta.fit(np.array(X), np.array(y))

    # print results
    green_area = X.columns[boruta.support_].to_list()

    return green_area


# Feature Selection
def rfe_feature_selection(X, y):
    """

    :param X:
    :param y:
    :return:
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import RFE

    # Selecting the Best important features according to Logistic Regression
    rfe_selector = RFE(estimator=RandomForestRegressor(), n_features_to_select=12, step=1)
    rfe_selector.fit(X, y)
    return X.columns[rfe_selector.get_support()]


def important_features(df_features, df_index):
    """

    :param df_features:
    :param df_index:
    :return:
    """
    df_important_features = pd.DataFrame()
    green_area_RFE = rfe_feature_selection(
        df_features.loc[df_features.index <= "2003-01-31"], df_index.loc[df_index.index <= "2003-01-31"].values.ravel()
    )
    listrfe = sorted(green_area_RFE.tolist())
    df_important_features["period 2000 - 2003"] = listrfe
    green_area_RFE = rfe_feature_selection(
        df_features.loc[(df_features.index > "2003-01-31") & (df_features.index < "2009-01-31")],
        df_index.loc[(df_index.index > "2003-01-31") & (df_index.index < "2009-01-31")].values.ravel(),
    )
    listrfe = sorted(green_area_RFE.tolist())
    df_important_features["period 2003-2009"] = listrfe
    green_area_RFE = rfe_feature_selection(
        df_features.loc[df_features.index >= "2009-01-31"], df_index.loc[df_index.index >= "2009-01-31"].values.ravel()
    )
    listrfe = sorted(green_area_RFE.tolist())
    df_important_features["period 2009-2020"] = listrfe
    return df_important_features


# Train_test_split. For the task only one-month ahead will be predicted and the rest in training period size
# is used as a training set.


def data_split_time_series(X, y, start_month, train_size_months):
    """

    :param X:
    :param y:
    :param start_month:
    :param train_size_months:
    :return:
    """
    split = start_month + train_size_months
    return X[start_month:split], X[split:split+1], y[start_month:split], y[split:split+1]


def random_forest(X, y, start_month, train_size_months):
    """

    :param X:
    :param y:
    :param start_month:
    :param train_size_months:
    :return:
    """
    X_train, X_test, y_train, y_test = data_split_time_series(X, y, start_month, train_size_months)
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    # We’re now going to apply the constructed model to make predictions on the training set and test set as follows:
    y_rf_test_pred = rf.predict(X_test)
    y_rf_test_pred = pd.DataFrame(y_rf_test_pred, columns=["Prediction"])
    y_rf_test_pred.index = y_test.index

    return pd.concat([y_rf_test_pred, y_test], axis=1)


def benchmarking_strategy(df_index):
    """

    :param df_index:
    :return:
    """
    # Initialize the short and long windows
    short_window = 3
    long_window = 24

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=df_index.index)
    signals["signal"] = 0.0

    # Create short simple moving average over the short window
    signals["short_mavg"] = df_index.rolling(window=short_window, min_periods=1, center=False).mean()

    # Create long simple moving average over the long window
    signals["long_mavg"] = df_index.rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals["signal"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:], 1.0, 0.0
    )

    # Generate trading orders
    signals["positions"] = signals["signal"].diff()
    # Add a subplot and label for y-axis
    fig = plt.figure()
    ax1 = fig.add_subplot(111, ylabel="Index")

    # Plot the index price
    df_index.plot(ax=ax1, color="r", lw=2.0)

    # Plot the short and long moving averages
    signals[["short_mavg", "long_mavg"]].plot(ax=ax1, lw=2.0)

    # # Plot the buy signals
    signals.signal.plot()
    plt.ylabel('Buy/Sell/Hold Signal') 
    plt.show()
    return signals


def strategy(prediction, df_index, short_window, long_window):
    """

    :param prediction:
    :param df_index:
    :param short_window:
    :param long_window:
    :return:
    """
    # Initialize the short and long windows
    df3 = pd.concat([prediction, df_index], axis=1)
    df3["roll_short"] = df3["MSCI All Country World Index"].rolling(short_window - 1).sum().shift(+1)
    df3["roll_long"] = df3["MSCI All Country World Index"].rolling(long_window - 1).sum().shift(+1)
    df3["short_mavg"] = (df3["Prediction"] + df3["roll_short"]) / short_window
    df3["long_mavg"] = (df3["Prediction"] + df3["roll_long"]) / long_window

    # Initialize the `signals` DataFrame with the `signal` column
    signals = pd.DataFrame(index=df3.index)
    signals["signal"] = 0.0

    # Create short simple moving average over the short window
    # signals['short_mavg'] = df_index.rolling(window=short_window, min_periods=1, center=False).mean()
    signals["short_mavg"] = df3["short_mavg"]

    # Create long simple moving average over the long window
    signals["long_mavg"] = df3["long_mavg"]
    # df_index.rolling(window=long_window, min_periods=1, center=False).mean()

    # Create signals
    signals["signal"][short_window:] = np.where(
        signals["short_mavg"][short_window:] > signals["long_mavg"][short_window:], 1.0, 0.0
    )

    # Generate trading orders
    signals["positions"] = signals["signal"].diff()
    fig = plt.figure()
    # Add a subplot and label for y-axis
    ax1 = fig.add_subplot(111, ylabel="Index")

    # Plot the closing price
    df_index.plot(ax=ax1, color="r", lw=2.0)

    # Plot the short and long moving averages
    signals[["short_mavg", "long_mavg"]].plot(ax=ax1, lw=2.0)

    # Plot the buy signals
    ax1.plot(
        signals.loc[signals.positions == 1.0].index,
        signals.short_mavg[signals.positions == 1.0],
        "^",
        markersize=10,
        color="m",
    )

    # Plot the sell signals
    ax1.plot(
        signals.loc[signals.positions == -1.0].index,
        signals.short_mavg[signals.positions == -1.0],
        "v",
        markersize=10,
        color="k",
    )
    # # Plot the buy/hold/sell signals
    fig = plt.figure()
    signals.signal.plot()
    plt.ylabel('Buy/Sell/Hold Signal') 
    plt.show()
    return signals


def backtesting(signals, df_index, initial_capital=float(100000.0)):
    """

    :param signals:
    :param df_index:
    :param initial_capital:
    :return:
    """

    # Create a DataFrame `positions`
    positions = pd.DataFrame(index=signals.index).fillna(0.0)

    # Buy a 100 shares
    positions["Index"] = 100 * signals["signal"]

    # Initialize the portfolio with value owned
    portfolio = positions.multiply(df_index["MSCI All Country World Index"], axis=0)
    # Store the difference in shares owned
    pos_diff = positions.diff()

    # Add `holdings` to portfolio
    portfolio["holdings"] = (positions.multiply(df_index["MSCI All Country World Index"], axis=0)).sum(axis=1)

    # Add `cash` to portfolio
    portfolio["cash"] = (
        initial_capital - (pos_diff.multiply(df_index["MSCI All Country World Index"], axis=0)).sum(axis=1).cumsum()
    )

    # Add `total` to portfolio
    portfolio["total"] = portfolio["cash"] + portfolio["holdings"]

    # Add `returns` to portfolio
    portfolio["returns"] = portfolio["total"].pct_change()

    # Create a figure
    fig = plt.figure(figsize=(16, 3))

    ax1 = fig.add_subplot(111, ylabel="Portfolio value")

    # Plot the equity curve
    portfolio["total"].plot(ax=ax1, lw=2.0)

    ax1.plot(
        portfolio.loc[signals.positions == 1.0].index,
        portfolio.total[signals.positions == 1.0],
        "^",
        markersize=10,
        color="m",
    )
    ax1.plot(
        portfolio.loc[signals.positions == -1.0].index,
        portfolio.total[signals.positions == -1.0],
        "v",
        markersize=10,
        color="k",
    )

    # Show the plot

    # Plot the results
    # Show the plot
    plt.title("Portfolio Value")
    plt.show()
    fig = ax1.get_figure()
    fig.savefig("Portfolio_Value.png")

    return portfolio.iloc[:-1, :]


def portfolio_metrics(portfolio, index):
    """

    :param portfolio:
    :param index:
    :return:
    """
    # 12 months are within a year
    window = 12
    # Isolate the returns of your strategy
    returns = portfolio["returns"]
    largest_monthly_gain = returns.max()
    largest_monthly_loss = returns.min()

    # annualized Sharpe ratio
    sharpe_ratio = np.sqrt(window) * (returns.mean() / returns.std())
    # 3Y Rolling Annualised Sharpe
    rolling = returns.rolling(window=36)
    rolling_sharpe_s = np.sqrt(36) * (rolling.mean() / rolling.std())
    # Print the Sharpe ratio
    # print('Annualaized Sharpe Rario:',sharpe_ratio)

    # Calculate the max drawdown in the past window months for each month
    rolling_max = index["MSCI All Country World Index"].rolling(window, min_periods=1).max()
    monthly_drawdown = index["MSCI All Country World Index"] / rolling_max - 1.0

    # Calculate the minimum (negative) monthly drawdown
    max_monthly_drawdown = monthly_drawdown.rolling(window, min_periods=1).min()

    # Plot the results
    monthly_drawdown.plot(figsize=(16, 3), linewidth=3)
    ax = max_monthly_drawdown.plot(figsize=(16, 3), linewidth=3)
    # Show the plot
    plt.title("Drawdown")
    plt.show()
    fig = ax.get_figure()
    fig.savefig("Drawdown.png")

    ax1 = rolling_sharpe_s.plot(figsize=(16, 3), linewidth=3)
    plt.title("3 years Rolling Sharpe Ratio")
    plt.show()
    fig = ax1.get_figure()
    fig.savefig("3_years_Rolling_SR.png")

    # Plot the results
    ax2 = returns.plot(figsize=(16, 3), linewidth=3)

    # Show the plot
    plt.title("Returns")
    plt.show()
    fig = ax2.get_figure()
    fig.savefig("Returns.png")

    return sharpe_ratio, max_monthly_drawdown.min(), largest_monthly_gain, largest_monthly_loss


def reporting():
    # Set up multiple variables to store the titles, text within the report
    page_title_text = "Asset Allocation Assignment Report (Daria Mustafina)"
    # Combine them together using a long f-string
    html = f"""
        <html>
            <head>
                <title>{page_title_text}</title>
            </head>
            <body>
                <h1>{page_title_text}</h1>
                <h2>{'Portoflio Performance Metrics'}</h2>
                <p>{'Drawdown'}</p>
                <img src='Drawdown.png' width="900">
                <p>{'3 years Rolling Sharpe Ratio'}</p>
                <img src='3_years_Rolling_SR.png' width="900">
                <p>{'Portfolio Value'}</p>
                <img src='Portfolio_value.png' width="900">
                <h2>{'Portoflio Performance Metrics'}</h2>
                <img src='performance.png' width="900">
                <h2>{'Important Features for different periods'}</h2>
                <img src='important_features.png' width="900">
            </body>
        </html>
        """
    # Write the html string as an HTML file
    with open("html_assignment_report.html", "w") as f:
        f.write(html)


# Tables
def render_mpl_table(
    data,
    col_width=12.0,
    row_height=1,
    font_size=24,
    header_color="#40466e",
    row_colors=None,
    edge_color="w",
    bbox=None,
    header_columns=0,
    ax=None,
    **kwargs,
):
    if bbox is None:
        bbox = [0, 0, 1, 1]
    if row_colors is None:
        row_colors = ["#f1f1f2", "w"]
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="xlsx file", default="Data_MultiAsset.xlsx")
    args = parser.parse_args()

    # Read the features and index from Excel file
    df_features = pd.read_excel(args.filename, sheet_name=0).set_index("Date")
    df_index = pd.read_excel(args.filename, sheet_name=1).set_index("Date")
    df = pd.concat([df_features, df_index], axis=1)

    # Exploratory Data Analysis
    # plotting the index, moving average, difference and percent_change
    data_plot(df_index)

    # Check for null values and data type
    df.info()
    if df.isnull().any().any():
        print("there are missing values")
    else:
        print("No missing values in the dataset are observed")

    # Look at data distribution for features and index
    df.describe()

    # Plot the pair plots for all the features and index (commented because of too many plots)
    #for col in df_features.columns:
    #    ax = df_features.plot(kind="line", y=col)
    #    df_index.plot(kind="line", y="MSCI All Country World Index", color="red", ax=ax, secondary_y=True)
    #plt.show()
    print(
        "Some features are relevant only for particular period of time (like Fed Balance Sheet before and after 2008)"
    )
    df_abs_corr = df.corr()[["MSCI All Country World Index"]].abs()
    print(df_abs_corr.sort_values(by=["MSCI All Country World Index"], ascending=False))
    # Correlations for features and the difference of features with each other and index
    high_cor = correlation(df_features, df_index)
    correlation(df_features.diff(axis=0, periods=1).add_suffix("_pct"), df_index)

    # Remove highly-correlated features from features
    # From domain knowledge and correlation coefficients we could exclude US Real Personal Income exTrans
    # and Eurozone Industrial Production exconstr and FED Excess Reserves from the features as they
    # highly correlated with the other once US Real Personal Income, Eurozone Industrial Production manuf
    # and Fed Balance Sheet, Sentiment environmental composite GLOBAL News Socia
    print("Highly Correlated Features\n", high_cor)
    df_features_selected = df_features.drop(
        ["US Real Personal Income exTrans", "Adv Retail Sales US exFood Services", "FED Excess Reserves"], axis=1
    )

    # Important features
    # Let us consider three periods 2001 - 2003, 2003 - 2009, 2009 - 2020
    df_important_features = important_features(df_features_selected, df_index)
    print("Important features from RFE method for different periods\n", df_important_features)
    # Feature selection from chosen method
    #    df_features=df_features_selected[boruta_feature_selection(df_features_selected,df_index.values.ravel())]
    df_features = df_features_selected[rfe_feature_selection(df_features_selected, df_index.values.ravel())]

    # Predicting next month index value
    results = pd.DataFrame()
    # The size of the training data set
    train_size_months = 12
    for i in range(0, (len(df_features) - train_size_months - 1)):
        results = results.append(random_forest(df_features, df_index[df_index.columns[0]], i, train_size_months))
    ax = results.plot()
    results["mape"] = 100 * abs(results["Prediction"].abs() - results.iloc[:, -1]).div(results.iloc[:, -1].values).abs()
    MAPE_model = results["mape"].mean()
    results.plot(kind="line", y="mape", color="red", ax=ax, secondary_y=True)
    print("Mean average percentage error of the model\n", MAPE_model, "%")
    plt.show()
    fig = ax.get_figure()
    fig.savefig("RF Index Prediction.png")
    pred = results["Prediction"]
    print(results)
    print(type(results))

    # Simple Allocation Strategy MA Crossover
    short_window = 3
    long_window = 24
    # Simple Allocation Strategy MA Crossover for historical index values
    signals_becnhmarking = benchmarking_strategy(df_index)
    
    portfolio = backtesting(signals_becnhmarking, df_index)
    # Allocation Strategy Performance Evaluation    
    total_return = (portfolio["total"].iloc[-1] - portfolio["total"].iloc[0]) / portfolio["total"].iloc[0]
    sharpe_ratio, max_monthly_drawdown, largest_monthly_gain, largest_monthly_loss = portfolio_metrics(
        portfolio, df_index
    )
    performance = pd.DataFrame(
        [
            sharpe_ratio,
            100 * max_monthly_drawdown,
            100 * largest_monthly_gain,
            100 * largest_monthly_loss,
            100 * total_return,
        ],
        columns=["Values"],
        index=[
            "Annualized sharpe_ratio",
            "Maximum Drawdown, %",
            "Largest Monthly Gain, %",
            "Largest Monthly Loss, %",
            "Total Return, %",
        ],
    )
    performance = performance.round(2)
    fig, ax = render_mpl_table(df_important_features, header_columns=0, col_width=12.0)
    fig.savefig("important_features.png")
    fig, ax = render_mpl_table(
        performance.rename_axis("Performance Metrics").reset_index(), header_columns=0, col_width=12.0
    )
    fig.savefig("performance_benchmarking.png")
    print(performance)

    # Simple Allocation Strategy MA Crossover for predicted index values
    signals = strategy(pred, df_index, short_window, long_window)

    # Allocation Strategy back-testing
    portfolio = backtesting(signals, df_index)
    total_return = (portfolio["total"].iloc[-1] - portfolio["total"].iloc[0]) / portfolio["total"].iloc[0]

    # Allocation Strategy Performance Evaluation
    sharpe_ratio, max_monthly_drawdown, largest_monthly_gain, largest_monthly_loss = portfolio_metrics(
        portfolio, df_index
    )
    performance = pd.DataFrame(
        [
            sharpe_ratio,
            100 * max_monthly_drawdown,
            100 * largest_monthly_gain,
            100 * largest_monthly_loss,
            100 * total_return,
        ],
        columns=["Values"],
        index=[
            "Annualized sharpe_ratio",
            "Maximum Drawdown, %",
            "Largest Monthly Gain, %",
            "Largest Monthly Loss, %",
            "Total Return, %",
        ],
    )
    performance = performance.round(2)
    fig, ax = render_mpl_table(df_important_features, header_columns=0, col_width=12.0)
    fig.savefig("important_features.png")
    fig, ax = render_mpl_table(
        performance.rename_axis("Performance Metrics").reset_index(), header_columns=0, col_width=12.0
    )
    fig.savefig("performance.png")
    print(performance)

    # Reporting
    reporting()


if __name__ == "__main__":
    main()
