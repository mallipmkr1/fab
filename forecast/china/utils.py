from subprocess import call

import numpy as np
import pandas as pd


def calc_metrics(y, yhat):
    """
    Calculate metrics for a single series.
    """
    # mae
    mae = np.mean(np.abs(y - yhat))

    # smape
    smape = 2 * np.abs((y - yhat) / (y + yhat))
    smape = np.nan_to_num(smape, 0)
    smape = np.mean(smape)

    # mape
    mape = np.abs((y - yhat) / y)
    mape = np.nan_to_num(mape, 0)
    mape = np.mean(mape)

    # maape
    maape = np.arctan2(np.abs(y - yhat), np.abs(y))
    maape = np.mean(maape)

    # rmse
    rmse = (y - yhat) * (y - yhat)
    rmse = np.sqrt(np.mean(rmse))

    return {
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "mape": mape,
        "maape": maape
    }


def get_dates(
        ts_test_start,
        ts_test_end,
        ts_predict_start,
        ts_predict_end,
        ts_validate_start=None,
        ts_validate_end=None,
        freq="W-Mon",
):
    if ts_validate_start is not None and ts_validate_end is not None:
        val_dates = pd.date_range(
            start=ts_validate_start,
            end=ts_validate_end,
            freq=freq,
        )
        val_dates = val_dates[:-1]
    else:
        val_dates = None
    test_dates = pd.date_range(
        start=ts_test_start,
        end=ts_test_end,
        freq=freq,
    )
    test_dates = test_dates[:-1]
    forecast_dates = pd.date_range(
        start=ts_predict_start,
        end=ts_predict_end,
        freq=freq,
    )

    if val_dates is not None:
        return {"val": val_dates, "test": test_dates, "forecast": forecast_dates}

    return {"test": test_dates, "forecast": forecast_dates}


def make_future_df(df, forecast_dates, date_col="ts", non_static_cols=None, fill_with_nan=True):
    """
    Add future dates to existing dataframe
    """
    if non_static_cols is None:
        non_static_cols = ["price", "list_price"]

    future_df = [df.loc[df[date_col] < np.min(forecast_dates)].copy()]
    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        b.sort_values(by=date_col, inplace=True)
        b = b.loc[b[date_col] < np.min(forecast_dates)]
        b.reset_index(drop=True, inplace=True)

        a = pd.DataFrame(columns=df.columns)
        a[date_col] = forecast_dates
        a["id"] = s
        if "price" in b.columns:
            if fill_with_nan:
                a["price"] = np.nan
            else:
                a["price"] = b["price"].mean()

        if "list_price" in b.columns:
            if fill_with_nan:
                a["list_price"] = np.nan
            else:
                a["list_price"] = b["list_price"].values[-1]

        skip_columns = [date_col, "y"] + non_static_cols
        for c in df.columns:
            if c in skip_columns:
                continue
            a[c] = b[c].values[-1]
        future_df.append(a)

    future_df = pd.concat(future_df).reset_index(drop=True)
    future_df["y"] = future_df["y"].astype(float)
    future_df["year"] = future_df[date_col].dt.year
    future_df["month"] = future_df[date_col].dt.month
    future_df["woy"] = future_df[date_col].dt.isocalendar().week
    future_df["woy"] = future_df["woy"].astype(int)
    return future_df


def upgrade_pandas():
    call("pip install --upgrade pandas --user", shell=True)


def install_old_pandas(version="1.3.3"):
    call("pip install pandas=={} --user".format(version), shell=True)
