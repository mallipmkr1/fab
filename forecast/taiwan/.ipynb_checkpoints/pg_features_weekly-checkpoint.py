import multiprocessing

import numpy as np
import pandas as pd
from tqdm import tqdm
from tsfresh import extract_features


def clean_basic_material(description):
    description = description.replace("\xa0", "")
    return description


def get_events():
    """
    Hard coded events for now.
    Following Prophet's format
    """
  
    women_day_promotion = pd.DataFrame({
        'holiday': 'women day',
        'ds': pd.to_datetime(['2018-03-01','2019-03-01', '2020-03-01', '2021-03-01', '2022-03-01', '2023-03-01']),
        'lower_window': -7,
        'upper_window': 14,
        'type': 'promotion_specific_date',
    })
    
    lazada_bday_promotion = pd.DataFrame({
        'holiday': 'lazada_bday_promotion',
        'ds': pd.to_datetime(['2018-04-01','2019-04-01', '2020-04-01', '2021-04-01', '2022-04-01', '2023-04-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'promotion',
    })
    
    labour_day_promotion = pd.DataFrame({
        'holiday': 'Labour day',
        'ds': pd.to_datetime(['2018-05-01','2019-05-01', '2020-05-01', '2021-05-01', '2022-05-01', '2023-05-01']),
        'lower_window': -7,
        'upper_window': 0,
        'type': 'promotion_specific_date',
    })


    day_99_promotion = pd.DataFrame({
        'holiday': '9.9',
        'ds': pd.to_datetime(['2019-09-01','2019-09-01', '2020-09-01', '2021-09-01', '2022-09-01', '2023-09-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'promotion_specific_date',
    })
    
    day_1010_promotion = pd.DataFrame({
        'holiday': '10.10',
        'ds': pd.to_datetime(['2018-10-01','2019-10-01', '2020-10-01', '2021-10-01', '2022-10-01', '2023-10-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'promotion_specific_date',
    })


    day_1111_promotion = pd.DataFrame({
        'holiday': '11.11',
        'ds': pd.to_datetime(['2018-11-01','2019-11-01', '2020-11-01', '2021-11-01', '2022-11-01', '2023-11-01']),
        'lower_window': -14,
        'upper_window': 7,
        'type': 'promotion_specific_date',
    })

    christmas_promotion = pd.DataFrame({
        'holiday': 'christmas',
        'ds': pd.to_datetime(['2018-12-01','2019-12-01', '2020-12-01', '2021-12-01', '2022-12-01', '2023-12-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'festival',
    })

    events = pd.concat((
        women_day_promotion,
        lazada_bday_promotion,
        labour_day_promotion,
        day_99_promotion,
        day_1010_promotion,
        day_1111_promotion,
        christmas_promotion,
    )).reset_index(drop=True)

    a = pd.get_dummies(events["type"])
    events = pd.concat([events, a], axis=1)
    return events


def load_data(vol_file=None, price_file=None, att_file=None, main_group_index=None):
    if main_group_index is None:
        main_group_index = ["material", "customer_group"]
    if att_file is None:
        att_file = "../data/all_att_data.csv"
    if vol_file is None:
        vol_file = "../data/si_weekly_volume.parquet"
    if price_file is None:
        price_file = "../data/si_weekly_price.parquet"

    att = pd.read_csv(att_file)
    _id = att[main_group_index].drop_duplicates().reset_index().rename(columns={"index": "id"})
    att = pd.merge(_id, att, on=main_group_index, how="left")
    # some cleaning
    att["basic_material"] = att["basic_material"].astype(str)
    att["basic_material"] = [clean_basic_material(x) for x in att["basic_material"]]

    df = pd.read_parquet(vol_file).rename(columns={"si_net_volume": "y", "date": "ts"})
    df = df.loc[df.y >= 0]
    df.drop(columns=["si_volume", "si_return_volume", "si_return_value", "si_net_value", "si_revenue"], inplace=True)
    df = pd.merge(df, att, on=main_group_index, how="left")

    weekly_price = pd.read_parquet(price_file).rename(columns={"date": "ts"})
    df = pd.merge(df, weekly_price, on=main_group_index + ["ts"], how="left")

    df["id"] = df["id"].astype(np.int32)
    df.rename(columns={
        "si_weekly_list_price": "list_price",
        "si_weekly_price": "price",
    }, inplace=True)

    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        mean_price = b["price"].mean()
        mean_list_price = b["list_price"].mean()
        df.loc[df["id"] == s, "price"] = df.loc[df["id"] == s, "price"].fillna(mean_price)
        df.loc[df["id"] == s, "list_price"] = df.loc[df["id"] == s, "list_price"].fillna(mean_list_price)
    df.sort_values(by=["id", "ts"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def load_monthly(vol_file=None, price_file=None, att_file=None, main_group_index=None):
    if main_group_index is None:
        main_group_index = ["material", "customer_group"]
    if att_file is None:
        att_file = "../data/all_att_data.csv"
    if vol_file is None:
        vol_file = "../data/si_monthly_volume.parquet"
    if price_file is None:
        price_file = "../data/si_monthly_price.parquet"

    att = pd.read_csv(att_file)
    _id = att[main_group_index].drop_duplicates().reset_index().rename(columns={"index": "id"})
    att = pd.merge(_id, att, on=main_group_index, how="left")
    # some cleaning
    att["basic_material"] = att["basic_material"].astype(str)
    att["basic_material"] = [clean_basic_material(x) for x in att["basic_material"]]

    df = pd.read_parquet(vol_file).rename(columns={"si_monthly_volume": "y", "date": "ts"})
    df = df.loc[df.y >= 0]
    df.drop(columns=["so_monthly_qty", "si_gsv1", "so_gsv", "si_nsv", "si_gsv"], inplace=True)
    df = pd.merge(df, att, on=main_group_index, how="left")
    drop = df.loc[
        (df["customer_group"] == "Robinsons") &
        (df["prod_category"].isin(["Soy Sauce", "Beverage"]))
        ]
    df = df.drop(drop.index)
    df.reset_index(drop=True, inplace=True)

    monthly_price = pd.read_parquet(price_file).rename(columns={"date": "ts"})
    df = pd.merge(df, monthly_price, on=main_group_index + ["ts"], how="left")

    df["id"] = df["id"].astype(np.int32)
    df.rename(columns={
        "si_monthly_list_price": "list_price",
        "si_monthly_price": "price",
    }, inplace=True)

    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        mean_price = b["price"].mean()
        mean_list_price = b["list_price"].mean()
        df.loc[df["id"] == s, "price"] = df.loc[df["id"] == s, "price"].fillna(mean_price)
        df.loc[df["id"] == s, "list_price"] = df.loc[df["id"] == s, "list_price"].fillna(mean_list_price)
    df.sort_values(by=["id", "ts"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def resample_data(df, value_var="y", time_var="ts", group_var="id", freq="W-Mon"):
    ddf = df.copy()
    a = ddf.set_index(time_var).groupby(group_var).resample(freq)[value_var].sum().reset_index()
    ddf = pd.merge(a, ddf.drop(columns=value_var), on=["id", "ts"], how="left")
    for s in ddf["id"].unique():
        b = ddf.loc[ddf["id"] == s].copy()
        if "price" in b.columns:
            mean_price = b["price"].mean()
            ddf.loc[ddf["id"] == s, "price"] = ddf.loc[ddf["id"] == s, "price"].fillna(mean_price)
        if "list_price" in b.columns:
            mean_list_price = b["list_price"].mean()
            ddf.loc[ddf["id"] == s, "list_price"] = ddf.loc[ddf["id"] == s, "list_price"].fillna(mean_list_price)
        ddf.loc[ddf["id"] == s] = ddf.loc[ddf["id"] == s].fillna(method="ffill")
    ddf.sort_values(by=["id", "ts"], inplace=True)

    return ddf


def make_future_df(df, forecast_dates, date_col="ts", non_static_cols=None, fill_with_nan=True, freq="W-Mon"):
    """
    Add future dates to existing dataframe
    """
    if non_static_cols is None:
        non_static_cols = [
            "price", "list_price"
        ]

    future_df = [df.loc[df.ts < np.min(forecast_dates)].copy()]
    max_date = future_df[0].ts.max()
    # max_date = np.min(forecast_dates)

    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        b.sort_values(by=date_col, inplace=True)
        b = b.loc[b.ts < np.min(forecast_dates)]
        b.reset_index(drop=True, inplace=True)
        if len(b) <= 1:
            continue
        # pad with zeros first
        b_max_date = b.ts.max()
        if b_max_date < max_date:
            d = pd.date_range(b_max_date, max_date, freq=freq)
            a = pd.DataFrame(columns=df.columns)
            a[date_col] = d
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

        # pad with zeros first
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
    return future_df


def project_future(df, all_dates):
    c = {}
    for a in all_dates.keys():
        b = make_future_df(df, all_dates[a]["forecast"])
        c.update({a: b})

    return c


def extract_sliding(df, window=8, gap=4, target="y"):
    lags = [x for x in range(gap + 1, gap + window + 1)]
    aa = df[["id", "ts", "y"]].assign(**{
        "{}_lag_{}".format(col, l): df.groupby(["id"], observed=True)[col].transform(lambda x: x.shift(l))
        for l in lags
        for col in [target]
    })
    aa.sort_values(by=["id", "ts"], inplace=True)
    aa.reset_index(inplace=True)
    aa.rename(columns={"index": "sub_id"}, inplace=True)

    b = aa.drop(columns=["id", "ts", "y"]).melt(
        id_vars="sub_id",
        value_name="y",
        var_name="lags"
    )
    b["lags"] = [int(x.replace("y_lag_", "")) for x in b["lags"]]
    return aa, b


def extract_tsfresh_features(
        df,
        all_dates,
        use_val=False,
        nan_ratio=0.5,
        gap=12,
        window=8,
        n_jobs=0,
):
    if n_jobs <= 0:
        n_jobs = multiprocessing.cpu_count()

    n_jobs = min(n_jobs, multiprocessing.cpu_count())

    if use_val:
        last_train_date = np.min(all_dates["val"])
    else:
        last_train_date = np.min(all_dates["test"])

    df_lags, df_lags_long = extract_sliding(df, gap=gap, window=window)

    print("calling tsfresh package")
    sku_with_sale = df_lags_long.dropna()
    

    
    filter_df = sku_with_sale[sku_with_sale['y'] > 0]
    filter_df = filter_df.groupby('sub_id')['y'].count().reset_index()
    filter_df = filter_df[filter_df['y'] >= 2]
    filter_ids = filter_df.sub_id.unique()
    sku_with_sale = sku_with_sale[sku_with_sale['sub_id'].isin(filter_ids)]    
    
    tsfresh_features = extract_features(
        sku_with_sale,
        # default_fc_parameters=MinimalFCParameters(),
        column_id="sub_id",
        column_sort="lags",
        column_value="y",
        n_jobs=n_jobs
    )
    
    
    tsfresh_features.reset_index(inplace=True)
    tsfresh_features.rename(columns={"index": "sub_id", "level_1": "ts"}, inplace=True)

    tsfresh_features = pd.merge(
        df_lags.drop(columns=["y"]),
        tsfresh_features,
        on="sub_id",
        how="right"
    )
    tsfresh_features.drop(columns=["sub_id"], inplace=True)
    tsfresh_features = pd.merge(df, tsfresh_features, on=["id", "ts"], how="left")

    drop_cols = []
    for col in tsfresh_features.select_dtypes(include=["number", "bool_"]).columns:
        aa = tsfresh_features.loc[tsfresh_features.ts < last_train_date][col]
        if np.sum(aa.isnull()) >= nan_ratio * len(aa):
            drop_cols.append(col)
    tsfresh_features.drop(columns=drop_cols, inplace=True)

    return tsfresh_features


def extract_summary(
        df,
        all_dates,
        group,
        var_name,
        summary_stats=None,
        use_val=False,
):
    if summary_stats is None:
        summary_stats = ["sum", "mean", "max", "min", "std"]
    if use_val:
        last_train_date = np.min(all_dates["val"])
    else:
        last_train_date = np.min(all_dates["test"])
    group_str = "_".join(group)

    a = df.loc[df.ts < last_train_date].copy()
    a = a.groupby(group).agg({var_name: summary_stats}).reset_index()
    cols = group.copy()
    for s in summary_stats:
        b = "{}_{}_{}".format(var_name, s, group_str)
        cols.append(b)
    a.columns = cols
    print(cols)
    sum_col = [x for x in cols if "sum" in x]
    a.sort_values(
        by=sum_col,
        ascending=False,
        inplace=True
    )
    a = a.reset_index(drop=True)
    a["{}_percent".format(group_str)] = a[sum_col] / a[sum_col].sum()
    for x in a.columns:
        if (x not in group) and (x in df.columns):
            df.drop(columns=[x], inplace=True)

    df_feature = pd.merge(df, a, on=group, how="left")

    return df_feature


def count_series(
        df,
        all_dates,
        group,
        use_val=False,
):
    if use_val:
        last_train_date = np.min(all_dates["val"])
    else:
        last_train_date = np.min(all_dates["test"])

    a = df.loc[df.ts < last_train_date].copy()
    a = a[["id"] + group].drop_duplicates()
    a = a.groupby(group).count()
    a.reset_index(inplace=True)
    column_name = "num_id_" + "_".join(group)
    a.rename(columns={
        "id": column_name
    }, inplace=True)

    if column_name in df.columns:
        df.drop(columns=[column_name], inplace=True)

    df_feature = pd.merge(df, a, on=group, how="left")

    return df_feature


def get_features(
        df,
        all_dates,
        freq="W-Mon",
        value_var="y",
        time_var="ts",
        group_var="id",
        non_static_cols=None,
        fill_with_nan=True,
        summary_stats=None,
        groups=None,
        use_val=False,
        nan_ratio=0.5,
        gap=12,
        window=8,
        n_jobs=0,
        use_tsfresh = False
):
    # 1. resample
    print("Resampling")
    df_resample = resample_data(
        df,
        value_var=value_var,
        time_var=time_var,
        group_var=group_var,
        freq=freq
    )
    print(df_resample.shape)

    # 2. make future
    print("Make future")
    df_future = make_future_df(
        df_resample,
        all_dates["forecast"],
        date_col=time_var,
        non_static_cols=non_static_cols,
        fill_with_nan=fill_with_nan,
        freq=freq,
    )
    del df_resample
    print(df_future.shape)

    # 3. create days features
    print("Create days features")
    df_future["week"] = df_future["ts"].dt.week
    df_future["year"] = df_future["ts"].dt.year
    df_future["month"] = df_future["ts"].dt.month
    df_future["quarter"] = df_future["ts"].dt.quarter
    if freq == "W-Mon":
        df_future["woy"] = df_future["ts"].dt.isocalendar().week
        df_future["woy"] = df_future["woy"].astype(int)
    df_future["is_month_start"] = df_future["ts"].dt.is_month_start
    df_future["is_month_end"] = df_future["ts"].dt.is_month_end
    df_future["is_quarter_start"] = df_future["ts"].dt.is_quarter_start
    df_future["is_quarter_end"] = df_future["ts"].dt.is_quarter_end
    df_future["is_quarter_start"] = df_future["ts"].dt.is_quarter_start
    df_future["is_year_start"] = df_future["ts"].dt.is_year_start
    df_future["is_year_end"] = df_future["ts"].dt.is_year_end

    print(df_future.shape)

    # 4. create tsfresh features
    print("Create tsfresh")
    
    
    

    # 5. create summary features
    if(use_tsfresh):
        df_tsfresh = extract_tsfresh_features(
            df_future,
            all_dates,
            use_val=use_val,
            nan_ratio=nan_ratio,
            gap=gap,
            window=window,
            n_jobs=n_jobs,
        )
        df_summary = df_tsfresh.copy()
        del df_tsfresh
        print(df_tsfresh.shape)
    else:
        df_summary = df_future.copy()
    del df_future
   

    if groups is None:
        groups = [
            ['sku'],
            ["sku", "month"],
            ["sku", "week"]
        ]
       

    print("Create summary features")
    for g in groups:
        df_summary = extract_summary(
            df_summary,
            all_dates,
            group=g,
            var_name=value_var,
            summary_stats=summary_stats,
            use_val=use_val,
        )
        print(df_summary.columns)

    df_feature = df_summary.copy()
    print(df_feature.columns)
    del df_summary
    print(df_feature.shape)

    return df_feature
