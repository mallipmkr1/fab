from tqdm import tqdm
import numpy as np
import pandas as pd


def mean_encode(data, gb_cols, not_nan_mask=None, quantile=0.95):
    original_y = data["y"].copy()
    if not_nan_mask is not None:
        data.loc[~not_nan_mask, "y"] = np.nan
    data.loc[data.y > np.quantile(data["y"], quantile), "y"] = np.nan

    for col in gb_cols:
        if not isinstance(col, str):
            col_name = "meanenc_" + "_".join(col) + "_"
        else:
            col_name = "meanenc_" + "".join(col) + "_"
        for s in ["mean", "std", "max"]:
            data[col_name + s] = data.groupby(col)["y"].transform(s)

    data["y"] = original_y
    return data



def make_future_df(df, forecast_dates, date_col="ts", non_static_cols=None, fill_with_nan=True):
    """
    Add future dates to existing dataframe
    """
    if non_static_cols is None:
        non_static_cols = ["price", "list_price"]

    future_df = [df.loc[df.ts < np.min(forecast_dates)].copy()]
    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        b.sort_values(by=date_col, inplace=True)
        b = b.loc[b.ts < np.min(forecast_dates)]
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
    future_df["year"] = future_df.ts.dt.year
    future_df["month"] = future_df.ts.dt.month
    future_df["woy"] = future_df.ts.dt.isocalendar().week
    future_df["woy"] = future_df["woy"].astype(int)
    return future_df



def find_spike_with_demand(df, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q)] = 1
    spikes[(df["lag_demand"] > q)] = 1
    df["dummy_spike"] = spikes
    return df

def find_spike_with_events(df, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q)] = 1
    spikes[(df["events"] != "no_events")] = 1
    df["dummy_spike"] = spikes
    return df

def find_spike(df, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q)] = 1
    df["dummy_spike"] = spikes
    return df
   


def find_spike_with_demand(df, val_date, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q) & (df.ts < val_date)] = 1
    spikes[(df["expected_demand"] > q)] = 1
    df["dummy_spike"] = spikes
    return df
from tqdm import tqdm
import numpy as np
import pandas as pd


def mean_encode(data, gb_cols, not_nan_mask=None, quantile=0.95):
    original_y = data["y"].copy()
    if not_nan_mask is not None:
        data.loc[~not_nan_mask, "y"] = np.nan
    data.loc[data.y > np.quantile(data["y"], quantile), "y"] = np.nan

    for col in gb_cols:
        if not isinstance(col, str):
            col_name = "meanenc_" + "_".join(col) + "_"
        else:
            col_name = "meanenc_" + "".join(col) + "_"
        for s in ["mean", "std", "max"]:
            data[col_name + s] = data.groupby(col)["y"].transform(s)

    data["y"] = original_y
    return data


def make_future_df(df, forecast_dates, date_col="ts", non_static_cols=None, fill_with_nan=True):
    """
    Add future dates to existing dataframe
    """
    if non_static_cols is None:
        non_static_cols = ["price", "list_price"]

    future_df = [df.loc[df.ts < np.min(forecast_dates)].copy()]
    for s in df["id"].unique():
        b = df.loc[df["id"] == s].copy()
        b.sort_values(by=date_col, inplace=True)
        b = b.loc[b.ts < np.min(forecast_dates)]
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
    future_df["year"] = future_df.ts.dt.year
    future_df["month"] = future_df.ts.dt.month
    future_df["woy"] = future_df.ts.dt.isocalendar().week
    future_df["woy"] = future_df["woy"].astype(int)
    return future_df


def find_spike_with_demand(df, val_date, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q) & (df.ts < val_date)] = 1
    spikes[(df["lag_demand"] > q)] = 1
    df["dummy_spike"] = spikes
    return df


def find_spike_with_events(df, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q)] = 1
    spikes[(df["events"] != "no_events")] = 1
    df["dummy_spike"] = spikes
    return df


def find_spike(df, quantile=0.9):
    q = np.quantile(df["y"].dropna(), quantile)
    spikes = np.zeros((len(df),))
    spikes[(df["y"] > q)] = 1
    df["dummy_spike"] = spikes
    return df


def get_events():
    """
    Hard coded events for now.
    Following Prophet's format
    """
    cny_promotion = pd.DataFrame({
        'holiday': 'cny',
        'ds': pd.to_datetime(['2019-02-01', '2020-01-01', '2021-02-01', '2022-02-01', '2023-01-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'festival',
        'impact' : 'small_promo'
    })

    women_day_promotion = pd.DataFrame({
        'holiday': 'women day',
        'ds': pd.to_datetime(['2019-03-01', '2020-03-01', '2021-03-01', '2022-03-01', '2023-03-01']),
        'lower_window': 0,
        'upper_window': 0,
        'type': 'promotion_specific_date',
        'impact' : 'medium_promo'
    })

    mother_day_promotion = pd.DataFrame({
        'holiday': 'mothers day',
        'ds': pd.to_datetime(['2019-05-01', '2020-05-01', '2021-05-01', '2022-05-01', '2023-05-01']),
        'lower_window': -7,
        'upper_window': 0,
        'type': 'promotion_specific_date',
        'impact' : 'small_promo'
    })


    day_618_promotion = pd.DataFrame({
        'holiday': '618',
        'ds': pd.to_datetime(['2019-06-01', '2020-06-01', '2021-06-01', '2022-06-01', '2023-06-01']),
        'lower_window': -14,
        'upper_window': 14,
        'type': 'promotion_specific_date',
        'impact' : 'medium_promo'
    })

    day_99_promotion = pd.DataFrame({
        'holiday': '9.9',
        'ds': pd.to_datetime(['2019-09-01', '2020-09-01', '2021-09-01', '2022-09-01', '2023-09-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'promotion_specific_date',
        'impact' : 'medium_promo'
    })

    day_1111_promotion = pd.DataFrame({
        'holiday': '11.11',
        'ds': pd.to_datetime(['2019-11-01', '2020-11-01', '2021-11-01', '2022-11-01', '2023-11-01']),
        'lower_window': -14,
        'upper_window': 7,
        'type': 'promotion_specific_date',
        'impact' : 'main_promo'
    })



    christmas_promotion = pd.DataFrame({
        'holiday': 'christmas',
        'ds': pd.to_datetime(['2019-12-01', '2020-12-01', '2021-12-01', '2022-12-01', '2023-12-01']),
        'lower_window': -7,
        'upper_window': 7,
        'type': 'festival',
        'impact' : 'small_promo'
    })

    events = pd.concat((
        cny_promotion,
        women_day_promotion,
        mother_day_promotion,
        day_618_promotion,
        day_99_promotion,
        day_1111_promotion,
        christmas_promotion,
    )).reset_index(drop=True)

    a = pd.get_dummies(events["type"])
    b =  pd.get_dummies(events["impact"])
    events = pd.concat([events, a], axis=1)
    events = pd.concat([events, b], axis=1)
    return events

def get_events_full_data(event_init_df, events_df, events, event_types, event_types1):
    _df = event_init_df
    df_with_event = pd.merge(_df, events_df.rename(columns={"ds": "ts"}), on="ts", how="left")
    df_with_event["holiday"].fillna("no_events", inplace=True)
    df_with_event["type"].fillna("no_events", inplace=True)
    print(event_types)
    df_with_event[event_types] = df_with_event[event_types].fillna(0)
    df_with_event[event_types1] = df_with_event[event_types1].fillna(0)
    df_with_event.rename(columns={"holiday": "events"}, inplace=True)
    events_df.sort_values(by=['ds','type'], inplace = True)
    events.sort_values(by=['ds','type'], inplace = True)
    tmp = []
    for _id in df_with_event.id.unique():
        ts = df_with_event.loc[df_with_event.id == _id].sort_values(by="ts").reset_index(drop=True)
        ts["last_event"] = None
        ts["next_event"] = None
        ts["weeks_since_last_event"] = None
        ts["weeks_to_next_event"] = None
        ts["days_since_last_event"] = None
        ts["days_to_next_event"] = None
        for i in range(len(ts)):
            if ts.events[i] == "no_events":
                a = events_df.loc[events_df.ds < ts.ts[i]]
                b = events_df.loc[events_df.ds > ts.ts[i]]

                # to calculate days to events
                aa = events.loc[(events.ds < ts.ts[i])]
                bb = events.loc[(events.ds > ts.ts[i])]

                ts.loc[i, "last_event"] = a.holiday.values[-1]
                ts.loc[i, "next_event"] = b.holiday.values[0]
                ts.loc[i, "weeks_since_last_event"] = (ts.ts[i] - a.ds.values[-1]).days / 7
                ts.loc[i, "weeks_to_next_event"] = (b.ds.values[0] - ts.ts[i]).days / 7
                ts.loc[i, "days_since_last_event"] = (ts.ts[i] - aa.ds.values[-1]).days
                ts.loc[i, "days_to_next_event"] = (bb.ds.values[0] - ts.ts[i]).days
            else:
                ts.loc[i, "last_event"] = "event_this_week"
                ts.loc[i, "next_event"] = "event_this_week"
                ts.loc[i, "weeks_since_last_event"] = 0
                ts.loc[i, "weeks_to_next_event"] = 0
                ts.loc[i, "days_since_last_event"] = 0
                ts.loc[i, "days_to_next_event"] = 0
        tmp.append(ts)

    df_with_event = pd.concat(tmp).reset_index(drop=True)
    df_with_event["days_to_next_event"] = df_with_event["days_to_next_event"].astype(int)
    df_with_event["days_since_last_event"] = df_with_event["days_since_last_event"].astype(int)
    df_with_event["weeks_to_next_event"] = df_with_event["weeks_to_next_event"].astype(int)
    df_with_event["weeks_since_last_event"] = df_with_event["weeks_since_last_event"].astype(int)
    return df_with_event

def get_features(df, all_dates, outlier_quantile=0.9, verbose=1,freq="W-Mon"):
    #### 1. get events
    if verbose:
        print("Get events")
    
    events = get_events()
    events['festival'] = events['festival'].astype('int64')
    events['promotion'] = events['promotion'].astype('int64')
    events['promotion_specific_date'] = events['promotion_specific_date'].astype('int64')
    events_weekly = []
       
    for h in events.holiday.unique():
        # for each event, resample it to W-Mon
        e = events.loc[events.holiday == h]
        e = e.set_index("ds").resample(freq, label="left").first()
        e.dropna(inplace=True)
        events_weekly.append(e.reset_index())
    event_types = pd.get_dummies(events["type"]).columns.to_list()
    events_weekly = pd.concat(events_weekly)
    tmp = []
    for t in events_weekly.ds.unique():
        e = events_weekly.loc[events_weekly.ds == t].reset_index(drop=True)
        if len(e) == 1:
            tmp.append(e.drop(columns=["lower_window", "upper_window"]))
        else:
            e.drop(columns=["lower_window", "upper_window"], inplace=True)
            e["holiday"] = ",".join(e["holiday"].tolist())
            e["type"] = ",".join(e["type"].tolist())
            _sum = e[event_types].sum()
            for c in event_types:
                e[c] = _sum[c]
            tmp.append(e.drop_duplicates())
    events_weekly = pd.concat(tmp)
    events_weekly.sort_values(by="ds", inplace=True)
    events_weekly.reset_index(inplace=True, drop=True)
    del tmp, e

    
    #### 2. make future
    if verbose:
        print("Make future dates")

    future_df = make_future_df(df, all_dates["forecast"])

    #### 3. add overall average demand from train
    _df = future_df.copy()
    _df = _df.loc[_df.ts < all_dates["val"].min()].reset_index()
    aa = []
    for _id in _df.id.unique():
        ts = _df.loc[_df.id == _id]
        outliers = np.quantile(ts.y, outlier_quantile)
        ts = ts.loc[ts.y <= outliers]
        aa.append(ts)
    aa = pd.concat(aa)
    aa = aa[aa['y'] > 0]
    avgs = aa.groupby(["id"])["y"].mean().reset_index()
    avgs.rename(columns={"y": "average_train_demand"}, inplace=True)
    _df = pd.merge(future_df.copy(), avgs, on=["id"], how="left")
    del aa, ts, avgs, outliers
    
    #### 3. add days since launch
    
    if verbose:
        print("Add days since launch")
    tmp = []
    for _id in tqdm(_df.id.unique()):
        ts = _df.loc[_df.id == _id].sort_values(by="ts").reset_index(drop=True)
        ts["days_since_launch"] = np.arange(len(ts))
        _idx = np.where(ts.y > 0)[0]
        if len(_idx) > 0:
            _idx = _idx[0]
            ts["days_since_launch"] = ts["days_since_launch"] - _idx
        else:
            ts["days_since_launch"] = -9999
        tmp.append(ts)
    tmp = pd.concat(tmp).reset_index(drop=True)
    tmp = tmp[tmp.drop(columns=["days_since_launch"]).columns.tolist() + ["days_since_launch"]]
    df_since_launch = tmp
    del tmp, _df, future_df, ts, _idx



    #### 4. merge with events
    if verbose:
        print("Merge with events")
    # sort events
    events.sort_values(by="ds", inplace=True)
    events.reset_index(drop=True, inplace=True)

    _df = df_since_launch
    df_with_event = pd.merge(_df, events_weekly.rename(columns={"ds": "ts"}), on="ts", how="left")
    df_with_event["holiday"].fillna("no_events", inplace=True)
    df_with_event["type"].fillna("no_events", inplace=True)
    df_with_event[event_types] = df_with_event[event_types].fillna(0)
    df_with_event.rename(columns={"holiday": "events"}, inplace=True)
    events_df.sort_values(by=['ds','type'], inplace = True)
    events.sort_values(by=['ds','type'], inplace = True)
    tmp = []
    for _id in tqdm(df_with_event.id.unique()):
        ts = df_with_event.loc[df_with_event.id == _id].sort_values(by="ts").reset_index(drop=True)
        ts["last_event"] = None
        ts["next_event"] = None
        ts["weeks_since_last_event"] = None
        ts["weeks_to_next_event"] = None
        ts["days_since_last_event"] = None
        ts["days_to_next_event"] = None
        for i in range(len(ts)):
            if ts.events[i] == "no_events":
                a = events_weekly.loc[events_weekly.ds < ts.ts[i]]
                b = events_weekly.loc[events_weekly.ds > ts.ts[i]]

                # to calculate days to events
                aa = events.loc[(events.ds < ts.ts[i])]
                bb = events.loc[(events.ds > ts.ts[i])]

                ts.loc[i, "last_event"] = a.holiday.values[-1]
                ts.loc[i, "next_event"] = b.holiday.values[0]
                ts.loc[i, "weeks_since_last_event"] = (ts.ts[i] - a.ds.values[-1]).days / 7
                ts.loc[i, "weeks_to_next_event"] = (b.ds.values[0] - ts.ts[i]).days / 7
                ts.loc[i, "days_since_last_event"] = (ts.ts[i] - aa.ds.values[-1]).days
                ts.loc[i, "days_to_next_event"] = (bb.ds.values[0] - ts.ts[i]).days
            else:
                ts.loc[i, "last_event"] = "event_this_week"
                ts.loc[i, "next_event"] = "event_this_week"
                ts.loc[i, "weeks_since_last_event"] = 0
                ts.loc[i, "weeks_to_next_event"] = 0
                ts.loc[i, "days_since_last_event"] = 0
                ts.loc[i, "days_to_next_event"] = 0
        tmp.append(ts)

    df_with_event = pd.concat(tmp).reset_index(drop=True)
    df_with_event["days_to_next_event"] = df_with_event["days_to_next_event"].astype(int)
    df_with_event["days_since_last_event"] = df_with_event["days_since_last_event"].astype(int)
    df_with_event["weeks_to_next_event"] = df_with_event["weeks_to_next_event"].astype(int)
    df_with_event["weeks_since_last_event"] = df_with_event["weeks_since_last_event"].astype(int)
    
    del tmp, df_since_launch, ts, a, b, aa, bb

    #### 5. expected demand from last year or 2 years ago
    if verbose:
        print("Get the first spike using historical demands")

    np.random.seed(42)

    _df = df_with_event.copy()
    if freq == "W-Mon":    
        print("monthly")
        _df["demand_2_year"] = _df.groupby(["id"])["y"].shift(104).fillna(0)
        _df["demand_1_year"] = _df.groupby(["id"])["y"].shift(52).fillna(0)
    else:
        print("weekly")
        _df["demand_2_year"] = _df.groupby(["id"])["y"].shift(24).fillna(0)
        _df["demand_1_year"] = _df.groupby(["id"])["y"].shift(12).fillna(0)
        
    _df["average_expected_demand"] = _df["demand_1_year"]
    _df.loc[(_df["demand_2_year"] > 0) & (_df["demand_1_year"] > 0),"average_expected_demand"] = (_df["demand_1_year"] + _df["demand_2_year"]) / 2
    _df.loc[(_df["average_expected_demand"] == 0), "average_expected_demand"] = _df["demand_1_year"]
    _df.loc[(_df["average_expected_demand"] == 0), "average_expected_demand"] = _df["demand_2_year"]

    _df["dummy1"] = _df["demand_1_year"].copy()
    _df["dummy2"] = _df["demand_2_year"].copy()
    _df.loc[_df["dummy1"] < 0, "dummy1"] = 0
    _df.loc[_df["dummy2"] < 0, "dummy2"] = 0
    _df.loc[_df["dummy1"] > 0, "dummy2"] = 0
    _df["expected_demand"] = _df["dummy1"] + _df["dummy2"]
    # ## Need this part otherwise all expected demand in train will be 0
    # _df.loc[
    #     (_df["expected_demand"] == 0) &
    #     (_df.ts < np.min(all_dates["val"])), "expected_demand"] = _df["y"]

    _df.loc[(_df["expected_demand"] == 0), "expected_demand"] = _df["average_train_demand"]
    # aaa = _df.loc[(_df["expected_demand"] == 0) & (_df.ts.isin(all_dates["val"]))]
    # _df.loc[
    #     (_df["expected_demand"] == 0) &
    #     (_df.ts.isin(all_dates["val"])), "expected_demand"] = aaa["y"] * np.abs(np.random.randn(len(aaa)))
    # aaa = _df.loc[(_df["expected_demand"] == 0) & (_df.ts.isin(all_dates["test"]))]
    # _df.loc[
    #     (_df["expected_demand"] == 0) &
    #     (_df.ts.isin(all_dates["test"])), "expected_demand"] = aaa["y"] * np.abs(np.random.randn(len(aaa)))

    _df["expected_demand"] = _df["expected_demand"].fillna(0)
    del _df["dummy1"], _df["dummy2"]
    # del aaa
    
    df_with_last_demand = _df.copy()
    del df_with_event, _df

    #### 6. add spikes to events
    if verbose:
        print("Get the second spike using events")
    cols = ["y", "days_to_next_event", "days_since_last_event", "weeks_to_next_event", "weeks_since_last_event"]

    _df = df_with_last_demand.copy()
    _df = _df.groupby(["id"]).apply(find_spike_with_demand, val_date=all_dates["val"].min())
    tmp = _df.loc[
        (_df.year >= 2020) &
        (_df.dummy_spike == 1)
        ].groupby(["id", "events", "next_event"]).mean()[cols].reset_index()

    # add spikes to events
    _df["dummy_event_spike"] = 0
    _df.loc[(_df.events != "no_events") & (_df.year >= 2020), "dummy_event_spike"] = 1

    # add spikes a few weeks before each event
    _df["dummy_spike_before_event"] = 0
    _df.reset_index(drop=True, inplace=True)
    for i in range(len(_df)):
        if _df["year"][i] < 2020:
            # don't want noise from 2019
            continue
        if _df.events[i] != "no_events":
            a = tmp.loc[
                (tmp["id"] == _df["id"][i]) &
                (tmp["next_event"] == _df.events[i])
                ].reset_index(drop=True)
            if len(a) > 0:
                # should just be 1 row
                b = a["weeks_to_next_event"][0]
                if (i - b) >= 0:
                    _df.loc[i - b:i, "dummy_spike_before_event"] = 1
    _df["dummy_expect_a_spike"] = np.clip(_df["dummy_event_spike"] + _df["dummy_spike_before_event"], 0, 1)
    df_with_spikes = _df
    del tmp, _df, a, b, df_with_last_demand

    #### 7. add price
    if verbose:
        print("Process price")
    _df = df_with_spikes.copy()
    _df["new_price"] = _df["price"]

    avg_price = _df.loc[(_df.dummy_expect_a_spike == 0)].groupby(["id"]).mean()["price"].reset_index()

    tmp = _df.loc[
        (_df.year >= 2020) & (_df.dummy_expect_a_spike == 1)
        ].groupby(["id", "events", "next_event"]).mean()[["price"]].reset_index()
    tmp2 = _df.loc[(_df.year >= 2020)].groupby(["id", "events", "next_event"]).mean()[["price"]].reset_index()
    for i in range(len(_df)):
        if np.isnan(_df.price[i]):
            if _df.dummy_expect_a_spike[i] == 1:
                a = tmp.loc[(tmp["id"] == _df["id"][i]) &
                            (tmp.events == _df["events"][i]) &
                            (tmp.next_event == _df["next_event"][i])]
                if len(a) == 0:
                    a = tmp.loc[(tmp["id"] == _df["id"][i]) &
                                (tmp.next_event == _df["events"][i])]
                if np.isnan(a["price"].values[0]):
                    _df.loc[i, "new_price"] = tmp2.loc[
                        (tmp2["id"] == _df["id"][i]) &
                        (tmp2.events == _df["events"][i]) &
                        (tmp2.next_event == _df["next_event"][i])]["price"].values[0]
                else:
                    _df.loc[i, "new_price"] = a["price"].values[0]
            else:
                _df.loc[i, "new_price"] = avg_price.loc[avg_price["id"] == _df["id"][i]]["price"].values[0]

    df_with_prices = _df
    del _df, df_with_spikes, tmp2, tmp, avg_price

    #### 8. Add spikes to price
    if verbose:
        print("Get the third spike using price")

    _df = df_with_prices.copy()
    _df["dummy_expect_a_spike_price"] = _df["dummy_expect_a_spike"]
    for _id in _df.id.unique():
        ts = _df.loc[_df.id == _id]
        q = np.quantile(ts.price.dropna(), 0.1)
        _df.loc[(_df.id == _id) & (_df.dummy_expect_a_spike == 0) & (
                _df.new_price < q), "dummy_expect_a_spike_price"] = 1
    df_with_spikes = _df
    del df_with_prices, _df, ts, q
  
    return df_with_spikes


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


def get_ts_features(df, targets=None):
    if targets is None:
        targets = ["y", "price"]
