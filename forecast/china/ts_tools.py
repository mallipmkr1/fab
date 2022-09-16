import warnings

import numpy as np
import pandas as pd

from tsfresh import defaults
from tsfresh.utilities.distribution import (
    DistributorBaseClass,
    MapDistributor,
    MultiprocessingDistributor,
)

from utils.timeseries import load_data, resample_data, project_future
from utils.tools import generate_dates


def _roll_out_time_series(
        timeshift,
        grouped_data,
        rolling_direction,
        gap,
        max_timeshift,
        min_timeshift,
        column_sort,
        column_id,
):
    """
    Internal helper function for roll_time_series.
    This function has the task to extract the rolled forecast data frame of the number `timeshift`.
    This means it has shifted a virtual window if size `max_timeshift` (or infinite)
    `timeshift` times in the positive direction (for positive `rolling_direction`) or in negative direction
    (for negative `rolling_direction`).
    It starts counting from the first data point for each id (and kind) (or the last one for negative
    `rolling_direction`).
    The rolling happens for each `id` and `kind` separately.
    Extracted data smaller than `min_timeshift` + 1 are removed.

    Implementation note:
    Even though negative rolling direction means, we let the window shift in negative direction over the data,
    the counting of `timeshift` still happens from the first row onwards. Example:

        1   2   3   4

    If we do positive rolling, we extract the sub time series

      [ 1 ]               input parameter: timeshift=1, new id: ([id=]X,[timeshift=]1)
      [ 1   2 ]           input parameter: timeshift=2, new id: ([id=]X,[timeshift=]2)
      [ 1   2   3 ]       input parameter: timeshift=3, new id: ([id=]X,[timeshift=]3)
      [ 1   2   3   4 ]   input parameter: timeshift=4, new id: ([id=]X,[timeshift=]4)

    If we do negative rolling:

      [ 1   2   3   4 ]   input parameter: timeshift=1, new id: ([id=]X,[timeshift=]1)
          [ 2   3   4 ]   input parameter: timeshift=2, new id: ([id=]X,[timeshift=]2)
              [ 3   4 ]   input parameter: timeshift=3, new id: ([id=]X,[timeshift=]3)
                  [ 4 ]   input parameter: timeshift=4, new id: ([id=]X,[timeshift=]4)

    If you now reverse the order of the negative examples, it looks like shifting the
    window from the back (but it is implemented to start counting from the beginning).

    """

    def _f(x):
        if rolling_direction > 0:
            # For positive rolling, the right side of the window moves with `timeshift`
            shift_until = timeshift + gap
            shift_from = max(shift_until - max_timeshift - 1, 0)

            df_temp = x.iloc[shift_from:shift_until] if shift_until <= len(x) else None
        else:
            # For negative rolling, the left side of the window moves with `timeshift`
            shift_from = max(timeshift + gap - 1, 0)
            shift_until = shift_from + max_timeshift + 1

            df_temp = x.iloc[shift_from:shift_until]

        if df_temp is None or len(df_temp) < min_timeshift + 1:
            return

        df_temp = df_temp.copy()

        # and set the shift correctly
        if column_sort and rolling_direction > 0:
            timeshift_value = df_temp[column_sort].iloc[-1]
        elif column_sort and rolling_direction < 0:
            timeshift_value = df_temp[column_sort].iloc[0]
        else:
            timeshift_value = timeshift - 1

        if rolling_direction > 0:
            df_temp = df_temp.iloc[0:-gap].copy()
        else:
            df_temp = df_temp.iloc[-gap:].copy()

        # and now create new ones ids out of the old ones
        df_temp["id"] = df_temp[column_id].apply(lambda row: (row, timeshift_value))

        return df_temp

    return [grouped_data.apply(_f)]


def roll_time_series(
        df_or_dict,
        column_id="id",
        column_sort=None,
        column_kind=None,
        rolling_direction=1,
        gap=None,
        max_timeshift=None,
        min_timeshift=0,
        chunksize=defaults.CHUNKSIZE,
        n_jobs=defaults.N_PROCESSES,
        show_warnings=defaults.SHOW_WARNINGS,
        disable_progressbar=defaults.DISABLE_PROGRESSBAR,
        distributor=None,
):
    """
    This method creates sub windows of the time series. It rolls the (sorted) data frames for each kind and each id
    separately in the "time" domain (which is represented by the sort order of the sort column given by `column_sort`).

    For each rolling step, a new id is created by the scheme ({id}, {shift}), here id is the former id of
    the column and shift is the amount of "time" shifts.
    You can think of it as having a window of fixed length (the max_timeshift) moving one step at a time over
    your time series.
    Each cut-out seen by the window is a new time series with a new identifier.

    A few remarks:

     * This method will create new IDs!
     * The sign of rolling defines the direction of time rolling, a positive value means we are shifting
       the cut-out window foreward in time. The name of each new sub time series is given by the last time point.
       This means, the time series named `([id=]4,[timeshift=]5)` with a `max_timeshift` of 3 includes the data
       of the times 3, 4 and 5.
       A negative rolling direction means, you go in negative time direction over your data.
       The time series named `([id=]4,[timeshift=]5)` with `max_timeshift` of 3 would then include the data
       of the times 5, 6 and 7.
       The absolute value defines how much time to shift at each step.
     * It is possible to shift time series of different lengths, but:
     * We assume that the time series are uniformly sampled
     * For more information, please see :ref:`forecasting-label`.

    :param gap:
    :param df_or_dict: a pandas DataFrame or a dictionary. The required shape/form of the object depends on the rest of
        the passed arguments.
    :type df_or_dict: pandas.DataFrame or dict

    :param column_id: it must be present in the pandas DataFrame or in all DataFrames in the dictionary.
        It is not allowed to have NaN values in this column.
    :type column_id: basestring

    :param column_sort: if not None, sort the rows by this column. It is not allowed to
        have NaN values in this column. If not given, will be filled by an increasing number,
        meaning that the order of the passed dataframes are used as "time" for the time series.
    :type column_sort: basestring or None

    :param column_kind: It can only be used when passing a pandas DataFrame (the dictionary is already assumed to be
        grouped by the kind). Is must be present in the DataFrame and no NaN values are allowed.
        If the kind column is not passed, it is assumed that each column in the pandas DataFrame (except the id or
        sort column) is a possible kind.
    :type column_kind: basestring or None

    :param rolling_direction: The sign decides, if to shift our cut-out window backwards or forwards in "time".
        The absolute value decides, how much to shift at each step.
    :type rolling_direction: int

    :param max_timeshift: If not None, the cut-out window is at maximum `max_timeshift` large. If none, it grows
         infinitely.
    :type max_timeshift: int

    :param min_timeshift: Throw away all extracted forecast windows smaller or equal than this. Must be larger
         than or equal 0.
    :type min_timeshift: int

    :param n_jobs: The number of processes to use for parallelization. If zero, no parallelization is used.
    :type n_jobs: int

    :param chunksize: How many shifts per job should be calculated.
    :type chunksize: None or int

    :param show_warnings: Show warnings during the feature extraction (needed for debugging of calculators).
    :type show_warnings: bool

    :param disable_progressbar: Do not show a progressbar while doing the calculation.
    :type disable_progressbar: bool

    :param distributor: Advanced parameter: set this to a class name that you want to use as a
             distributor. See the utilities/distribution.py for more information. Leave to None, if you want
             TSFresh to choose the best distributor.
    :type distributor: class

    :return: The rolled data frame or dictionary of data frames
    :rtype: the one from df_or_dict
    """

    if rolling_direction == 0:
        raise ValueError("Rolling direction of 0 is not possible")

    if max_timeshift is not None and max_timeshift <= 0:
        raise ValueError("max_timeshift needs to be positive!")

    if min_timeshift < 0:
        raise ValueError("min_timeshift needs to be positive or zero!")

    if isinstance(df_or_dict, dict):
        if column_kind is not None:
            raise ValueError(
                "You passed in a dictionary and gave a column name for the kind. Both are not possible."
            )

        return {
            key: roll_time_series(
                df_or_dict=df_or_dict[key],
                column_id=column_id,
                column_sort=column_sort,
                column_kind=column_kind,
                rolling_direction=rolling_direction,
                gap=gap,
                max_timeshift=max_timeshift,
                min_timeshift=min_timeshift,
                chunksize=chunksize,
                n_jobs=n_jobs,
                show_warnings=show_warnings,
                disable_progressbar=disable_progressbar,
                distributor=distributor,
            )
            for key in df_or_dict
        }

    # Now we know that this is a pandas data frame
    df = df_or_dict

    if len(df) <= 1:
        raise ValueError(
            "Your time series container has zero or one rows!. Can not perform rolling."
        )

    if column_id is not None:
        if column_id not in df:
            raise AttributeError(
                "The given column for the id is not present in the data."
            )
    else:
        raise ValueError(
            "You have to set the column_id which contains the ids of the different time series"
        )

    if column_kind is not None:
        grouper = [column_kind, column_id]
    else:
        grouper = [
            column_id,
        ]

    if column_sort is not None:
        # Require no Nans in column
        if df[column_sort].isnull().any():
            raise ValueError("You have NaN values in your sort column.")

        df = df.sort_values(column_sort)

        if df[column_sort].dtype != np.object:
            # if rolling is enabled, the data should be uniformly sampled in this column
            # Build the differences between consecutive time sort values

            differences = df.groupby(grouper)[column_sort].apply(
                lambda x: x.values[:-1] - x.values[1:]
            )
            # Write all of them into one big list
            differences = sum(map(list, differences), [])
            # Test if all differences are the same
            if differences and min(differences) != max(differences):
                warnings.warn(
                    "Your time stamps are not uniformly sampled, which makes rolling "
                    "nonsensical in some domains."
                )

    # Roll the data frames if requested
    rolling_amount = np.abs(rolling_direction)
    rolling_direction = np.sign(rolling_direction)

    grouped_data = df.groupby(grouper)
    prediction_steps = grouped_data.count().max().max()

    max_timeshift = max_timeshift or prediction_steps

    # Todo: not default for columns_sort to be None
    if column_sort is None:
        df["sort"] = range(df.shape[0])

    if rolling_direction > 0:
        range_of_shifts = list(reversed(range(prediction_steps, 0, -rolling_amount)))
    else:
        range_of_shifts = range(1, prediction_steps + 1, rolling_amount)

    if distributor is None:
        if n_jobs == 0 or n_jobs == 1:
            distributor = MapDistributor(
                disable_progressbar=disable_progressbar, progressbar_title="Rolling"
            )
        else:
            distributor = MultiprocessingDistributor(
                n_workers=n_jobs,
                disable_progressbar=disable_progressbar,
                progressbar_title="Rolling",
                show_warnings=show_warnings,
            )

    if not isinstance(distributor, DistributorBaseClass):
        raise ValueError("the passed distributor is not an DistributorBaseClass object")

    kwargs = {
        "grouped_data": grouped_data,
        "rolling_direction": rolling_direction,
        "gap": gap,
        "max_timeshift": max_timeshift,
        "min_timeshift": min_timeshift,
        "column_sort": column_sort,
        "column_id": column_id,
    }

    shifted_chunks = distributor.map_reduce(
        _roll_out_time_series,
        data=range_of_shifts,
        chunk_size=chunksize,
        function_kwargs=kwargs,
    )

    distributor.close()

    df_shift = pd.concat(shifted_chunks, ignore_index=True)

    return df_shift.sort_values(by=["id", column_sort or "sort"])


if __name__ == "__main__":
    main_group_index = ['material', 'customer_group']

    df = load_data()

    all_dates = generate_dates(n_rounds=0)

    df = resample_data(df)

    df = project_future(df, all_dates)

    print(df)

    df_rolled = roll_time_series(
        df[12],
        n_jobs=1,
        column_id="id",
        column_sort="ts",
        rolling_direction=1,
        gap=4,
        max_timeshift=8,
        min_timeshift=8,
    )

    print(df_rolled.loc[df_rolled["id"] == (21, pd.to_datetime("2021-12-20"))])


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
