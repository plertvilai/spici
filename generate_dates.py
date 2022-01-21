""" Generate time periods for SPICI

USAGE
# set the configuration under `config`. Output is below command
>>> python generate_dates.py
Generating random dates from 2021-01-01 to 2021-08-30     (n=100)
        Times: start_time: 18:00 || offset: H: 5 m: 0
        Size range: 0.03 mm - 1.0
Saved file to ./examples/time_period.txt

"""
import pandas as pd
import numpy as np

np.random.seed(0)

def generate_time_periods(
    output_csv,
    df=None,
    datefmt=None,
    timefmt="%H%M",
    offset_hours=0,
    offset_min=0,
    min_camera=0.03,
    max_camera=0.07,
    camera="SPCP2",
    time_col="Time Collected hhmm (PST)",
    date_col="SampleID (YYYYMMDD)",
):
    """Create a time period text file for pulling SPC images

    Args:
        output_csv: Output file name
        df: Pandas dataframe to calculate offsets
        datefmt: Date format
        timefmt: Time format
        offset_hours: Hours to offset from the start time
        offset_min: Minutes to offset from the start time
        min_camera: Minimum size range
        max_camera: Maximum size range
        camera: Camera to pull data from. Default: SPCP2
        time_col: Name of the time column
        date_col: Name of the date column

    Returns:

    """
    # for each date, get the time, add the offset hours/minutes
    df["end_time"] = (
        pd.to_datetime(df[time_col], format=timefmt)
        + pd.DateOffset(hours=offset_hours, minutes=offset_min)
    ).dt.time
    df["start_time"] = (
        pd.to_datetime(df[time_col], format=timefmt)
        - pd.DateOffset(hours=offset_hours, minutes=offset_min)
    ).dt.time

    # reformat the date if neccessary
    temp = "date"
    if datefmt:
        df[temp] = pd.to_datetime(df[date_col], format=datefmt).dt.strftime("%Y-%m-%d")
    else:
        df[temp] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

    # Set the final datetimes, size range, etc.
    # and save the dataset
    df["start_datetime"] = df[temp].astype(str) + " " + df["start_time"].astype(str)
    df["end_datetime"] = df[temp].astype(str) + " " + df["end_time"].astype(str)
    df["min_camera"] = min_camera
    df["max_camera"] = max_camera
    df["camera"] = camera
    column_order = [
        "start_datetime",
        "end_datetime",
        "min_camera",
        "max_camera",
        "camera",
    ]
    df = df[column_order]
    print(f"Saved file to {output_csv}")
    df.to_csv(output_csv, index=False, header=False)


def random_dates(start, end, n, unit="D", seed=None):
    assert isinstance(start, str), "Must be str ex: " "2021-01-01" ""
    assert isinstance(end, str), "Must be str ex: " "2021-01-01" ""

    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if not seed:  # from piR's answer
        np.random.seed(0)

    ndays = (end - start).days + 1
    return start + pd.to_timedelta(np.random.randint(0, ndays, n), unit=unit)


if __name__ == "__main__":
    # Configuration
    output_csv = "./examples/time_period.txt"
    config = {
        "start_date": "2021-01-01",
        "end_date": "2021-08-30",
        "n_dates": 100,
        "start_time": "18:00",
        "offset_hours": 5,
        "offset_min": 0,
        "min_camera": 0.03,
        "max_camera": 1.0,
    }

    # Verify configuration via sys.stdout
    print(
        f"\nGenerating random dates from {config['start_date']} to "
        f"{config['end_date']} (n={config['n_dates']})"
    )
    print(
        f"\tTimes: start_time: {config['start_time']} || offset: "
        f"H: {config['offset_hours']} m: {config['offset_min']}"
    )
    print(f"\tSize range: {config['min_camera']} mm - " f"{config['max_camera']}")

    # Generate dates
    date_col = "date"
    data = {
        date_col: random_dates(
            start=config["start_date"], end=config["end_date"], n=config["n_dates"]
        )
    }

    # Generate time periods
    df = pd.DataFrame(data)
    time_col = "time"
    df[time_col] = config["start_time"]
    generate_time_periods(
        output_csv=output_csv,
        df=df,
        timefmt="%H:%M",
        offset_hours=config["offset_hours"],
        offset_min=config["offset_min"],
        min_camera=config["min_camera"],
        max_camera=config["max_camera"],
        date_col=date_col,
        time_col=time_col,
    )
