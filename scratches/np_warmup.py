import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sensor Data Normalization (Min-Max Scaling)
    **Scenario**: You have an array of raw sensor readings and need to scale all values to a range between 0 and 1.

    **Task**: Write a function that takes a 1D array of numbers and scales them using the formula:

    $$\text{scaled} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$
    """)
    return


@app.cell
def _():
    import numpy as np
    from numpy.typing import NDArray


    def min_max_normalize(data: NDArray[np.float32]) -> NDArray[np.float32]:
        min_v = np.min(data)
        max_v = np.max(data)

        return (data - min_v) / delta if (delta := max_v - min_v) != 0 else np.zeros_like(data)

    return NDArray, min_max_normalize, np


@app.cell
def _(min_max_normalize, np):
    min_max_normalize(np.array([1., 2., 3., 4.]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Moving Average Smoothing
    **Scenario**: A time-series of temperature measurements contains high-frequency noise. You need to smooth the curve using a rolling window of size 3.

    **Task**: Use np.convolve to compute a moving average. Ensure the output length matches the input or handles boundaries cleanly (using the 'valid' mode).
    """)
    return


@app.cell
def _(NDArray, np):
    def moving_average(data: NDArray[np.float32], sliding_window: int = 3) -> NDArray[np.float32]:
        sec = np.ones(sliding_window) / sliding_window
        return np.convolve(data, sec, mode='valid')


    temperatures = np.array([20.1, 20.5, 21.0, 22.2, 21.8, 22.5, 23.0])
    moving_average(temperatures)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Outlier Filtering via Z-Scores
    **Scenario**: You need to clean a dataset by removing values that deviate significantly from the mean.

    **Task**: Compute the Z-score for each element in an array (how many standard deviations away from the mean it is) and return a boolean mask for elements where the absolute Z-score is less than or equal to 2.
    """)
    return


@app.cell
def _(NDArray, np):
    def filter_outliers(data: NDArray[np.float32], threshold=2.0) -> NDArray[np.bool_]:
        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return np.zeros_like(data, dtype=np.bool_)
        z_scores = (data - mean) / std
        return np.abs(z_scores) <= threshold


    measurements = np.array([10.2, 9.8, 10.1, 25.5, 10.3, 9.9, -5.0])
    mask = filter_outliers(measurements, threshold=2.0)

    measurements[mask]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Sensor Industrial Log Analysis
    **Scenario**: You have a 2D NumPy array representing sensor logs from a production line over 100 time steps. The array shape is (100, 4), where the columns correspond to:
    - Temperature (°C)
    - Vibration ($\text{mm/s}$)
    - Pressure ($\text{bar}$)
    - Current ($\text{A}$)

    **Tasks**
    **Column-wise Statistics:** Compute the mean and maximum values for each of the four sensors independently using the axis parameter.

    **Peak Vibration Index:** Find the row index (time step) that recorded the absolute highest vibration value.

    **Safety Threshold Mask:** Create a boolean mask identifying all time steps where any sensor exceeds its critical threshold:
     - Temperature > 82.0
     - Vibration > 6.0
     - Pressure > 48.0
     - Current > 15.0
    """)
    return


@app.cell
def _(np):
    np.random.seed(42)
    n_steps = 100
    temp = np.random.normal(70, 5, n_steps)
    vibration = np.random.exponential(1.5, n_steps)
    pressure = np.random.uniform(30.0, 50.0, n_steps)
    current = np.random.normal(12.0, 1.2, n_steps)
    sensor_data = np.round(np.column_stack((temp, vibration, pressure, current)), decimals=1)
    means = np.round(np.mean(sensor_data, axis=0), decimals=1)
    maxs = np.round(np.max(sensor_data, axis=0), decimals=1)
    print(f'means: {means}, maxs: {maxs}')
    peak_vibr_idx = np.argmax(sensor_data[:, 1], axis=0)
    peak_vibr_val = sensor_data[peak_vibr_idx, 1]
    print(f' vibr idx = {peak_vibr_idx} and val {peak_vibr_val}')
    thresholds = np.array([82.0, 6.0, 48.0, 15.0])
    critical_events = sensor_data[np.any(sensor_data > thresholds, axis=1)]
    print(f'critical events: {critical_events}')
    return


if __name__ == "__main__":
    app.run()
