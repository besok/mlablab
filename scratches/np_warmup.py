import numpy as np
from numpy.typing import NDArray


def min_max_normalize(data: NDArray[np.float32]) -> NDArray[np.float32]:
    min_v = np.min(data)
    max_v = np.max(data)
    delta = max_v - min_v
    return (data - min_v) / delta if delta != 0 else np.zeros_like(data)


def moving_average(data: NDArray[np.float32], sliding_window: int = 3) -> NDArray[np.float32]:
    sec = np.ones(sliding_window) / sliding_window
    return np.convolve(data, sec, mode='valid')


def filter_outliers(data: NDArray[np.float32], threshold=2.0) -> NDArray[np.bool_]:
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return np.zeros_like(data, dtype=np.bool_)
    z_scores = (data - mean) / std
    return np.abs(z_scores) <= threshold


def main():
    print("Sensor Data Normalization (Min-Max Scaling)")
    print(min_max_normalize(np.array([1., 2., 3., 4.])))

    print("\n---\nMoving Average Smoothing")
    temperatures = np.array([20.1, 20.5, 21.0, 22.2, 21.8, 22.5, 23.0])
    print(moving_average(temperatures))

    print("\n---\nOutlier Filtering via Z-Scores")
    measurements = np.array([10.2, 9.8, 10.1, 25.5, 10.3, 9.9, -5.0])
    mask = filter_outliers(measurements, threshold=2.0)
    print(measurements[mask])

    print("\n---\nMulti-Sensor Industrial Log Analysis")
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


if __name__ == "__main__":
    main()
