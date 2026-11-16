import numpy as np

np.random.seed(7)
readings = np.random.randint(20, 41, size=(6, 3))
print(readings)

# (1) Mean of each sensor
mean_per_sensor = readings.mean(axis=0)
print("Mean per sensor:", mean_per_sensor)

# (2) Standardize using broadcasting and check column means ≈ 0
std_per_sensor = readings.std(axis=0)
normalized = (readings - mean_per_sensor) / std_per_sensor
print("Normalized column means:", normalized.mean(axis=0).round(10))  # should be ~0

# (3) Hour in which each sensor peaked
hour_of_max = readings.argmax(axis=0)
print("Hour of max per sensor:", hour_of_max)

# (4) Append the mean row
mean_row = mean_per_sensor.reshape(1, -1)
readings_with_mean = np.vstack([readings, mean_row])
print(readings_with_mean.shape)

# (5) Reproducibility check
np.random.seed(7)
readings2 = np.random.randint(20, 41, size=(6, 3))
print("Reproducible:", np.array_equal(readings, readings2))

# Overall hottest reading (required for the branch commit)
print("Overall hottest reading:", readings.max())