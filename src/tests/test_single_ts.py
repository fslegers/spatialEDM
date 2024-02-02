from src.classes import TimeSeries, ConcatenatedTimeSeries, Point, EmbeddingVector, EDM
import src.classes as me
from src.simulate_lorenz import *
import numpy as np

if __name__ == "__main__":

    x = np.arange(1,15)
    y_1 = np.sin(x/10.0)

    x = np.arange(5,25)
    y_2 = np.sin(x/10.0)

    y_1_test = y_1[-5:]
    y_2_test = y_2[-5:]
    y_1 = y_1[:-5]
    y_2 = y_2[:-5]

    ts_1 = me.transform_array_to_ts(y_1, loc="A")
    ts_2 = me.transform_array_to_ts(y_2, loc="B")
    ts = ConcatenatedTimeSeries([ts_1, ts_2])

    ts_1 = me.transform_array_to_ts(y_1_test, loc="A")
    ts_2 = me.transform_array_to_ts(y_2_test, loc="B")
    ts_test = ConcatenatedTimeSeries([ts_1, ts_2])

    model = EDM(1, 1, 10, "LB", "0.5")
    model.train(ts)
    model.predict(ts, ts_test)

