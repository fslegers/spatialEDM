import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
import os

os.chdir("C:/Users/5605407/Documents/PhD/PythonProjects/timeseriesanalysis/results/output/single ts, length and interval length per sd")

files = os.listdir()

for file in files:

    df = pd.read_csv(file, index_col=0)

    sd = file.replace("df_", "")
    sd = sd.replace(" .csv", "")
    sd = sd.replace(",", ".")


    lengths_str = list(df.columns)
    lengths = [int(i) for i in lengths_str]


    intervals_str = list(df.index)
    intervals = [int(j) for j in intervals_str]

    cmap = matplotlib.colormaps['Blues']

    for row in range(len(intervals)):
        col = cmap((len(intervals) - row)/len(intervals))

        #TODO: Add line to plot with label/color based on interval length
        plt.plot(lengths, df.iloc[row].values, color=col, label=str(intervals[row]))
        plt.ylim((-0.3, 1.1))
        #plt.scatter(lengths, df.iloc[row].values, color=col)


    plt.xlabel("Time series length")
    plt.ylabel("Correlation coefficient")
    plt.title("Standard deviation: " + sd)
    plt.legend(title = "Sampling interval", loc='center left', bbox_to_anchor=(1, 0.5))

    path_name = "figure_sd_" + str(sd)
    path_name = path_name.replace('.', ',')
    plt.savefig(path_name, bbox_inches='tight')

    plt.show()


    #TODO: save plots, fit legend in picture
