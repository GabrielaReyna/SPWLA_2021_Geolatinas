"""
@author: Ana Gabriela Reyna
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns




def rows_count(well_data):
    well_data = well_data.iloc[:, 3::] #drop first three columns (index, well_num, depth)
    columns_names = well_data.columns
    well_data = well_data.to_numpy()
    print("rows on well:", well_data.shape)

    boolean_arr =  np.invert(np.isnan(well_data)) #boolean array where True when numeric number and False where nan
    count_nans = boolean_arr.sum(axis=0) #count the non_nans across the columns
    number_rows = np.max(np.delete(count_nans,2, axis=0))#drop bit size
    count_nans = count_nans /number_rows
    return count_nans, columns_names

def plot_data(data_dir, title, save_path):
    wells_files = sorted([img for img in os.listdir(data_dir) if img != ".DS_Store" and img.endswith(".csv")])
    plot_array = []
    columns = []
    for file in wells_files:
        print(file)
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)
        count, columns = rows_count(df)
        plot_array.append(count)
    print(len(plot_array))
    plot_array = np.asarray(plot_array)
    plot_df = pd.DataFrame(plot_array, columns=columns)
    wells_files =  [well.replace(".csv","") for well in wells_files]
    ax = sns.heatmap(plot_df, linewidths=.5, yticklabels=wells_files)
    ax.set_title(title)

    plt.savefig(save_path)
    plt.show()

dataset_dir = "./dataset/separated_wells/test/"
save_path = "/Users/astromeria/PycharmProjects/SPWLA_2021_Geolatinas/dataset/plots/heat_maps/test.png"
print(save_path)
plot_data(dataset_dir,"Test data", save_path)




