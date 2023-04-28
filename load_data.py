import os
import pandas as pd
from sklearn.model_selection import train_test_split


def import_data(input):
   read = pd.read_csv(os.path.join(input.data, "simulation_log.csv"))
   p1 = read["steering"].values
   p1_training = train_test_split(p1, p2, size=input.size, random_state=0)
   p1_testing = train_test_split(p1, p2, size=input.size, random_state=0)


   p2 = read[["center", "left", "right"]].values
   p2_training = train_test_split(p1, p2, size=input.size, random_state=0)
   p2_testing = train_test_split(p1, p2, size=input.size, random_state=0)


   return p1_training, p1_testing, p2_training, p2_testing
