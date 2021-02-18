
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tkinter as tk
from tkinter import filedialog
from pandas import DataFrame
import pandas as pd
import post_data
import numpy as np

# !! These functions/tests are still under development
# They relate specifically to the OPENTRONS pipetting robot, and will be refined
# for inner-lab use of the package/OPENTRONS combination.

# def test_molfrac_prep():
#     DES_molfrac = [[0.5,0.5],[0.2,0.8],[0.4,0.6]]
#     ordered1,ordered2 = post_data.molfrac_prep(DES_molfrac)
#     assert len(ordered1)==len(DES_molfrac),'Number of mole fractions for component is wrong'
#     assert len(ordered2)==len(DES_molfrac),'Number of mole fractions for component is wrong'
#     return
#
# def test_create_dataframe_DES():
#     all_melt = [48,49,50]
#     all_possible = [48,49,50]
#     DES_molfrac = [[0.5,0.5],[0.2,0.8],[0.4,0.6]]
#     names = ['Citric Acid','Quinone']
#     ordered1,ordered2 = post_data.molfrac_prep(DES_molfrac)
#     all_samples = [1,2,3]
#     final_data = post_data.create_dataframe_DES(all_melt,all_possible,samples,ordered1,ordered2)
#     assert isinstance(final_data,pd.Dataframe),'Output is not a dataframe'
#     assert len(final_data)==len(all_samples),'Wrong number of samples in the output dataframe'

# def test_create_dataframe():
#     all_melt = [48,49,50]
#     all_possible = [48,49,50]
#     samples = 3
#     final_data = post_data.create_dataframe(all_melt, all_possible, samples)
#     assert isinstance(final_data,pd.DataFrame),'Output is not a dataframe'
#     assert len(final_data)== samples,'Wrong number of samples in the output dataframe'
