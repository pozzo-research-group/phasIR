import tkinter as tk
from tkinter import filedialog


# Function: Exports data as a csv for user use
# Step: creates UI buttons to save, exports to user chosen place
# Input: dataframe of final data
# Output: no output - prompt to save the data to a csv
def export_csv(final_data):
    '''
    Exports the dataframe of the run into a csv file to be saved or
    modifed as desired

    Parameters
    -----------
    final_data : Dataframe
        A dataframe containing the final melting point
        data of all the samples.

    Returns
    --------
    Prompt to save the data to a csv.
    '''
    global df
    df = final_data

    root = tk.Tk()

    canvas1 = tk.Canvas(root, width=300, height=300, bg='lightgray',
                        relief='raised')
    canvas1.pack()

    root.attributes("-topmost", True)

    def exportCSV():
        global df

        export_file_path = filedialog.asksaveasfilename(
            defaultextension='.csv')
        df.to_csv(export_file_path, index=None, header=True)

    saveAsButton_CSV = tk.Button(text='Export CSV', command=exportCSV,
                                 bg='green', fg='white',
                                 font=('helvetica', 12, 'bold'))
    canvas1.create_window(150, 150, window=saveAsButton_CSV)

    root.mainloop()
    return

#  !! These functions/tests are still under development
# They relate specifically to the OPENTRONS pipetting robot,
# and will be refined for inner-lab use of the package/OPENTRONS combination.


# Function: adapts the pipetting order to the temperature reading order
# Step: collects all of the created lists, combines and labels the data
# Input: produced molfractions, and list of component names
# Output: reordered list of named compoents
# def molfrac_prep(DES_molfrac):
#     '''Adapts the pipetting order to the temperature reading order'''
#     #!! Assumes a 96 well plate in the opentrons pipetting
#         robot with 2 components
#     number = len(DES_molfrac[0])
#     array1 = np.zeros((8,12))
#     array2 = np.zeros((8,12))
#     comp1 = []
#     comp2 = []
#     ordered1 = []
#     ordered2 = []

#     for i in range(number):
#         if i == 0:
#             for row in DES_molfrac:
#                 hold = row[i]
#                 comp1.append(hold)
#         elif i == 1:
#             for row in DES_molfrac:
#                 hold = row[i]
#                 comp2.append(hold)
#         else:
#             pass
#     index = 0
#     for i in range(len(array1[0])):
#         for j in range(len(array1)):
#             array1[j, i] = comp1[index]
#             index = index +1
#     index= 0
#     for i in range(len(array2[0])):
#         for j in range(len(array2)):
#             array2[j, i] = comp2[index]
#             index = index +1
#     index=0
#     for i in range(len(array1)):
#         hold1 = array1[i]
#         ordered1.extend(hold1)
#     index=0
#     for i in range(len(array2)):
#         hold2 = array2[i]
#         ordered2.extend(hold2)

#     mol_data = pd.Dataframe()
#     mol_data['Component 1'] = ordered1
#     mol_data['Component 2'] = ordered2
#     return ordered1, ordered2


# Function: Creates the final dataframe of data
# Step: collects all of the created lists, combines and labels the data
# Input: all of the final data lists
# Output: dataframe of final info (mol inputs, melting temps, sample index)
# def create_dataframe_DES(all_melt, all_possible,
#                          samples, ordered1, ordered2):
#     '''Exports the dataframe of the run into a
#        csv file to be saved or modifed as desired'''
#     all_samples = []
#     for i in range(samples):
#         all_samples.append(i)

#     final_data = pd.DataFrame()

#     final_data['Sample Index'] = all_samples
#     final_data['Component 1'] = ordered1
#     final_data['Component 2'] = ordered2
#     final_data['Melting Temperature'] = all_melt
#     final_data['Other index possibilites'] = all_possible
#     return final_data

# Function: Creates the final dataframe of data NOT OPENTRONS DATA
# Step: collects all of the created lists, combines and labels the data
# Input: all of the final data lists
# Output: dataframe of final info (mol inputs, melting temps, sample index)
# def create_dataframe(all_melt, all_possible, samples):
#     '''Exports the dataframe of the run into a csv
#        file to be saved or modifed as desired'''
#     all_samples = []
#     for i in range(samples):
#         all_samples.append(i)

#     final_data = pd.DataFrame()

#     final_data['Sample Index'] = all_samples
#     final_data['Melting Temperature'] = all_melt
#     final_data['Other index possibilites'] = all_possible

#     return final_data
