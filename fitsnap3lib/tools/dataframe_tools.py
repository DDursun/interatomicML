import numpy as np
import pandas as pd
from itertools import compress
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

class DataframeTools():
    """
    Class containing methods that help process our dataframe for linear models.

    Attributes
    ----------

    filename: string
        filename/location of the FitSNAP.df dataframe
    
    df: Pickled Pandas dataframe
    """
    def __init__(self, dataframe):

        #if (filename==None and df==None):
        #    raise ValueError('You must supply a filename or a pickled pandas DF.')

        #print(type(dataframe))

        """
        self.filename = filename
        self.df = df

        if(filename is not None):
            print("filename!")
        """
        if(isinstance(dataframe, str)):
            self.dftype = "file"
        elif(type(dataframe)=='pickled_whatever'):
            print(dataframe)
        else:
            raise ValueError('Dataframe must be str or pickled Pandas dataframe')

        self.dataframe = dataframe

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def read_dataframe(self):
        self.df = pd.read_pickle(self.dataframe)
        return(self.df)

    def calc_error(self, quantity, fitting_set="Testing", group_set=None):
        """
        Calculate errors using the dataframe.

        Attributes
        ----------

        quantity: str
            Fitting quantity ("Energy" or "Force") that we want to calculate errors on.

        fitting_set: str
            Fitting set ("Training" or "Testing") that we want to calculate errors on.

        group_set: str
            Group to calculate errors on. If not supplied, will calculate errors on all groups.
        """

        preds = None
        truths = None
        row_type = None
        groups = None
        if (hasattr(self, "df")):
            preds = self.df.loc[:,"preds"]
            truths = self.df.loc[:, "truths"]
            row_type =  self.df.loc[:, "Row_Type"]
            groups = self.df.loc[:, "Groups"]
        else:
            self.read_dataframe()

        test_calc_bool = None
        if (fitting_set=="Training"):
            test_calc_bool = False
        elif(fitting_set=="Testing"):
            test_calc_bool = True
        else:
            raise ValueError('Must declare Training or Testing set.')

        if quantity=="Energy":

            # use Row_Type rows to count number of atoms per config
            # see if this number matches that extracted from pt.shared_arrays

            nconfigs = row_type.tolist().count("Energy")
            natoms_per_config = np.zeros(nconfigs).astype(int)

            config_indx = -1
            for element in row_type.tolist():
                if element=="Energy":
                    config_indx+=1
                elif element=="Force":
                    natoms_per_config[config_indx]+=1
                else:
                    pass 

                assert (config_indx>=0)

            natoms_per_config = natoms_per_config/3
            natoms_per_config = natoms_per_config.astype(int)

            row_indices = row_type[:] == "Energy"
            row_indices = row_indices.tolist()

            test_bool = self.df.loc[:, "Testing"].tolist()
            if (test_calc_bool): # calculate errors on test set
                test_bool = test_bool
            else: # calculate errors on training set
                test_bool = [not item for item in test_bool]
            if not np.any(test_bool):
                raise ValueError(f"{fitting_set} set is empty in this dataframe.")

            # get indices of desired group, otherwise use same indices as test bool

            if (group_set is not None):
                group_row_indices = groups[:] == group_set
                group_row_indices = group_row_indices.tolist()
                if not np.any(group_row_indices):
                    raise ValueError(f"{group_set} set is not in this dataframe.")
            else:
                group_row_indices = test_bool

            # get indices associated with all truths 
            # test_row_indices = [row_indices and test_bool for row_indices, test_bool in zip(row_indices, test_bool)]

            if (group_set is not None):
                test_row_indices = [row_indices and test_bool and group_row_indices \
                                            for row_indices, test_bool, group_row_indices \
                                            in zip(row_indices, test_bool, group_row_indices)]
            else:
                test_row_indices = [row_indices and test_bool for row_indices, test_bool in zip(row_indices, test_bool)]
            
            test_truths = np.array(truths[test_row_indices])
            num_test_components = np.shape(test_truths)[0]

            test_preds = np.array(preds[test_row_indices])

            # extract number of atoms for test configs
            # need to know which configs are testing, and which are training, so we extract correct natoms

            test_configs = []
            indx=0
            count=0
            for element in row_type.tolist():
                if (element=="Energy" and test_row_indices[indx]):
                    count+=1
                    test_configs.append(True)
                if (element=="Energy" and not test_row_indices[indx]):
                    test_configs.append(False)
                indx+=1
            
            assert(len(test_configs) == np.shape(natoms_per_config)[0]) 
            assert((sum(test_row_indices) == count) and (sum(test_configs) == count) )

            natoms_test = natoms_per_config[test_configs]
            
            diff = np.abs(test_preds - test_truths)
            diff_per_atom = diff #/natoms_test
            mae = np.mean(diff_per_atom)

            return(mae)

        if quantity=="Force":

            force_row_indices = row_type[:] == "Force"
            force_row_indices = force_row_indices.tolist()

            testing_bool = self.df.loc[:, "Testing"].tolist()

            # get indices of desired group, otherwise use same indices as test bool

            if (group_set is not None):
                group_row_indices = groups[:] == group_set
                group_row_indices = group_row_indices.tolist()
                if not np.any(group_row_indices):
                    raise ValueError(f"{group_set} set is not in this dataframe.")
            else:
                group_row_indices = testing_bool

            if (test_calc_bool): # calculate errors on test set
                testing_bool = testing_bool
            else: # calculate errors on training set
                testing_bool = [not item for item in testing_bool]
            if not np.any(testing_bool):
                raise ValueError(f"{fitting_set} set is empty in this dataframe.")

            # use list comprehension to extract row indices that are both force and testing
            #testing_force_row_indices = [force_row_indices and testing_bool for force_row_indices, testing_bool in zip(force_row_indices, testing_bool)]
            if (group_set is not None):
                testing_force_row_indices = [force_row_indices and testing_bool and group_row_indices \
                                            for force_row_indices, testing_bool, group_row_indices \
                                            in zip(force_row_indices, testing_bool, group_row_indices)]
            else:
                testing_force_row_indices = [force_row_indices and testing_bool for force_row_indices, testing_bool in zip(force_row_indices, testing_bool)]

            testing_force_truths = np.array(truths[testing_force_row_indices])
            number_of_testing_force_components = np.shape(testing_force_truths)[0]
            testing_force_truths = np.reshape(testing_force_truths, (int(number_of_testing_force_components/3), 3))

            testing_force_preds = np.array(preds[testing_force_row_indices])
            assert(np.shape(testing_force_preds)[0] == number_of_testing_force_components)
            natoms_test = int(number_of_testing_force_components/3)
            testing_force_preds = np.reshape(testing_force_preds, (natoms_test, 3))

            diff = testing_force_preds - testing_force_truths
            #norm = np.linalg.norm(diff, axis=1)
            mae = np.mean(np.abs(diff))
            #mae = np.mean(norm)

            return mae 

    def plot_agreement(self, quantity,fitting_set="Testing", mode="Distribution", group_set=None, legend=True):
        """
        Plot agreement between truth and pred for some quantity. 

        Add optional coloring for groups.

        Attributes
        ----------

        quantity: str
            "Force" or "Energy"

        legend: bool
            Whether to include legend in plot or not

        mode: str
            "Distribution" or "Linear" for different ways of looking at disagreements
        """

        preds = None
        truths = None
        row_type = None
        groups = None
        if (hasattr(self, "df")):
            preds = self.df.loc[:,"preds"]
            truths = self.df.loc[:, "truths"]
            row_type =  self.df.loc[:, "Row_Type"]
            groups = self.df.loc[:, "Groups"]
        else:
            self.read_dataframe()

        test_calc_bool = None
        if (fitting_set=="Training"):
            test_calc_bool = False
        elif(fitting_set=="Testing"):
            test_calc_bool = True
        else:
            raise ValueError('Must declare Training or Testing set.')

        if quantity=="Energy":

            # use Row_Type rows to count number of atoms per config
            # see if this number matches that extracted from pt.shared_arrays

            nconfigs = row_type.tolist().count("Energy")
            natoms_per_config = np.zeros(nconfigs).astype(int)

            config_indx = -1
            for element in row_type.tolist():
                if element=="Energy":
                    config_indx+=1
                elif element=="Force":
                    natoms_per_config[config_indx]+=1
                else:
                    pass 

                assert (config_indx>=0)

            natoms_per_config = natoms_per_config/3
            natoms_per_config = natoms_per_config.astype(int)

            row_indices = row_type[:] == "Energy"
            row_indices = row_indices.tolist()

            test_bool = self.df.loc[:, "Testing"].tolist()
            if (test_calc_bool): # calculate errors on test set
                test_bool = test_bool
            else: # calculate errors on training set
                test_bool = [not item for item in test_bool]
            if not np.any(test_bool):
                raise ValueError(f"{fitting_set} set is empty in this dataframe.")

            # get indices of desired group, otherwise use same indices as test bool

            if (group_set is not None):
                group_row_indices = groups[:] == group_set
                group_row_indices = group_row_indices.tolist()
                if not np.any(group_row_indices):
                    raise ValueError(f"{group_set} set is not in this dataframe.")
            else:
                group_row_indices = test_bool

            # get indices associated with all truths 
            # test_row_indices = [row_indices and test_bool for row_indices, test_bool in zip(row_indices, test_bool)]

            if (group_set is not None):
                test_row_indices = [row_indices and test_bool and group_row_indices \
                                            for row_indices, test_bool, group_row_indices \
                                            in zip(row_indices, test_bool, group_row_indices)]
            else:
                test_row_indices = [row_indices and test_bool for row_indices, test_bool in zip(row_indices, test_bool)]
            
            test_truths = np.array(truths[test_row_indices])
            num_test_components = np.shape(test_truths)[0]

            test_preds = np.array(preds[test_row_indices])

            # extract number of atoms for test configs
            # need to know which configs are testing, and which are training, so we extract correct natoms

            test_configs = []
            indx=0
            count=0
            for element in row_type.tolist():
                if (element=="Energy" and test_row_indices[indx]):
                    count+=1
                    test_configs.append(True)
                if (element=="Energy" and not test_row_indices[indx]):
                    test_configs.append(False)
                indx+=1
            
            assert(len(test_configs) == np.shape(natoms_per_config)[0]) 
            assert((sum(test_row_indices) == count) and (sum(test_configs) == count) )

            natoms_test = natoms_per_config[test_configs]

            truths = test_truths/natoms_test
            preds = test_preds/natoms_test

            #print(truths)
            # use group row indices to get
            # print(groups.tolist()[group_row_indices])

            # get all groups associated with fitting set

            filtered_groups = list(compress(groups.tolist(), test_row_indices))
            #print(len(filtered_groups))
            assert (len(filtered_groups) == len(truths))
            #print(filtered_groups[0])
            unique_groups_set = set()
            for x in filtered_groups:
                unique_groups_set.add(x)
            unique_groups = sorted(list(unique_groups_set)) #set(filtered_groups))
            #print(unique_groups)
            #print(unique_groups.sort(key=len))
            cmap = self.get_cmap(n=len(unique_groups), name='gist_rainbow')
            cmap = [cmap(i) for i in range(0,len(unique_groups))]
            # assign color to each value based on group name
            colors = []
            for g in filtered_groups:
                indx_bool = [g==unique_groups[i] for i in range(0,len(unique_groups))]
                indx = [i for i, x in enumerate(indx_bool) if x][0]
                colors.append(cmap[indx])
            #print(colors)
            #print(truths[0], preds[0])
            for i in range(0,len(truths)):
                plt.plot(truths[i], preds[i], c=colors[i], marker='o',markersize=8, alpha=0.5)

            #plt.xlabel("Target energy (eV/atom)")
            #plt.ylabel("Model energy (eV/atom)")

            #min_val = np.min(np.abs(truths))
            #max_val = np.max(np.abs(truths))
            #lims = [min_test_force, max_test_force]
            #plt.axline((0, 0.5), slope=1.0, color="black")

            if (mode=="Distribution"):
                abs_diff = np.abs(truths-preds)
                abs_truth = np.abs(truths)
                plt.scatter(abs_truth, abs_diff, c=colors, marker='o', alpha=0.5)
                plt.xscale("log")
                plt.yscale("log")
                plt.gca().set_xlim(left=1e-1)
                #plt.gca().set_ylim(bottom=1e-2)
                plt.xlabel(r"Target energy magnitude (eV/atom)")
                plt.ylabel(r"Absolute error (eV/atom)")
            elif(mode=="Linear"):
                plt.scatter(truths, preds, c=colors, marker='o',alpha=0.5)
                min_val = np.min(truths)
                max_val = np.max(truths)
                lims = [min_val, max_val]
                plt.plot(lims, lims, 'k-')
                plt.xlabel(r"Target energy (eV/atom)")
                plt.ylabel(r"Model energy (eV/atom)")

            if (legend):
                legend_handle = [mpatches.Patch(color=cmap[i], 
                                 label=unique_groups[i]) for i in range(0,len(unique_groups))]
                plt.legend(fancybox=True, framealpha=0.5, handles = legend_handle, bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.savefig("energy_agreement.png", bbox_inches="tight", dpi=500)

        if quantity=="Force":

            force_row_indices = row_type[:] == "Force"
            force_row_indices = force_row_indices.tolist()

            testing_bool = self.df.loc[:, "Testing"].tolist()

            # get indices of desired group, otherwise use same indices as test bool

            if (group_set is not None):
                group_row_indices = groups[:] == group_set
                group_row_indices = group_row_indices.tolist()
                if not np.any(group_row_indices):
                    raise ValueError(f"{group_set} set is not in this dataframe.")
            else:
                group_row_indices = testing_bool

            if (test_calc_bool): # calculate errors on test set
                testing_bool = testing_bool
            else: # calculate errors on training set
                testing_bool = [not item for item in testing_bool]
            if not np.any(testing_bool):
                raise ValueError(f"{fitting_set} set is empty in this dataframe.")

            # use list comprehension to extract row indices that are both force and testing
            #testing_force_row_indices = [force_row_indices and testing_bool for force_row_indices, testing_bool in zip(force_row_indices, testing_bool)]
            if (group_set is not None):
                testing_force_row_indices = [force_row_indices and testing_bool and group_row_indices \
                                            for force_row_indices, testing_bool, group_row_indices \
                                            in zip(force_row_indices, testing_bool, group_row_indices)]
            else:
                testing_force_row_indices = [force_row_indices and testing_bool for force_row_indices, testing_bool in zip(force_row_indices, testing_bool)]

            truths = np.array(truths[testing_force_row_indices])

            preds = np.array(preds[testing_force_row_indices])

            # get all groups associated with fitting set

            filtered_groups = list(compress(groups.tolist(), testing_force_row_indices))
            #print(len(filtered_groups))
            assert (len(filtered_groups) == len(truths))
            print(filtered_groups[0])
            unique_groups_set = set()
            for x in filtered_groups:
                unique_groups_set.add(x)
            unique_groups = sorted(list(unique_groups_set)) #set(filtered_groups))
            #print(unique_groups)
            #print(unique_groups.sort(key=len))
            cmap = self.get_cmap(n=len(unique_groups), name='gist_rainbow')
            cmap = [cmap(i) for i in range(0,len(unique_groups))]
            # assign color to each value based on group name
            colors = []
            for g in filtered_groups:
                indx_bool = [g==unique_groups[i] for i in range(0,len(unique_groups))]
                indx = [i for i, x in enumerate(indx_bool) if x][0]
                colors.append(cmap[indx])
            #print(colors)
            #print(truths[0], preds[0])

            if (mode=="Distribution"):
                abs_diff = np.abs(truths-preds)
                abs_truth = np.abs(truths)
                plt.scatter(abs_truth, abs_diff, c=colors, marker='o', alpha=0.5)
                plt.xscale("log")
                plt.yscale("log")
                plt.gca().set_xlim(left=1e-8)
                plt.gca().set_ylim(bottom=1e-8)
                plt.xlabel(r"Target force magnitude (eV/$\AA$)")
                plt.ylabel(r"Absolute error (eV/$\AA$)")
            elif(mode=="Linear"):
                plt.scatter(truths, preds, c=colors, marker='o',alpha=0.5)
                min_val = np.min(truths)
                max_val = np.max(truths)
                lims = [min_val, max_val]
                plt.plot(lims, lims, 'k-')
                plt.xlabel(r"Target force (eV/$\AA$)")
                plt.ylabel(r"Model force (eV/$\AA$)")

            if (legend):
                legend_handle = [mpatches.Patch(color=cmap[i], 
                                 label=unique_groups[i]) for i in range(0,len(unique_groups))]
                plt.legend(fancybox=True, framealpha=0.5, handles = legend_handle, bbox_to_anchor=(1.04, 1), loc="upper left")
                #plt.tight_layout(rect=[0, 0, 1, 0.8])
            plt.savefig("force_agreement.png", bbox_inches="tight", dpi=500)