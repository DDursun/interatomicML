import os, random, time, resource
from ase.geometry import get_distances
from copy import deepcopy
import numpy as np
from mpi4py import MPI
from scipy.linalg import lstsq
from fitsnap3lib.fitsnap import FitSnap
from fitsnap3lib.scrapers.ase_funcs import get_apre
import pandas
from sklearn.linear_model import Ridge
from settings import ACE_settings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

np.seterr(all="ignore")  # Optional: Suppresses floating-point warnings
np.float = np.float32  # Set default float type to float32


# os.chdir("lanl/W/")

def ase_scraper(snap, frames, energies, forces, stresses):
    """
    Function to organize groups and allocate shared arrays used in Calculator. For now when using 
    ASE frames, we don't have groups.

    Args:
        s: fitsnap instance.
        data: List of ASE frames or dictionary group table containing frames.

    Returns a list of data dictionaries suitable for fitsnap descriptor calculator.
    If running in parallel, this list will be distributed over procs, so that each proc will have a 
    portion of the list.
    """

    snap.data = [collate_data(snap, indx, len(frames), a, e, f, s) for indx, (a,e,f,s) in enumerate(zip(frames, energies, forces, stresses))]
    # Simply collate data from Atoms objects if we have a list of Atoms objecst.
    # if type(frames) == list:
        # s.data = [collate_data(atoms) for atoms in data]
    # If we have a dictionary, assume we are dealing with groups.
    # elif type(data) == dict:
    #     assign_validation(data)
    #     snap.data = []
    #     for name in data:
    #         frames = data[name]["frames"]
    #         # Extend the fitsnap data list with this group.
    #         snap.data.extend([collate_data(atoms, name, data[name]) for atoms in frames])
    # else:
    #     raise Exception("Argument must be list or dictionary for ASE scraper.")

def collate_data(s, indx, size, atoms, energy, forces, stresses):
    """
    Function to organize fitting data for FitSNAP from ASE atoms objects.

    Args: 
        atoms: ASE atoms object for a single configuration of atoms.
        name: Optional name of this configuration.
        group_dict: Optional dictionary containing group information.

    Returns a data dictionary for a single configuration.
    """

    # Transform ASE cell to be appropriate for LAMMPS.
    apre = get_apre(cell=atoms.cell)
    R = np.dot(np.linalg.inv(atoms.cell), apre)
    positions = np.matmul(atoms.get_positions(), R)
    cell = apre.T

    # Make a data dictionary for this config.

    data = {}
    data['PositionsStyle'] = 'angstrom'
    data['AtomTypeStyle'] = 'chemicalsymbol'
    data['StressStyle'] = 'bar'
    data['LatticeStyle'] = 'angstrom'
    data['EnergyStyle'] = 'electronvolt'
    data['ForcesStyle'] = 'electronvoltperangstrom'
    data['Group'] = 'All'
    data['File'] = None
    data['Stress'] = stresses
    data['Positions'] = positions
    data['Energy'] = energy
    data['AtomTypes'] = atoms.get_chemical_symbols()
    data['NumAtoms'] = len(atoms)
    data['Forces'] = forces
    data['QMLattice'] = cell
    data['test_bool'] = indx>=s.config.sections["GROUPS"].group_table["All"]["training_size"]*size
    data['Lattice'] = cell
    data['Rotation'] = np.array([[1,0,0],[0,1,0],[0,0,1]])
    data['Translation'] = np.zeros((len(atoms), 3))
    data['eweight'] = s.config.sections["GROUPS"].group_table["All"]["eweight"]
    data['fweight'] = s.config.sections["GROUPS"].group_table["All"]["fweight"]
    data['vweight'] = s.config.sections["GROUPS"].group_table["All"]["vweight"]

    return data

def load_files(file_name_structures, file_name_energies):
    df_structures = pandas.read_hdf(file_name_structures)
    print("df_structures: ", (len(df_structures)))
    df_structures.sort_index(inplace=True)
    df_structures = df_structures.iloc[:len(df_structures)//10]
    #print(df_structures.head(10))

    df_energies = pandas.read_hdf(file_name_energies)
    print("df_energies: ", (len(df_energies)))
    df_energies.sort_values(by=["index"], inplace=True)
    df_energies = df_energies.iloc[:len(df_energies)//10]
    #print(df_energies.head(10))

    df_structures = df_structures[df_structures.index.isin(df_energies["index"].values)]
    
    return df_structures, df_energies

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

fs_instance1 = FitSnap(ACE_settings, comm=comm, arglist=["--overwrite"])
file_name_structures = "Be_large_subset_structures.h5"
file_name_energies = "Be_high_3.hdf"
atoms_clmn = "ASEatoms_rescale"

start_time_load = time.time()
df_structures, df_energies = load_files(file_name_structures, file_name_energies)
print(len(df_structures), len(df_energies))
print("Files are loadedd in ", time.time()-start_time_load, "sec")

configs_num = df_structures[atoms_clmn].shape[0]
ratio = configs_num//size
rem = configs_num%size
a1 = rank*ratio + min(rank,rem)
a2 = (rank+1)*ratio + min(rank,rem-1) + 1

print("Scraping process starts!")
start_time_scrape = time.time()
ase_scraper(fs_instance1, df_structures[atoms_clmn].values[a1:a2], df_energies['energy'].values[a1:a2], df_energies["forces"].values[a1:a2], df_energies["stress"].values[a1:a2])
# Now `fs_instance1.data` is a list of dictionaries containing configuration/structural info.
print("Scraping finished in", time.time()-start_time_scrape, "sec")


print("Process Config starts!")
start_time_scrape = time.time()
fs_instance1.process_configs(allgather=True)
print("Process config finished in", time.time()-start_time_scrape, "sec")

print("Solving process starts!")
fs_instance1.solver.create_datasets()

del df_energies
print("Dataset created, df_energies deleted!")
# del df_structures
#####################################################################################################################################################
##### look if ACE already has one useless descriptor or not. if not then you will have to add to have descriptors untouched during outer product#####
#####################################################################################################################################################
num_desc_ACE = fs_instance1.solver.configs[0].descriptors.shape[1]+1
print("Number of descriptors:", num_desc_ACE)
num_desc = num_desc_ACE
aw = []
bw = []
energy_selector = np.array([])
force_selector = np.array([])

# Loop in each configuration
for config_ACE in fs_instance1.solver.configs[a1:a2]:
    # Creating design matrix
    B = np.concatenate((np.ones((config_ACE.natoms,1)),config_ACE.descriptors),axis=1)

    # Loading gradients from FitSNAP
    dB = np.zeros((config_ACE.natoms,3,config_ACE.natoms,num_desc_ACE))
    dB[config_ACE.dgrad_indices[:,1].astype(int), config_ACE.dgrad_indices[:,2].astype(int),
        config_ACE.dgrad_indices[:,0].astype(int), 1:num_desc_ACE] = config_ACE.dgrad
    dB = np.reshape(dB, (config_ACE.natoms*3,config_ACE.natoms,num_desc_ACE))


    aw_temp = np.zeros((1+3*config_ACE.natoms, round(num_desc*(num_desc+1)/2)))
    energy_selector = np.concatenate((energy_selector,[1],[0 for i in range(3*config_ACE.natoms)]),axis=0)
    force_selector = np.concatenate((force_selector,[0],[1 for i in range(3*config_ACE.natoms)]),axis=0)

    # Calculating the new descriptor pairs 
    for i in range(num_desc):
        for j in range(i,num_desc):
            aw_temp[0,num_desc*i-round((i-1)*(i+2)/2)+j-1] = np.sum(B[:,i]*B[:,j])*150/config_ACE.natoms
            aw_temp[1:,num_desc*i-round((i-1)*(i+2)/2)+j-1] = np.dot(dB[:,:,i],B[:,j])+np.dot(dB[:,:,j],B[:,i])
    aw.append(aw_temp[:,1:])

    # bw = np.concatenate((bw,[150*config_ACE.energy],config_ACE.forces))  Replaced:
    bw_list = [150 * config_ACE.energy] + list(config_ACE.forces)
    bw.extend(bw_list)


    del dB
    del B
    del aw_temp

bw = np.array(bw)
print("Rank: ", rank)
np.savez('/home/dursun/FitSNAP/qACE/output/output'+str(rank)+'.npz',*aw)
del aw
# big_aw = comm.gather(aw, root=0)
big_bw = comm.gather(bw, root=0)
del bw
big_es = comm.gather(energy_selector, root=0)
del energy_selector
big_fs = comm.gather(force_selector, root=0)
del force_selector

del fs_instance1

if rank == 0:
    # big_aw = [item for sublist in big_aw for item in sublist]
    # aw=np.vstack(big_aw)
    bw=np.concatenate(big_bw)
    del big_bw
    energy_selector=np.concatenate(big_es)
    del big_es
    force_selector=np.concatenate(big_fs)
    del big_fs
    # np.save('bw.npy',bw)
    # np.save('energy_selector.npy',energy_selector)
    # np.save('force_selector.npy',force_selector)

    last_index = 0
    configs_index = []
    for i in range(configs_num):
        configs_index.append([last_index+j for j in range(1+3*len(df_structures[atoms_clmn].values[i]))])
        last_index += 1+3*len(df_structures[atoms_clmn].values[i])
    ind_configs_index = [i for i in range(configs_num)]
    random.seed(58)
    random.shuffle(ind_configs_index)
    hlfpnt = len(ind_configs_index)//2
    hlfpnt1 = [item for sublist_index in ind_configs_index[:hlfpnt] for item in configs_index[sublist_index]]
    hlfpnt2 = [item for sublist_index in ind_configs_index[hlfpnt:] for item in configs_index[sublist_index]]

    aw = np.zeros((len(hlfpnt1)+len(hlfpnt2),round(num_desc*(num_desc+1)/2)-1))
    aw_ind = 0
    for i in range(size):
        loaded_aw = np.load('/home/dursun/FitSNAP/qACE/output/output'+str(i)+'.npz')
        for name in loaded_aw.files:
            aw[aw_ind:(aw_ind+loaded_aw[name].shape[0]),:] = loaded_aw[name]
            aw_ind += loaded_aw[name].shape[0]
        os.remove('/home/dursun/FitSNAP/qACE/output/output'+str(i)+'.npz')
        del loaded_aw
    print("Gathered all data")
    print(configs_index[-1][-1] == aw.shape[0]-1)
    # aw = np.load("../../../npy_files/aw.npy")
    # bw = np.load("../../../npy_files/bw_high.npy")
    # energy_selector = np.load("../../../npy_files/energy_selector.npy")
    # force_selector = np.load("../../../npy_files/force_selector.npy")

    print("Starting training")
    print("Memory usage 1 is",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)

    alpha = 0.03
    start_time = time.time()
    coefficients, *_ = lstsq(np.vstack([np.sqrt(alpha)*np.eye(aw.shape[1]),aw[hlfpnt1]]), np.concatenate([np.zeros(aw.shape[1]),bw[hlfpnt1]]), 1.0e-13)
    print("Fitting finished in", time.time()-start_time, "sec")
    print("Memory usage 2 is",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024)
    train_residual = np.square(np.dot(aw[hlfpnt1],coefficients) - bw[hlfpnt1])
    print("Energy training RMSE is", np.sqrt(np.sum(train_residual*energy_selector[hlfpnt1]/22500)/energy_selector[hlfpnt1].sum()))
    print("Force training RMSE is", np.sqrt(np.sum(train_residual*force_selector[hlfpnt1])/force_selector[hlfpnt1].sum()))
    test_residual = np.square(np.dot(aw[hlfpnt2],coefficients) - bw[hlfpnt2])
    print("Energy testing RMSE is", np.sqrt(np.sum(test_residual*energy_selector[hlfpnt2]/22500)/energy_selector[hlfpnt2].sum()))
    print("Force testing RMSE is", np.sqrt(np.sum(test_residual*force_selector[hlfpnt2])/force_selector[hlfpnt2].sum()))
    entire_test_diff = np.dot(aw,coefficients) - bw
    entire_test_residual = np.square(entire_test_diff)
    print("Energy RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*energy_selector/22500)/energy_selector.sum()))
    print("Force RMSE if tested on the entire dataset is", np.sqrt(np.sum(entire_test_residual*force_selector)/force_selector.sum()))
    np.save('coefficients.npy',coefficients)

    detailed_results = np.zeros((len(hlfpnt1)+len(hlfpnt2),4))
    for j in ind_configs_index[hlfpnt:]:
        atoms = df_structures[atoms_clmn].values[j]
        distance_matrix = get_distances(atoms.positions,cell=atoms.cell,pbc=atoms.pbc)[1]
        detailed_results[configs_index[j][1:],0] = j
        detailed_results[configs_index[j][1:],1] = bw[configs_index[j][1:]] #forces
        detailed_results[configs_index[j][1:],2] = entire_test_diff[configs_index[j][1:]] #error
        detailed_results[configs_index[j][1:],3] = np.repeat(np.nanmin(np.where(distance_matrix==0,np.nan,distance_matrix),axis=1),3) #nearest_neighbor
    np.save('detailed_results_forces.npy',detailed_results)
    del detailed_results

    detailed_results = np.zeros((len(ind_configs_index[hlfpnt:]),4+aw.shape[1]))
    for i,j in enumerate(ind_configs_index[hlfpnt:]):
        atoms = df_structures[atoms_clmn].values[j]
        distance_matrix = get_distances(atoms.positions,cell=atoms.cell,pbc=atoms.pbc)[1]
        detailed_results[i,0] = j
        detailed_results[i,1] = bw[configs_index[j][0]]/150 #energy
        detailed_results[i,2] = entire_test_diff[configs_index[j][0]]/150 #error
        detailed_results[i,3] = np.min(distance_matrix[distance_matrix!=0]) #min_bond_len
        detailed_results[i,4:] = aw[configs_index[j][0]]*coefficients #contributions
    np.save('detailed_results_energies.npy',detailed_results)
