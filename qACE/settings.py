ACE_settings = \
{
"ACE":
    {
    "numTypes": 1,
    "ranks": "1 2",
    "lmin": "0 0",
    "lmax": "1 2",
    "nmax": "8 3",
    "rcutfac": 6.0,
    "lambda": 3.059235105,
    #"rfac0": 0.99363,
    #"rmin0": 0.0,
    #"wj": 1.0,
    #"radelem": 0.5,
    "type": "Be",
    #"wselfallflag": 0,
    #"chemflag": 0,
    "bzeroflag": 0,
    "bikflag" : 1,
    "dgradflag" : 1
    },
"CALCULATOR":
    {
    "calculator": "LAMMPSPACE",
    "energy": 1,
    "per_atom_energy": 1,
    "force": 1,
    "stress" : 0,
    "nonlinear" : 1
    },
"ESHIFT":
    {
    "Be" : 0.0
    },
"PYTORCH":
    {
    "layer_sizes" : "num_desc 32 32 1",
    "learning_rate" : 5e-5,
    "num_epochs" : 1,
    "batch_size" : 1, # 363 configs in entire set
    "save_state_output" : "W_Pytorch.pt",
    "energy_weight" : 150.0,
    "force_weight" : 1.0,
    # "training_fraction" : 0.5,
    },
"GROUPS":
    {
    # name size eweight fweight vweight
    "group_sections" : "name training_size testing_size eweight fweight vweight",
    "group_types" : "str float float float float float",
    "smartweights" : 0,
    "random_sampling" : 0,
    "All" :  "0.5    0.5    150.0      1.0  0.0"
    },
"OUTFILE":
    {
    "output_style" : "PACE",
    "metrics" : "Be_metrics.md",
    "potential" : "Be_pot"
    },
"REFERENCE":
    {
    "units": "metal",
    "pair_style": "zero 10.0",
    "pair_coeff": "* *",
    },
"SOLVER":
    {
    "solver": "PYTORCH",
    "compute_testerrs": 1,
    "detailed_errors": 1
    },
"EXTRAS":
    {
    "dump_descriptors": 0,
    "dump_truth": 0,
    "dump_weights": 0,
    "dump_dataframe": 0
    },
"MEMORY":
    {
    "override": 0
    }
}