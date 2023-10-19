

% path to data file
data_dir = "C:\Users\matti\OneDrive\Education\SDC\Master Thesis\master-project\data\";
filename = "subject_07_exp_phase_2_preprocessed";
fullpath = data_dir + filename;

% load data
data = readtable(fullpath, VariableNamingRule="preserve");

