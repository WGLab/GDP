source("SurvSimu.R")
filename="input_simple_example.csv"
getSimulated(filename,group_size=3,group_num=2,lambda=0.0000008,rho=2.8,effective_group=5,sample_size=10,within_group_cor=0.3)
