source("SurvSimu.R")
filename="input.csv"
getSimulated(filename,group_size=5,group_num=30,lambda=0.0000008,rho=2.8,effective_group=5,sample_size=500,within_group_cor=0.3)
