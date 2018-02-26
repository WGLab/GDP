library(MASS)

#group_size: size of each group
#group_num: number of groups
#get the covariance matrix
getsigma=function(group_size,within_group_cor){
	z_cov=matrix(nrow=group_size,ncol=group_size)
	for(i in c(1:group_size)){
		for(j in c(1:group_size)){
			if(i==j){
			z_cov[i,j]=1
			}else{
			z_cov[i,j]=within_group_cor
			}
		}
	}
	return(z_cov)
}

#get the simulated features 
getFeatures=function(group_size,group_num,sample_size,within_group_cor){
	for(g in c(1:group_num)){
		sig=getsigma(group_size,within_group_cor)
		z=mvrnorm(sample_size,mu=rep(0,group_size),Sigma=sig)
		if(g==1){
		x=z
		}else{
		x=cbind(x,z)
		}
	}
	return(x)
}


#main function to get the simulated data
#filename: the name of the output file
#group_size: the number of features within each group
#group_num: numbers of groups
#effective_group: the number of survival latent time relevant groups
#within_group_cor: correlations among the features within each group
#sample_size: the number of patients to be simulated
#lambda and rho: the coefficients for simulating latent survival time
#rateC: rate of expoential function for simulating censoring time
getSimulated=function(filename,group_size=20,group_num=200,effective_group=3,within_group_cor=0.3,sample_size=1000,lambda=0.0001,rho=4,rateC=0.005){
	features=getFeatures(group_size,group_num,sample_size,within_group_cor)
	group_levels=c()
	for(i in c(1:group_num)){
		group_levels=c(group_levels,rep(i,group_size))
	}


	#get random values from uniform distribution
	v=runif(sample_size,0,1)
	#get beta coefficients, where the survival
	beta=c()
	for(i in c(1:group_num)){
		if(i<=effective_group){
			b_sd=10/(effective_group*group_size)
			b_v=rnorm(group_size,mean=0,sd=b_sd)
		}else{
			b_v=rep(0,group_size)
		}
		beta=c(beta,b_v)
	}

	#get latent survival time
	print(dim(features))
	print(length(beta))
	tmp=features %*% beta
	print(dim(tmp))
	time_latent=(-log(v)/(lambda*exp(features %*% beta)))^(1/rho)
	#get censoring time
	time_censor=rexp(n=sample_size,rate=rateC)
	#get observed time
	days=pmin(time_latent,time_censor)
	#get censoring status,  1: censored, 0: un-censored
	censors=as.numeric(time_latent>time_censor)
	
	print("time latent:")
	print(summary(time_latent))

	print("time censored")
	print(summary(time_censor))

	print("censors")
	print(summary(censors))

	print("days")
	print(summary(days))
        write.table(t(group_levels),file=filename,sep=",",quote=F,row.names=F,col.names=F)
        data=data.frame(features,days=days,censors=censors)
        write.table(data,file=filename,sep=",",quot=F,row.names=F,append=T)
}
