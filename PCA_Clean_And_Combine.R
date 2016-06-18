library(data.table)
library(gtools)
library(parallel)
library(compiler)
library(zoo)
library(plyr)

PCA_PctVarianceExplainedCutoff <- .90
Dev_Size <- .2
Train_Size <- .8
Pct_Sample <- .1 #the percent of the total data to randomly sample for the training and dev data sets. 



# Path to folder containing files
setwd("C:/Users/marks/Google Drive/EEG Kaggle/")
folder.path <- "C:/Users/marks/Google Drive/EEG Kaggle/train" #training data path. 



#find all the Data files and event files in folder.path and store them in a list
Train_Data_Files <- list.files(path=folder.path, pattern="*data.csv", full.names=T, recursive=FALSE)
Train_Events_Files <- list.files(path=folder.path, pattern="*events.csv", full.names=T, recursive=FALSE)

num_Files <- length(Train_Data_Files)


PCA <- function(df,PCA_PercentVarExplainedCutoff){
  #Compute covariance matrix
  cov <- cov(df)
  
  #Manually compute the eigenvectors and eigenvalues
  eig <- eigen(cov,symmetric=T)
  
  #Extract eigenvectors and eigenvalues
  EigenValues <- eig$values
  EigenVectors <- eig$vectors
  
  #Determine number of PC's to keep based on PCA_PercentVarExplainedCutoff
  EigenSum <- sum(eig$values)
  PC_PercentVarExplained <- t(matrix(abs(EigenValues/EigenSum)))
  
  
  ##This loop will continue to add principal components until the percent variance
  ##explained is greater than PCA_PercentVarExplainedCutoff. Num PCs starts at 1
  numPCs <- 1  
  repeat{
    if(sum(PC_PercentVarExplained[1:numPCs])>=PCA_PercentVarExplainedCutoff){
      break
    }
    numPCs <- numPCs + 1
  }
  ##Do the Projections
#   PCA_Projs <- as.matrix(df) %*% EigenVectors[,1:numPCs]
  PCA_Projs <- as.matrix(df) %*% EigenVectors[,1:5] #going to fudge this for now and simply do 5 PCs. Was getting uneven amount of PCs for each series with other method.
  round(PCA_Projs,1) #round to limit data size.
}
PCA <- cmpfun(PCA) #compile function for speed.




#Create function to call in parallel to clean and combine all the data into a single file 
Combine_Files_To_List <- function(i,Files, Event_Files,Pct_Sample,PCA_PctVarianceExplainedCutoff){
  Current_Data_File_String <- paste("read.csv(\"", Files[i],"\",sep=\",\", stringsAsFactors = FALSE)",sep='')
  Current_Data <- data.frame(eval(parse(text = Current_Data_File_String)))
  
  Subject_Strings <- c(Current_Data[1])
  Current_Data <- Current_Data[-1]
  Subject_Strings <- lapply(Subject_Strings, as.character)
  #extract the subject ID and series ID from the strings. Save them as two columns of integers instead. This will save memory.
  Subject_ID_Strings <- sapply(Subject_Strings, function(x) lapply(strsplit(x,"_"), `[[`, 1))
  Subject_ID <- as.numeric(gsub("\\D", "", Subject_ID_Strings))   
  Series_ID_Strings <- sapply(Subject_Strings, function(x)lapply(strsplit(x,"_"), `[[`, 2))  
  Series_ID <- as.numeric(gsub("\\D", "", Series_ID_Strings)) 
  
  #remove them from memory to save RAM
  rm(Subject_Strings)
  rm(Subject_ID_Strings)
  rm(Series_ID_Strings)
  
  num_observations <- length(Series_ID)
  
  Principal_Components <- PCA(Current_Data,PCA_PctVarianceExplainedCutoff)
  Principal_Components <- as.data.frame(Principal_Components)
  
  #set the column names to PC1 thru PCn where n equals the resultant number of principal components. 
  PC_Colname_String <- ''
  for(j in 1:ncol(Principal_Components)){
    colname <- paste("PC", j,sep='')
    
    PC_Colname_String <- c(PC_Colname_String,colname)
  }
  colnames(Principal_Components) <- PC_Colname_String[-1]
  
  #remove them from memory to save RAM
  rm(Current_Data)
  #make features based on those defined in the functions above
  #Features_DF <- Create_Features(Principal_Components)  
  
  Current_Data <- cbind(Subject_ID,Series_ID, c(seq(1:num_observations)),Principal_Components)

  colnames(Current_Data)[3] <- 'Observation_Num'
  
  
  Current_Event_File_String <- paste("read.csv(\"", Event_Files[i],"\",sep=\",\")",sep='')
  Current_Event <- data.frame(eval(parse(text = Current_Event_File_String)))
  Current_Event <- Current_Event[-1]
  
  #Combine all the events into one string
  Y <- paste(Current_Event[,1],",",Current_Event[,2],",",Current_Event[,3],",",Current_Event[,4],",",Current_Event[,5],",",Current_Event[,6],sep='') 
  
  Processed_Data <- cbind(Current_Data,Y) #combine all the new data, features, and events together into one data.frame
  
  #return only a sample of the data based on the variable Pct_Sample to limit the total data size. set Pct_Sample = 1 to return all data.
  Sample_size <-num_observations*Pct_Sample
  
  Sampled_Processed_Data <- Processed_Data[sample(num_observations,Sample_size),]
  Sampled_Processed_Data
}
Combine_Files_To_List <- cmpfun(Combine_Files_To_List) #compile function for speed.




##This whole section sets up a parallel process to clean and combine the data. It will use all but one of the available cores on the machine. 

#Time processing time. Start the clock!
ptm <- proc.time()

#Initialize numcores-2 clusters for parallel processing
cl1 <- suppressWarnings(makeCluster(detectCores()-1, type="PSOCK"))

#Export Functions to clusters
clusterExport(cl1,list("PCA"))

#determine the number of dev and training files. 
Num_Dev <- round(Dev_Size*num_Files,0)
Num_Train <- round(Train_Size*num_Files,0)

#return the indices of the dev and training data files into a list. 
i_list_Dev <- seq(1:Num_Dev)
i_list_Train <- seq(Num_Dev+1,num_Files)


##Run a parallel lapply function over the clusters 
Dev_Data_List <-  parLapply(cl1,i_list_Dev,Combine_Files_To_List,Train_Data_Files,Train_Events_Files,Pct_Sample,PCA_PctVarianceExplainedCutoff) #returns list of dataframes
Dev_Data <- rbind.fill(Dev_Data_List) #combine list of dataframes into single dataframe
rm(Dev_Data_List)#remove list of dataframes from memory

Train_Data_List <-  parLapply(cl1,i_list_Train,Combine_Files_To_List,Train_Data_Files,Train_Events_Files,Pct_Sample,PCA_PctVarianceExplainedCutoff) #returns list of dataframes
Train_Data <- rbind.fill(Train_Data_List) #combine list of dataframes into single dataframe
rm(Train_Data_List) #remove list of dataframes from memory

stopCluster(cl1)

str(Dev_Data)
write.table(Dev_Data,"Dev_Data.txt",sep ='\t', row.names = F)
rm(Dev_Data)

write.table(Train_Data,"Train_Data.txt",sep ='\t', row.names = F)
rm(Train_Data)

# Stop the clock
proc.time() - ptm