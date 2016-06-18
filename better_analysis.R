library(nnet)
library(mlogit)
library(ROCR)

options(scipen = 100)

### Make train data ###
# Path to folder containing files
folder.path <- "C:\\Users\\carson.GROUP5\\Dropbox\\MIDS\\Machine Learning\\Final\\KaggleTest\\train\\"

# Make data file
path.first.file <- "C:\\Users\\carson.GROUP5\\Dropbox\\MIDS\\Machine Learning\\Final\\KaggleTest\\train\\subj1_series1_data.csv"

# Load first file to get structure for appending
master.train.data <- read.csv(path.first.file, header = T, stringsAsFactors = F)

# Only keep first row
master.train.data <- master.train.data[1,]

# Loop through each series and append to master

for(series in 1:8) {
  path.name <- paste0(folder.path,"subj",1,"_","series",series,"_","data",".csv")
  print(path.name)
  new.file <- read.csv(path.name, header = T, stringsAsFactors = F)
  master.train.data <- rbind(master.train.data, new.file)
}

# Remove duplicate first row
train.data <- master.train.data[-1,]

# Make events file
path.first.file.events <- "C:\\Users\\carson.GROUP5\\Dropbox\\MIDS\\Machine Learning\\Final\\KaggleTest\\train\\subj1_series1_events.csv"

master.train.events <- read.csv(path.first.file.events, header = T, stringsAsFactors = F)
master.train.events <- master.train.events[1,]

for(series in 1:8) {
  path.name <- paste0(folder.path,"subj","1","_","series",series,"_","events",".csv")
  print(path.name)
  new.file <- read.csv(path.name, header = T, stringsAsFactors = F)
  master.train.events <- rbind(master.train.events, new.file)
}

train.events <- master.train.events[-1,]

### Make Test Data ###
# Path to folder containing files
folder.path <- "C:\\Users\\carson.GROUP5\\Dropbox\\MIDS\\Machine Learning\\Final\\KaggleTest\\test\\"

# Make data file
path.first.file <- "C:\\Users\\carson.GROUP5\\Dropbox\\MIDS\\Machine Learning\\Final\\KaggleTest\\test\\subj1_series9_data.csv"

# Load first file to get structure for appending
master.test.data <- read.csv(path.first.file, header = T, stringsAsFactors = F)

# Only keep first row
master.test.data <- master.test.data[1,]

# Loop through each series and subject and append to master

for(subject in 1:12){
  for(series in 9:10) {
    path.name <- paste0(folder.path,"subj",subject,"_","series",series,"_","data",".csv")
    print(path.name)
    new.file <- read.csv(path.name, header = T, stringsAsFactors = F)
    master.test.data <- rbind(master.test.data, new.file)
  }
}

# Remove duplicate first row
test.data <- master.test.data[-1,]

### Features ###

### Modeling ###

# Create an empty data frame for appending output probabilities
probs <- data.frame(c(1:3144171))

# Build a logistic regression for each of the 6 outcome variables
for(i in 2:7) {
  model <- glm(train.events[,i] ~ ., data = train.data[,-1], family = "binomial")
  print(summary(model))
  predictions <- predict(model, test.data[,-1], type = "response")
  print(length(predictions))
  probs[,i] <- predictions
}

### Predictions and output ###

# Convert to data frame and add column headers
probs.df <- as.data.frame(probs)
colnames(probs.df) <- colnames(mini.train.events)

submission <- cbind(test.data[,1], probs.df[,-1])
colnames(submission) <- colnames(mini.train.events)

# Find and set threshhold
colMeans(train.events[,-1])

submission.binary <- sapply(submission[,-1], function(x) ifelse(x > .5, 1, 0))
submission.50.final <- cbind.data.frame(test.data$id, submission.binary)
colnames(submission.50.final) <- colnames(mini.train.events)

colMeans(submission.50.final[,-1])

submission.binary <- sapply(submission[,-1], function(x) ifelse(x > .40, 1, 0))
submission.40.final <- cbind.data.frame(test.data$id, submission.binary)
colnames(submission.40.final) <- colnames(mini.train.events)

colMeans(submission.40.final[,-1])

write.csv(submission.40.final, "FirstSubmission_Logistic", row.names = F)




