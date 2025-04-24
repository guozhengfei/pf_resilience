
# data are stored in yearly RDS files; retrieve file names
wd_1 <- "E:/Project 2 pf resilience/1_Input/Fire/hotspotFRPSum_20240229_TERRA"; setwd(wd_1)
frpsum_list <- list.files(pattern = "rds")

for(i in unique(frpsum_list)) {
  print(i)
  
  # load the summarised data for a given year
  dat <- readRDS(i)[[1]] 
  
  write.csv(dat, file = paste(strsplit(i, split = ".rds"),'.csv'), row.names = FALSE)
}