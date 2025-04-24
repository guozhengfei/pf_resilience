
# Functions to help the analysis of energetically extreme wildfire events
# Calum X. Cunningham, University of Tasmania
# 22 Jan 2024

# set ggplot theme for session
theme_set(theme_minimal() +
            theme(panel.grid.minor = element_blank(), 
                  panel.background = element_blank(),
                  plot.title = element_text(hjust = 0.5),
                  axis.ticks.length = unit(-0.15, "cm"),
                  axis.ticks.x = element_line(linewidth = 0.3)))

# function to select only the top n events from each year of various metrics (to keep file sizes manageable)
read_summarised_frp <- function(fileList, n = 10000){
  
  # initialise objects to store outputs in
  largest_events_dn <- NULL
  largest_events <- NULL
  n_events <- NULL
  frp_sum_vec <- NULL
  n_hotspots_total <- NULL
  
  for(i in unique(fileList)) {
    print(i)
    
    # load the summarised data for a given year
    dat <- readRDS(i)[[1]] 
    
    # how many events?
    n_events <- c(n_events, nrow(dat))
    
    # how many hotspots used to create those events
    n_hotspots_total <- c(n_hotspots_total, sum(dat$n))
    
    # select the largest n events for day and night separately
    dat1 <- dat %>% 
      group_by(dn) %>% 
      arrange(desc(FRP_sum)) %>%
      slice_head(n = n)
    largest_events_dn <- bind_rows(largest_events_dn, dat1)
    
    # add together day and night FRP for each day, and then select the largest n events 
    dat2 <- dat %>% 
      group_by(ID, date_) %>%
      summarise(FRP_sum = sum(FRP_sum), n_hotspots = sum(n)) 
    
    # save vector of all FRP_sum events so that I can calculate percentiles later (without exploding file sizes)
    frp_sum_vec <- c(frp_sum_vec, dat2$FRP_sum)
    
    # select only the top n events to save memory (only the very largest events are relevant for the analysis)
    dat2 <- dat2 %>%
      ungroup() %>%
      arrange(desc(FRP_sum)) %>%
      slice_head(n = n)
    largest_events <- bind_rows(largest_events, dat2)
  }
  
  return(list(largest_events_dn = largest_events_dn, largest_events = largest_events, n_events = n_events, n_hotspots_total = n_hotspots_total, frp_sum_vec = frp_sum_vec))
}

# function to read a summary of duplicated hotspots
read_duplicate_summary <- function(fileList) {
  summaryList <- list()
  for(i in unique(fileList)) {
    print(i)
    summaryList[[i]] <- readRDS(i)[[2]] 
  }
  return(bind_rows(summaryList))
}

# function to read a duplicated hotspots for subsequent mapping
map_duplicate_locations <- function(fileList) {
  summaryList <- list()
  for(i in unique(fileList)) {
    print(i)
    summaryList[[i]] <- readRDS(i)$duplicates
  }
  return(bind_rows(summaryList))
}

# predict trend line and 95% CI from model
predict_fun <- function(mod, dat, logBackTrans = T) {
  
  dat$fit <- predict(mod, dat) 
  dat$se <- predict(mod, dat, se.fit = T)$se.fit
  
  if(logBackTrans) {
    dat$fit.exp <- exp(dat$fit)
    dat$lcl <- exp(dat$fit - 1.96*dat$se)
    dat$ucl <- exp(dat$fit + 1.96*dat$se)
  }
  
  else {
    dat$lcl <- dat$fit - 1.96*dat$se
    dat$ucl <- dat$fit + 1.96*dat$se
  }
  
  return(dat)
}