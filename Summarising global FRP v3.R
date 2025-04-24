#rm(list = ls())

# Code for the analysis of energetically extreme wildfire events
# Calum X. Cunningham, University of Tasmania
# 7 March 2024

# The key sections of the analysis are as follows:

# 1. Create global lattice
# 2. Load and summarise hotspots each day (including identifying and omitting likely duplicates)
# 3. Read summarised data of FRP_sum
# 4. Load biome spatial data
# 5. Ratio of extreme events in continents, biogeographical realms, and biomes
# 6. Temporal trends among key biomes
# 7. Supplementary Analysis: Compare Terra with Terra + Aqua

# load packages
#devtools::install_github("hypertidy/L3bin")
pacman::p_load(L3bin, sf, units, dplyr,lubridate, ggplot2, cowplot, rnaturalearth, tidyverse, mgcv, ggh4x, ggsci, hashr, terra, viridis)

# source custom functions to help with organising data, predicting from models, graphing 
source("E:/Project_pf/3_Code/Summarising global FRP_functions v1.R")

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 1. Create global lattice ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

# create equal area, equal shape grid on which to conduct all analysis
# nr = NUMROWS, i.e. how many steps in latitude seq(-90, 90, length.out = nr) is the basis of the binning. E.g. nr of 360 will give cell sizes of approx 0.5 degrees (180 degrees divide by 360 rows = 0.5)
# thus, to get approx cell sizes of 0.1 degree, then 180/0.1 = 1800 (default)
create_grid_fun <- function(nr = 1800){
  
  # stores the information about the scheme
  bins <- L3bin(nr)
  
  # Get extent of each bin
  ex <- extent_from_bin(1:bins$totbins, nr)
  
  # function to convert extent table to to sf polygons
  to_sf <- function(ex, crs = st_crs(4326)) {
    idx <- c(1, 2, 2, 1, 1, 
             3, 3, 4, 4, 3)
    proto <- sf::st_polygon(list(cbind(0, c(1:4, 1))))
    sf::st_sfc(lapply(split(t(ex), rep(1:nrow(ex), each = 4L)), 
                      function(.x) {
                        proto[[1L]][] <- .x[idx]
                        proto
                      }), crs = crs)
  }
  
  ## Convert the list of extents to SF polygons
  sfx <- to_sf(ex)
  
  # Convert those polygons to a feature collection
  sfx_c <- st_sf(sfx)
  
  # Add an ID field
  sfx_c$ID <- 1:nrow(sfx_c)
  
  # Add an area field - areas very similar across cells (not exact)
  sfx_c$area <- st_area(sfx_c)
  
  return(sfx_c)
}

# create grid template
gridTemplate <- create_grid_fun(nr = 900)

# plot area of grid cells to verify they are very similar (not possible to be exactly the same shape as well as area)
gridTemplate %>% slice_sample(n = 5000) %>% ggplot(aes(area)) + geom_histogram()

# save grid
#saveRDS(gridTemplate, "C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/hotspotGridTemplate.rds")

# read grid
#gridTemplate <- readRDS("C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/hotspotGridTemplate.rds")
grid_centroid <- gridTemplate %>% st_centroid()

# report average cell size
mean(gridTemplate$area)/1000000


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 2. Load and summarise hotspots each day ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

wd <- "E:/Project_pf/1_Input/FIRMS_MODIS/"
setwd(wd)

# hotspots are saved as monthly text files
# get file names 
listOfFiles <- data.frame(fileName = list.files()) %>%
  mutate(year = substr(fileName, 9, 12),
         month = substr(fileName, 13, 14))

# function to identify likely duplicated hotspots (as a result of slightly overlapping satellite overpasses, especially poles)
remove_duplicates_fun <- function(file, conf_threshold = 30) {
  
  # Make a list to store the corrected output
  out <- list()
  idx <- 1
  
  # Process each satellite and day/night separately
  for(sat_x in c("T","A")){
    for(day_x in c("D","N")){
      
      # Filter satellite and day/night and confidence, and type == fire (0)
      filex <- filter(file, sat == sat_x, dn == day_x, conf >= conf_threshold, type == 0)
      
      # If eg. Aqua satellite is missing, skip to next
      if(nrow(filex)==0){next}
      
      # Convert hotspot subset to vector
      pts <- vect(filex,geom=c("lon","lat"),crs=crs("epsg:4326"),keepgeom=TRUE)
      
      # Project to equal area grid
      pts <- project(pts,crs("epsg:6933"))
      
      # Extract grid coordinates
      pts$x <- geom(pts)[,3]
      pts$y <- geom(pts)[,4]
      
      # Round coordinates to 500m
      pts$x <- floor(((pts$x+250)/500))*500
      pts$y <- floor(((pts$y+250)/500))*500
      
      # Generate hash from pair of coordinates as a unique index
      pts$hx <- hash(paste0(pts$x,pts$y))
      
      # Convert back to non-geographic data frame
      pts <- as.data.frame(pts)
      
      # Within groups of date and unique geo ID, add an index of duplicated rows
      ptsg <- pts %>% group_by(hx,YYYYMMDD) %>% mutate(gid = row_number())
      
      # Add to our output table
      out[[idx]]<-ptsg
      idx=idx+1
    }
  }
  
  # Merge all satellites/day-night into a single table and count how many duplicates, if any
  out_df <- bind_rows(out) %>% group_by(hx) %>% mutate(n_hs = length(unique(gid)))
  
  # report how many duplicates (values >= 2)
  print(paste(sum(out_df$gid != 1), "of", nrow(out_df), "are likely duplicates"))
  
  # randomly select one hotspot from each collection of duplicates
  set.seed(123)
  out_df_dup <- out_df %>% 
    filter(n_hs != 1) %>% 
    group_by(hx) %>% 
    slice(., sample(1:n())) %>% # randomise order (and therefore which observation will be selected)
    slice_head(n = 1) %>% # select one hotspot
    mutate(wasduplicated = "y")
  
  # join non-duplicates with randomly selected hotspots
  out_df1 <- out_df %>% 
    filter(n_hs == 1) %>%
    bind_rows(out_df_dup)
  
  # summarise frequency of duplicates
  summarisedFreq <- out_df1 %>% ungroup() %>% count(n_hs) 
  
  return(list(duplicates_removed = out_df1,
              summary_of_duplicates = summarisedFreq))
}

# function to loop through each year, summarise various metrics, and then save yearly summary file
# satellite = Aqua (A) and/or Terra (T)
# conf_threshold sets the threshold for the confidence that a hotspots was caused by a fire 
summarise_hotspots_annually <- function(fileList = listOfFiles, yearRange = 2003:2023, 
                                        satellite = c("A", "T"), conf_threshold = 30, outFolder) {
  
  # select years of interest
  listOfFiles1 <- listOfFiles %>% filter(year %in% yearRange)
  
  # join the 12 months of each year, separately for each year to keep file sizes tractable
  for(i in unique(listOfFiles1$year)){
    print(paste("year", i))
    
    listOfFiles2 <- listOfFiles1 %>% filter(year == i)
    
    # each month of the year
    yearlyFile <- NULL
    for(j in 1:nrow(listOfFiles2)) {
      print(paste("month", j))
      
      # load each csv file in a given year
      dat <- read.csv(paste(listOfFiles2[j,]$fileName, listOfFiles2[j,]$fileName, sep = "/"), sep = "") %>%
        # all values of "sat" in 2001 are "T", so R was reading it as TRUE, so modifying back to character "T"
        mutate(sat = ifelse(sat == T, "T", sat)) %>%  
        # select "fire" hotspots; thus, omit hotspots that the algorithm flagged as likely being from other sources i.e. exclude volcanoes, other static sources, or offshore (e.g. industrial).
        filter(type == 0) %>%
        # omit hotspots with "low" confidence (described as <30% :  https://modis-fire.umd.edu/files/MODIS_C6_C6.1_Fire_User_Guide_1.0.pdf)
        filter(conf >= conf_threshold) %>%
        # add Date
        mutate(date_ = as.Date(paste(substr(YYYYMMDD, 1, 4), substr(YYYYMMDD, 5, 6), substr(YYYYMMDD, 7, 8), sep = "-"))) %>%
        # filter for satellite(s) of interest
        filter(sat %in% satellite) 
      
      yearlyFile <- bind_rows(yearlyFile, dat)
    }
    
    # scan for duplicate hotspots. In cases where likely duplicates identified, select one randomly.
    yearlyFile1 <- remove_duplicates_fun(yearlyFile) 
    
    # summarise daily FRP sum separately for day and night 
    yearlySummary <- yearlyFile1$duplicates_removed %>%
      st_as_sf(coords = c("lon", "lat"), crs = st_crs(4326)) %>%
      dplyr::select(date_, dn, FRP)  %>%
      # add spatial grid cell ID by joining with the grid
      st_join(gridTemplate) %>%
      data.frame() %>% 
      # summarise for each grid cell on each day and night ("dn")
      group_by(date_, ID, dn) %>%
      # calc FRP sum, and the number of hotspots contributing to each value of FRP sum
      summarise(FRP_sum = sum(FRP), 
                n = n()) 
    
    # save each yearly file
    saveRDS(
      list(events = yearlySummary, 
           duplicate_summary = yearlyFile1$summary_of_duplicates %>% mutate(year = i),
           duplicates = yearlyFile1$duplicates_removed %>% filter(wasduplicated == "y")), 
      paste("E:/Project_pf/1_Input/Fire/", outFolder, "/FRP_daily_", i, ".rds", sep = ""))
  }
}

# calculate day and night FRP_sum for each day, and save one output file for each year. Repeating for Terra, and Terra + Aqua satellites
summarise_hotspots_annually(yearRange = 2003:2023, satellite = c("A", "T"), outFolder = "hotspotFRPSum_20240229_TERRAandAQUA")
summarise_hotspots_annually(yearRange = 2001:2023, satellite = c("T"), outFolder = "hotspotFRPSum_20240229_TERRA")
summarise_hotspots_annually(yearRange = 2003:2023, satellite = c("A"), outFolder = "hotspotFRPSum_20240229_AQUA")


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 3. Evaluate global trends in extreme FRP sum events ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
source("E:/Project_pf/3_Code/Summarising global FRP_functions v1.R")

# data are stored in yearly RDS files; retrieve file names
wd_1 <- "E:/Project_pf/1_Input/Fire/hotspotFRPSum_20240229_TERRA"; setwd(wd_1)
frpsum_list <- list.files(pattern = "rds")
# but drop 2001 and 2002 because of unequal sampling effort
# frpsum_list <- frpsum_list[!frpsum_list %in% c("FRP_daily_2001.rds", "FRP_daily_2002.rds")]

# read a reduced version without the really small events (to save memory)
FRP_sum <- read_summarised_frp(fileList = frpsum_list, n = 100000)

# Report some sample sizes
sum(FRP_sum$n_events) # how many "events"?
sum(FRP_sum$n_hotspots_total) # report how many hotspots used to create those events? 

# read summary of duplicates and report how many (and what proportion of) events associated with a duplication
duplicate_summary <- read_duplicate_summary(frpsum_list) %>% group_by(n_hs) %>% summarise(n = sum(n)) 
duplicate_summary %>% filter(n_hs != 1) %>% summarise(n = sum(n)) # how many duplicates removed?
duplicate_summary %>% filter(n_hs != 1) %>% summarise(n = sum(n)) / sum(FRP_sum$n_hotspots_total) * 100 # percentage of duplicates

# map duplicates
duplicates <- map_duplicate_locations(frpsum_list)
duplicates %>% ggplot(aes(lon, lat)) + 
  geom_hex(bins = 50) + 
  scale_fill_gradient2(midpoint = 500,
                       high = scales::muted("darkred"), 
                       low = scales::muted("steelblue"))

# calculate percentiles of FRP_sum
percentiles <- data.frame(FRP_sum = FRP_sum$frp_sum_vec) %>%
  filter(FRP_sum > 0) %>%
  mutate(FRP_sum_percentile = rank(FRP_sum)/length(FRP_sum))

# set threshold of what defines a major event, and identify FRP_sum value corresponding to the percentile threshold
p_thresh <- 0.9999 
percentile_threshold <- percentiles %>% arrange(desc(FRP_sum_percentile)) %>% slice(which.min(abs(FRP_sum_percentile - p_thresh)))

# identify major events (i.e. those exceeding the threshold)
FRP_sum_all <- FRP_sum$largest_events %>%
  arrange(desc(FRP_sum)) %>%
  filter(FRP_sum >= percentile_threshold$FRP_sum) %>%
  mutate(year = year(date_))

# report how many extreme events?
nrow(FRP_sum_all)

# how many events were in the same cell in successive days
sameCell <- FRP_sum_all %>%
  mutate(date_ = as.Date(date_)) %>%
  arrange(date_) %>%
  group_by(ID) %>%
  arrange(ID, date_) %>%
  mutate(n_hs_in_cell = n(),
         dayDiff = as.numeric(difftime(date_, lag(date_), units = "days")),
         dayDiff = ifelse(is.na(dayDiff), 0, dayDiff))
length(unique(sameCell$ID)) # number of cells containing an extreme event
sameCell %>% filter(n_hs_in_cell > 1) %>% nrow() # unique cells containing multiple extreme events
nrow(sameCell %>% filter(n_hs_in_cell > 1, dayDiff %in% c(1:7))) / nrow(FRP_sum_all) # proportion of extreme events occurring in the same cell in 7 day period

# calculate annual number of extreme events
FRP_sum_count <- FRP_sum_all %>%
  group_by(year) %>%
  summarise(tot = n())

# add spatial reference to extreme events 
grid_frp <- grid_centroid %>% 
  left_join(FRP_sum_all) %>% 
  filter(!is.na(FRP_sum)) 

# identify major events for day and night separately
FRP_sum_dn <- FRP_sum$largest_events_dn %>%
  group_by(dn) %>%
  arrange(desc(FRP_sum)) %>%
  slice_head(n = nrow(FRP_sum_all)) %>%
  mutate(year = year(date_))

# calculate annual number of extreme events, day and night separately
FRP_sum_dn_count <- FRP_sum_dn%>% 
  group_by(year, dn) %>%
  summarise(tot = n()) %>%
  mutate(dn = factor(dn))

# set common axes among plots
ymax <- max(max(FRP_sum_count$tot), max(FRP_sum_dn_count$tot))

# fit GAM of yearly count of extreme events globally
frpCount_gam1 <- gam(tot ~ s(year), select = T,  family = "nb", data = FRP_sum_count)
summary(frpCount_gam1)

# predict from model
FRP_sum_count <- predict_fun(mod = frpCount_gam1, dat = FRP_sum_count)

# report model-estimated change in frequency from 2001 to 2023
FRP_sum_count[FRP_sum_count$year == 2023,]$fit.exp / FRP_sum_count[FRP_sum_count$year == min(FRP_sum_count$year),]$fit.exp

# plot time series of count of extreme events, 
plot_frp <- ggplot(FRP_sum_count, aes(year, tot)) +
  geom_line(colour = "grey50") +
  geom_point() +
  geom_line(aes(y = fit.exp), colour = "steelblue", linewidth = 1) +
  geom_ribbon(aes(ymin = lcl, ymax = ucl), alpha = 0.2) +
  labs(y = "Count", title = "N#FRP") +
  coord_cartesian(ylim = c(0,ymax)) 
plot_frp

# fit model of extreme events, fitting separate smooth for day and night
frpsum_dn_gam <- gam(tot ~ s(year, by = dn), select = T, family = "nb", data = FRP_sum_dn_count)
summary(frpsum_dn_gam)

# predict
FRP_sum_dn_count <- predict_fun(mod = frpsum_dn_gam, dat = FRP_sum_dn_count)

# plot time series of count of extreme events, separating by day and night
plot_frp_dn <- ggplot(FRP_sum_dn_count, aes(year)) +
  facet_wrap(~dn, labeller = as_labeller(c(`D` = "Day N#FRP", `N` = "Night N#FRP"))) +
  geom_line(aes(y = tot), colour = "grey50") +
  geom_point(aes(y = tot)) +
  geom_line(aes(y = fit.exp), colour = "steelblue", linewidth = 1) +
  geom_ribbon(aes(ymin = lcl, ymax = ucl), alpha = 0.2) +
  theme_minimal() +
  labs(y = "Number of extreme events") +
  coord_cartesian(ylim = c(0,ymax)) +
  theme(axis.title.y = element_blank(), axis.text.y = element_blank(), strip.text = element_text(size = 14),
        panel.grid.minor = element_blank()) 
plot_frp_dn

# select the top n_per_year events from each year
n_per_year = 20
FRP_sum_yearly <- FRP_sum$largest_events %>%
  mutate(year = year(date_)) %>%
  group_by(year) %>%
  arrange(desc(FRP_sum)) %>%
  slice_head(n = n_per_year)

# calculate mean of top 20 each year
FRP_sum_yearly_mean <- FRP_sum_yearly %>% group_by(year) %>% summarise(FRP_sum = mean(FRP_sum))

# fit model
frp_magnitude_gam1 <- gam(FRP_sum ~ s(year), select = T, data = FRP_sum_yearly_mean)
summary(frp_magnitude_gam1)

# predict
FRP_sum_yearly_mean <- predict_fun(mod = frp_magnitude_gam1, dat = FRP_sum_yearly_mean, logBackTrans = F)

# change in magnitude
FRP_sum_yearly_mean[FRP_sum_yearly_mean$year == 2023,]$fit / FRP_sum_yearly_mean[FRP_sum_yearly_mean$year == 2003,]$fit

# plot using mean of those 20 obs per year
plot_topFRPeventsYearly_meanPoints <- ggplot(FRP_sum_yearly_mean, aes(year, FRP_sum)) +
  geom_line(colour = "grey50") +
  geom_point() +
  geom_line(aes(year, fit), colour = "steelblue", linewidth = 1) +
  geom_ribbon(aes(year, ymin = lcl, ymax = ucl), alpha = 0.2) +
  labs(y = "N#FRP (MW)", title = paste("Mean of top", n_per_year, "\nN#FRP events/year") ) +
  lims(y = c(0, max(FRP_sum_yearly_mean$FRP_sum))) 
plot_topFRPeventsYearly_meanPoints


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 4. Load biome spatial data ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

# get polygons of continents
worldPolys <- ne_countries(returnclass = "sf", scale = "medium")
sf_use_s2(FALSE)
worldPolys_forPlotting <- worldPolys %>% st_make_valid() %>% st_union() %>% st_simplify(dTolerance = 0.05); plot(worldPolys_forPlotting)
continent <- worldPolys %>% group_by(continent) %>% summarise()
continent$area_continent <- units::set_units(st_area(continent), km^2)

# read WWF biomes
wwf <- st_read("E:/Project_pf/1_Input/Ecoregions2017/Ecoregions2017.shp")
wwf1 <- wwf %>% group_by(BIOME_NAME) %>% summarise() 
wwf1$area_biome <- units::set_units(st_area(wwf1), km^2)

# exploratory graph of area of the biomes
wwf1 %>% data.frame() %>% 
  ggplot(aes(area_biome, BIOME_NAME)) +
  geom_bar(stat = "identity")

# select biogeographic realms from WWF delineation
wwf_realm <- wwf %>% mutate(REALM = ifelse(REALM %in% c("Australasia","Oceania"), "Australasia & Oceania", REALM)) %>% group_by(REALM) %>% summarise()
wwf_realm$area_realm <- units::set_units(st_area(wwf_realm), km^2)

# exploratory graph of the area of the realms
wwf_realm %>% data.frame() %>% 
  ggplot(aes(area_realm, REALM)) +
  geom_bar(stat = "identity")

# match biomes and realms with FRP events
grid_frp1 <- st_join(grid_frp, wwf1) %>%
  st_join(wwf_realm) %>%
  st_join(continent[,c("continent", "area_continent")])

# convert to Robinson projection so that land at the poles doesn't look ridiculously big
robinson_crs <- "+proj=robin +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84"
grid_frp_rob <- st_transform(grid_frp, robinson_crs) %>% mutate(Y = st_coordinates(.)[,2])
worldPolys_forPlotting_rob <- st_transform(worldPolys_forPlotting, robinson_crs)

# join by biogeographic realm
wwf_realm_rob <-  wwf_realm  %>% st_make_valid() %>% st_simplify(dTolerance = 0.05) %>% st_transform(robinson_crs) %>% 
  filter(!is.na(REALM), REALM != "N/A", REALM != "Antarctica") %>% rename(Realm = REALM)

# map for publication (Fig 1a)
set.seed(123) # randomising plot order so later no systematic bias in plot order
hottestFiresMap <- ggplot(data = grid_frp_rob %>% slice(., sample(1:n()))) +
  theme_map() +
  theme(panel.grid.major = element_line(linewidth = 0.25, colour = "grey")) +
  scale_x_continuous(breaks = c(-180, -90, 0, 90, 180)) +
  scale_y_continuous(breaks = c(-60, -30, 0, 30, 60), limits = c(-5304825, 7700581)) +
  geom_sf(data = wwf_realm_rob, aes(fill = Realm), alpha = 0.2, colour = NA) +
  geom_sf(aes(colour = year)) + # randomising plot order of points so as not to bias appearance of year effect
  scale_colour_gradient2(midpoint = 2011, high = scales::muted("darkred"), low = scales::muted("steelblue"), mid = "grey", name = "Year")  +
  scale_fill_lancet() +
  theme(legend.text = element_text(size = 9), legend.title = element_text(size = 11))
hottestFiresMap

# multipanel
multiGraphs <- plot_grid(plot_frp, plot_frp_dn, NULL, plot_topFRPeventsYearly_meanPoints, rel_widths = c(1, 1.6, 0.1, 1.3), align = "h", axis = "lrtb", nrow = 1, 
                         labels = c("(b)", "", "(c)"), label_fontface = "plain", label_size = 12)
multiPlot <- plot_grid(hottestFiresMap,
                       multiGraphs, nrow = 2, rel_heights = c(1.5,1),
                       labels = c("(a)", ""), label_fontface = "plain", label_size = 12) +
  draw_label(paste("Frequency of N#FRP b	% ", p_thresh*100, "th percentile", sep = ""), x = 0.34, y = 0.42)

#jpeg("C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/map_frp_sum_realm_20240301.jpg",width=9,height=7, units = "in", res = 1000)
multiPlot
dev.off()

# annual maps for supplement 
#jpeg("C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/map_facet_year.jpg",width=10,height=13, units = "in", res = 1000)
ggplot(data = grid_frp_rob %>% slice(., sample(1:n()))) +
  theme_map() +
  theme(panel.grid.major = element_line(linewidth = 0.25, colour = "grey")) +
  scale_x_continuous(breaks = c(-180, -90, 0, 90, 180)) +
  scale_y_continuous(breaks = c(-60, -30, 0, 30, 60), limits = c(-5304825, 7700581)) +
  geom_sf(aes(colour = year)) + # randomising plot order of points so as not to bias appearance of year effect
  scale_colour_gradient2(midpoint = 2011, high = scales::muted("darkred"), low = scales::muted("steelblue"), mid = "grey", name = "Year")  +
  scale_fill_lancet() +
  theme(legend.text = element_text(size = 9), legend.title = element_text(size = 11)) + 
  facet_wrap(~year, ncol = 3) + theme(legend.position = "none") +
  geom_sf(data = worldPolys_forPlotting_rob, fill = "grey90", colour = "grey60")  + geom_sf(colour = "black")
dev.off()


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 5. Ratio of extreme events in biomes and realms ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

### ### ### ### ### ### ###  
# REALMS
### ### ### ### ### ### ###  

# create data frame to insert zeroes for a given year when no major events occurred in a realm
realm_year_zeroes <- expand_grid(REALM = unique(grid_frp1$REALM),  
                                 year = sort(unique(grid_frp1$year)),
                                 n = 0)

# summarise proportion of events occurring in each biome
frp_realm_summary <- grid_frp1 %>% data.frame() %>%
  group_by(REALM) %>%
  summarise(n = n(), area = as.numeric(mean(area_realm))) %>%
  filter(!is.na(REALM)) %>%
  mutate(n_prop = n/sum(n), area_prop = area/sum(area, na.rm = T),
         events_to_area = n_prop/area_prop) %>%
  arrange(events_to_area) %>%
  mutate(REALM = factor(REALM, levels = .$REALM),
         events_to_area_cat = ifelse(events_to_area > 1, "high", "low")) 

# plot events to area ratio
p_ratio_realm <- ggplot(frp_realm_summary, aes(events_to_area, REALM, fill = events_to_area_cat)) +
  geom_bar(stat = "identity", alpha = 0.5) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(x = "Ratio of major N#FRP events to realm area", y = element_blank()) +
  geom_text(aes(label = paste("N =", n)), nudge_x = c(rep(0.33, 3), rep(0.33, 3))) +
  theme(panel.grid.major = element_blank(), axis.text.y = element_text(size = 11)) +
  lims(x = c(0,max(frp_realm_summary$events_to_area) + 1)) + 
  scale_fill_manual(values = c("darkorange", "grey50")) +
  theme(legend.position = "none")+
  scale_y_discrete(labels = label_wrap_gen(width = 30)) 
p_ratio_realm


### ### ### ### ### ### ###  
# BIOMES
### ### ### ### ### ### ### 

# create data frame to insert zeroes for a given year when no major events occurred in a biome
biome_year_zeroes <- expand_grid(BIOME_NAME = unique(grid_frp1$BIOME_NAME),  
                                 year = sort(unique(grid_frp1$year)),
                                 n = 0)

# summarise proportion of events occurring in each biome
grid_frp1_summary <- grid_frp1 %>% data.frame() %>%
  group_by(BIOME_NAME) %>%
  summarise(n = n(), area = as.numeric(mean(area_biome))) %>%
  filter(!is.na(BIOME_NAME)) %>%
  mutate(n_prop = n/sum(n), area_prop = area/sum(area, na.rm = T),
         events_to_area = n_prop/area_prop) %>%
  arrange(events_to_area) %>%
  mutate(BIOME_NAME = factor(BIOME_NAME, levels = .$BIOME_NAME),
         events_to_area_cat = ifelse(events_to_area > 1, "high", "low")) 

# plot events to area ratio
p_ratio <- ggplot(grid_frp1_summary, aes(events_to_area, BIOME_NAME, fill = events_to_area_cat)) +
  geom_bar(stat = "identity", alpha = 0.5) +
  geom_vline(xintercept = 1, linetype = "dashed") +
  labs(x = "Ratio of major N#FRP events to biome area", y = element_blank()) +
  geom_text(aes(label = paste("N =", n)), nudge_x = c(rep(1.4, 10), rep(0.65, 3))) +
  theme(panel.grid.major = element_blank(), axis.text.y = element_text(size = 11)) +
  lims(x = c(0,max(grid_frp1_summary$events_to_area) + 1)) + 
  scale_fill_manual(values = c("darkorange", "grey50")) +
  theme(legend.position = "none")+
  scale_y_discrete(labels = label_wrap_gen(width = 34)) 
p_ratio


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 6. Temporal trends among key biomes ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

# plot trends among biomes that are most affected by intense fires
highRatio <- grid_frp1_summary %>% filter(events_to_area > 1)
greater_than_n <- grid_frp1_summary %>% arrange(desc(n)) %>% slice_head(n = 6)

# plot using the sample of top 99.99 percentile events; only select biomes that are significantly affected by major FRP events
count_by_biome <- grid_frp1 %>% data.frame() %>%
  mutate(BIOME_NAME = factor(BIOME_NAME, levels = rev(grid_frp1_summary$BIOME_NAME))) %>%
  group_by(BIOME_NAME, year) %>% 
  summarise(n = n()) %>%
  # add in zeroes for years without major events in a given biome
  bind_rows(biome_year_zeroes) %>%
  group_by(BIOME_NAME, year) %>%
  summarise(n = sum(n)) 

# only select biomes that were substantially affected by intense fires
glm_count_dat <- count_by_biome %>%
  filter(BIOME_NAME %in% greater_than_n$BIOME_NAME) 

# fit GLM separately for each biome
mod_list <- list()
for(i in unique(glm_count_dat$BIOME_NAME)){
  print(i)
  mod_list[[i]] <- gam(n ~ year, family = "nb", data = glm_count_dat %>% filter(BIOME_NAME == i))
}

# retrieve beta_yr, p-value, and deviance explained
coef_df <- list()
for(i in names(mod_list)) {
  print(i)
  modsummary <- summary(mod_list[[i]])
  coef_df[[i]] <- data.frame(
    BIOME_NAME = i,
    intercept = modsummary$p.coeff[["(Intercept)"]],
    beta_yr = modsummary$p.coeff[["year"]],
    p_yr = modsummary$p.pv[["year"]],
    dev_expl = modsummary$dev.expl)}

coef_df <- bind_rows(coef_df) %>% 
  arrange(p_yr) %>%
  mutate(p_yr = plyr::round_any(p_yr, accuracy = 0.001),
         # reorder factor so that panels plot from most to least significant
         BIOME_NAME = factor(BIOME_NAME, levels = .$BIOME_NAME),
         p_yr_txt = ifelse(p_yr > 0, paste("p =", p_yr), paste("p <", 0.001))) %>%
  # add model formula
  mutate(p_yr_txt = paste(p_yr_txt, ", d.e. = ", plyr::round_any(dev_expl, accuracy = 0.01), sep = ""),
         lp = paste("log(y) = ", plyr::round_any(intercept,accuracy = 0.1),
                    " + ",
                    plyr::round_any(beta_yr, accuracy = 0.01), 
                    "x", sep = ""))

# predict from model
predictions <- list()
for(i in names(mod_list)) {
  print(i)
  d <- data.frame(year = seq(min(glm_count_dat$year), max(glm_count_dat$year), 1),
                  BIOME_NAME = i)
  predictions[[i]] <- predict_fun(mod_list[[i]], d) 
}

predictions <- bind_rows(predictions) %>% left_join(coef_df) %>%  mutate(BIOME_NAME = factor(BIOME_NAME, levels = coef_df$BIOME_NAME))

# percentage change
predictions[predictions$year == 2023 & predictions$BIOME_NAME == "Temperate Conifer Forests",]$fit.exp / predictions[predictions$year == 2003 & predictions$BIOME_NAME == "Temperate Conifer Forests",]$fit.exp
predictions[predictions$year == 2023 & predictions$BIOME_NAME == "Boreal Forests/Taiga",]$fit.exp / predictions[predictions$year == 2003 & predictions$BIOME_NAME == "Boreal Forests/Taiga",]$fit.exp

# get max value for axis
maxNonBoreal <- max(
  max(count_by_biome[!count_by_biome$BIOME_NAME %in% c("Boreal Forests/Taiga"), ]$n),
  max(predictions[!count_by_biome$BIOME_NAME %in% c("Boreal Forests/Taiga"), ]$ucl, na.rm = T))

# plot trend by biome
p_trend_biome <- count_by_biome %>% 
  left_join(coef_df) %>%
  mutate(BIOME_NAME = factor(BIOME_NAME, levels = coef_df$BIOME_NAME)) %>%
  filter(!is.na(BIOME_NAME)) %>%
  ggplot()+
  facet_wrap(~BIOME_NAME, ncol = 2, scales = "free_y", labeller = label_wrap_gen(width = 34)) +
  geom_bar(aes(year, n), stat = "identity", fill = "grey") +
  geom_line(aes(year, n), colour = "gray30", linewidth = 0.5) +
  geom_line(data = predictions %>% filter(p_yr <= 0.05), aes(year, fit.exp), colour = "darkred", linewidth = 1) +
  geom_ribbon(data = predictions %>% filter(p_yr <= 0.05), aes(year, ymin = lcl, ymax = ucl), fill = "darkred", alpha = 0.2) +
  geom_line(data = predictions %>% filter(p_yr > 0.05), aes(year, fit.exp), colour = "steelblue4", linewidth = 1) +
  geom_ribbon(data = predictions %>% filter(p_yr > 0.05), aes(year, ymin = lcl, ymax = ucl), fill = "steelblue4", alpha = 0.3) +
  labs(y = "Frequency of extreme events", x = "Year") +
  lims(y = c(0,maxNonBoreal)) +
  theme(strip.text.x = element_text(size = 11))
p_trend_biome

# change axes for some panels
p_trend_biome1 <- p_trend_biome + 
  facetted_pos_scales(
    # modify scale for some panels with less data so that trend is more visible
    y = list(BIOME_NAME == "Boreal Forests/Taiga" ~ scale_y_continuous(limits = c(0, max(count_by_biome[count_by_biome$BIOME_NAME == "Boreal Forests/Taiga",]$n, na.rm = T)))))

# add labels (diff locations depending on axis limits)
p_trend_biome2 <- p_trend_biome1 +  
  # label p values
  geom_text(data = coef_df %>% filter(BIOME_NAME %in% c("Boreal Forests/Taiga")), 
            aes(x = 2003, y = 175, label = p_yr_txt), colour = "darkred", hjust = 0)  +
  geom_text(data = coef_df %>% filter(BIOME_NAME %in% c("Temperate Conifer Forests")), 
            aes(x = 2003, y = 100, label = p_yr_txt), colour = "darkred", hjust = 0) +
  geom_text(data = coef_df %>% filter(!BIOME_NAME %in% c("Temperate Conifer Forests", "Boreal Forests/Taiga")), 
            aes(x = 2003, y = 100, label = p_yr_txt), colour = "steelblue4", hjust = 0) +
  # label model formula
  geom_text(data = coef_df %>% filter(BIOME_NAME %in% c("Boreal Forests/Taiga")), 
            aes(x = 2003, y = 210, label = lp), colour = "darkred", hjust = 0) +
  geom_text(data = coef_df %>% filter(BIOME_NAME %in% c("Temperate Conifer Forests")), 
            aes(x = 2003, y = 120, label = lp), colour = "darkred", hjust = 0) +
  geom_text(data = coef_df %>% filter(!BIOME_NAME %in% c("Temperate Conifer Forests", "Boreal Forests/Taiga")), 
            aes(x = 2003, y = 120, label = lp), colour = "steelblue4", hjust = 0)
p_trend_biome2

# export multi panel plot
#jpeg("C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/frp_sum_biomes_realm_continent_20240301.jpg",width=12.5,height=7, units = "in", res = 1000)
plot_grid(
  plot_grid(p_ratio_realm, NULL, p_ratio, ncol = 1, rel_heights = c(0.4,0.05,1), axis = "lrtb", align = "v", labels = c("(a)","", "(b)"), label_fontface = "plain", label_size = 12),
  p_trend_biome2, 
  labels = c("","(c)"), label_fontface = "plain", label_size = 12,
  rel_widths = c(1.1,1), nrow = 1) 
dev.off()



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# 7. Supplementary Analysis: Compare Terra with Terra + Aqua ----
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 

#summarise_hotspots_annually(yearRange = 2001:2023, satellite = c("T"), outFolder = "hotspotFRPSum_20240229_TERRA")
#summarise_hotspots_annually(yearRange = 2003:2023, satellite = c("A"), outFolder = "hotspotFRPSum_20240229_AQUA")

# load FRP_sum events calculated using Terra + Aqua separately
wd_terra <- "C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/MODIS_global/hotspotFRPSum_20240229_TERRA"; setwd(wd_terra)
frpsum_list_terra <- list.files(pattern = "rds")
FRP_sum_T <- read_summarised_frp(fileList = frpsum_list_terra, n = 100000)

wd_aqua <- "C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/MODIS_global/hotspotFRPSum_20240229_AQUA"; setwd(wd_aqua)
frpsum_list_aqua <- list.files(pattern = "rds")
FRP_sum_A <- read_summarised_frp(fileList = frpsum_list_aqua, n = 100000)

# calculate percentiles of FRP_sum
percentiles_T <- data.frame(FRP_sum = FRP_sum_T$frp_sum_vec) %>%
  filter(FRP_sum > 0) %>%
  mutate(FRP_sum_percentile = rank(FRP_sum)/length(FRP_sum))
percentile_T_threshold <- percentiles_T %>% arrange(desc(FRP_sum_percentile)) %>% slice(which.min(abs(FRP_sum_percentile - p_thresh)))

percentiles_A <- data.frame(FRP_sum = FRP_sum_A$frp_sum_vec) %>%
  filter(FRP_sum > 0) %>%
  mutate(FRP_sum_percentile = rank(FRP_sum)/length(FRP_sum))
percentile_A_threshold <- percentiles_A %>% arrange(desc(FRP_sum_percentile)) %>% slice(which.min(abs(FRP_sum_percentile - p_thresh)))

# major event count per year, then converted to proportion
FRP_sum_T_prop <- FRP_sum_T$largest_events %>%
  arrange(desc(FRP_sum)) %>%
  filter(FRP_sum >= percentile_T_threshold$FRP_sum) %>%
  mutate(year = year(date_)) %>%
  group_by(year) %>%
  summarise(tot = n()) %>%
  mutate(prop = tot/sum(tot))

FRP_sum_A_prop <- FRP_sum_A$largest_events %>%
  arrange(desc(FRP_sum)) %>%
  filter(FRP_sum >= percentile_A_threshold$FRP_sum) %>%
  mutate(year = year(date_)) %>%
  group_by(year) %>%
  summarise(tot = n()) %>%
  mutate(prop = tot/sum(tot))

# graph data points each year (Fig S2a)
plot_satellite_comparison <- ggplot() +
  geom_line(data = FRP_sum_A_prop, aes(year, prop), colour = "steelblue") +
  geom_point(data = FRP_sum_A_prop, aes(year, prop), colour = "steelblue") +
  geom_line(data = FRP_sum_T_prop, aes(year, prop), colour = "darkred") +
  geom_point(data = FRP_sum_T_prop, aes(year, prop), colour = "darkred") +
  labs(y = "Proportion")  +
  geom_text(aes(label = "Terra", x = 2001, y = 0.095), colour = "darkred", hjust = 0, size = 4.2) +
  geom_text(aes(label = "Aqua", x = 2001, y = 0.08), colour = "steelblue", hjust = 0, size = 4.2)
plot_satellite_comparison

# graph trend each year (Fig S2b)
plot_satellite_comparison1 <- ggplot() +
  geom_smooth(data = FRP_sum_A_prop, aes(year, prop), method = "gam", formula = y ~ s(x, bs = "cs"), method.args = list(family = "quasipoisson"), colour = "steelblue", alpha = 0.2, fill = "steelblue") +
  geom_smooth(data = FRP_sum_T_prop, aes(year, prop), method = "gam", formula = y ~ s(x, bs = "cs"), method.args = list(family = "quasipoisson"), colour = "darkred", alpha = 0.2, fill = "darkred") +
  labs(y = element_blank()) +
  geom_text(aes(label = "Terra", x = 2001, y = 0.095), colour = "darkred", hjust = 0, size = 4.2) +
  geom_text(aes(label = "Aqua", x = 2001, y = 0.08), colour = "steelblue", hjust = 0, size = 4.2)
plot_satellite_comparison1

# export multi panel figure (Fig S2)
#jpeg("C:/Users/cxc/OneDrive - University of Tasmania/UTAS/Fire/Global FRP/Analysis/satellite_comparison_20240304.jpg",width=6,height=2.5, units = "in", res = 1000)
plot_grid(plot_satellite_comparison, plot_satellite_comparison1, align = "h", axis = "lrtb")
dev.off()
