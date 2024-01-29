setwd('C:/Users/punco/OneDrive/Desktop/DS 785 Assignments')

# packages
library(dplyr)
library(stringr)
library(tidyr)
library(sjmisc)
library(ipred)
library(ggplot2)
library(caret)
library(gt)
library(PRROC) # calculate AUC for PR curve
library(rpart)
library(rvest) # scrape live data
library(baseballr)
library(tidyverse)

# pitches 2021
pitches <- read.csv("pitches2021.csv")
head(pitches)

# extract batter name from description
pitches <- pitches %>% mutate(batter_from_des = sub("^(\\S*\\s+\\S+).*", "\\1", des))


# remove pitch-related data (spin rate, speed, etc.)
pitches2 <- pitches %>% dplyr::select(game_date, player_name, batter, pitcher, events, description, stand, p_throws, home_team, away_team, balls,
                                      strikes, on_3b, on_2b, on_1b, outs_when_up, game_pk, delta_home_win_exp, delta_run_exp, count, des, batter_from_des)

# players
players <- read.csv('player_names.csv')
head(players)

# merge players with ids
players <- players %>% dplyr::mutate(player_name = paste(first_name, last_name))
pitches3 <- left_join(pitches2, players, by = c('batter'='id'))
# Remove pick offs and challenges
pitches3 <- pitches3[!str_detect(pitches3$des, "challenged"),] # challenged plays
pitches3 <- pitches3[!str_detect(pitches3$des, "With"),] #pick offs

# get venue and weather
parks <- data.frame(team=unique(pitches$home_team))
parks$ballpark <- c('Safeco Field', 'Globe Life Park in Arlington', 'Angel Stadium of Anaheim', 'Guaranteed Rate Field', 'Minute Maid Park',
                    'Petco Park', 'Oriole Park at Camden Yards', 'Progressive Field', 'Comerica Park', 'Yankee Stadium', 'Rogers Centre',
                    'Dodger Stadium', 'Target Field', 'O.co Coliseum', 'Tropicana Field', 'Chase Field', 'AT&T Park', 'Fenway Park', 'Kaufmann Stadium',
                    'Coors Field', 'Nationals Park', 'Citi Field', 'Marlins Park', 'SunTrust Park', 'Busch Stadium', 'Citizens Bank Park', 'PNC Park',
                    'Great American Ball Park', 'Miller Park', 'Wrigley Field')

pitches4 <- left_join(pitches3, parks, by = c('home_team'='team'))

# Home Run binary variable
pitches4 <- pitches4 %>% dplyr::mutate(home_run = ifelse(events=='home_run',1,0))

# More pitches cleaning
pitches4$player_name.x = iconv(pitches4$player_name.x, from = 'UTF-8', to = 'ASCII//TRANSLIT')
# split name on ',' then flip last and first name
pitches5 <- pitches4 %>% extract(player_name.x, c('last_name','first_name'), "([^,]+), ([^)]+)")
pitches5 <- pitches5 %>% mutate(pitcher_name = paste(first_name, last_name))
pitches6 <- pitches5 %>% dplyr::select(game_date, pitcher_name, batter_from_des, stand, p_throws, balls, strikes, outs_when_up, delta_home_win_exp,
                                       delta_run_exp, count, home_run)

# Get batter data
stats <- read.csv('2019_stats_mlb.csv')
# remove weird chars by converting from utf to ascii
stats$Name_x = iconv(stats$Name_x, from = 'UTF-8', to = 'ASCII//TRANSLIT')
# remove non-alpha chars
stats$Name_x = gsub('[*#]', '', stats$Name_x)
#stats$Name_x = trimws(stats$Name_x)
stats2 <- stats %>% dplyr::select(Name_x, HR_perc, Age, PA, ISO, FB., Pull., Cent., Oppo.)
names(stats2)[1] <- 'Name'

# get pitcher data
pitcher_stats <- read.csv('pitcher_stats_2019.csv')

# rookies and replacement level players

# Merge batter stats with pitches data
pitches7 <- left_join(pitches6, stats2 %>% dplyr::select(Name, HR_perc, Age, PA, ISO, FB.), by = c('batter_from_des'='Name'))
# convert char vectors to numeric percentages
pitches7$FB. <- as.numeric(sub("%", "", pitches7$FB.))/100

# Merge pitcher stats with updated pitches data
pitches8 <- left_join(pitches7, pitcher_stats %>% dplyr::select(Name, HR., FB.), by = c('pitcher_name'='Name'))

# Fill NA batter and pitcher stats with replacement level stats
hist(pitches8$delta_run_exp) # to show distribution of hitter fly ball % 

pitches8$HR_perc[is.na(pitches8$HR_perc)]<-.034
pitches8$FB..x[is.na(pitches8$FB..x)]<-.26
pitches8$FB..y[is.na(pitches8$FB..y)]<-.257
pitches8$HR.[is.na(pitches8$HR.)]<-.031
pitches8$delta_run_exp[is.na(pitches8$delta_run_exp)]<- -0.018
# remove NAs count and win expec
pitches8 <- pitches8[!is.na(pitches8$count),]
pitches8 <- pitches8[!is.na(pitches8$delta_home_win_exp),]

# log loss function
LogLoss <- function(pred, res){
  (-1/length(pred)) * sum (res * log(pred) + (1-res)*log(1-pred))
}

# Machine learning models
# logistic regression
set.seed(3)
hr_data = pitches8 %>% dplyr::select(-c(game_date, pitcher_name, batter_from_des, balls, strikes, PA, ISO, Age)) # remove id and other unnecessary columns

# 10 fold CV
n = dim(hr_data)[1]
nfolds = 10
groups = rep(1:nfolds, n)
cvgroups = sample(groups, n)

# initialize prediction vector
CVpredictions = numeric(length = n)

for(ii in 1:nfolds) {
  groupii = (cvgroups == ii)
  trainset = hr_data[!groupii,]
  testset = hr_data[groupii,]
  
  # Logistic regression model
  fit <- glm(home_run ~ ., data = trainset, family = 'binomial')
  predicted = predict(fit, newdata = testset, type = "response")   # predict for test set
  
  CVpredictions[groupii] = predicted
}

hr_data$hr_prob <- CVpredictions     
log_model_data <- hr_data %>% dplyr::select(home_run, hr_prob) # data for model assessment

# class outcome distribution
classes_tbl <- tibble(Outcome = c("No Home Run", "Home Run"), Proportion = as.numeric(prop.table(table(hr_data$home_run))))
classes_gt <- gt(classes_tbl) %>% tab_header(title = "Table 1. Home Run Class Imbalance", subtitle = "proportion of pitches that were home runs (2021 season)")  %>% 
              fmt_number(columns = "Proportion", decimals = 3) %>% cols_align(align="center")
classes_gt

# histogram of predicted probs - Logistic Regression
ggplot(log_model_data, aes(hr_prob)) + geom_histogram() + theme_minimal() + 
  labs(title = "Figure 1. Predicted Probabilities of Hitting a Home Run - Logistic Regression", x = "Predicted Probability", y = "Count") + 
  xlim(c(.01, 1)) + theme(plot.title = element_text(size=22), axis.title.x = element_text(size=15), axis.title.y = element_text(size=15),
                          text = element_text(size=20))

# calculate log loss - logistic regression
LogLoss(hr_data$hr_prob, hr_data$home_run)
LogLossBinary(hr_data$home_run, hr_data$hr_prob)

# BAGGING
set.seed(3)
hr_data = pitches8 %>% dplyr::select(-c(delta_run_exp, game_date, pitcher_name, batter_from_des, balls, strikes, PA, ISO, Age)) # reset data
n_train = round(.8 * n)
train_ind = sample(1:n, n_train)
train = hr_data[train_ind,]
test = hr_data[-train_ind,]
hr_bag = bagging(home_run ~ ., data = train, coob=TRUE)
pred = predict(hr_bag, newdata = test, type="prob")

test$hr_prob <- pred
LogLoss(test$hr_prob, test$home_run)

# under sampling get 50/50 sample
data_balanced_under <- ovun.sample(home_run ~ ., data = train, method = "under", N = 9360)$data

# class outcome distribution - under sampling
classes_tbl_under <- tibble(Outcome = c("No Home Run", "Home Run"), Proportion = as.numeric(prop.table(table(data_balanced_under$home_run))))
classes_gt <- gt(classes_tbl_under) %>% tab_header(title = "Table 4. Outcome Distribution with Under Sampling", subtitle = "proportion of pitches that were home runs (2021 season)")  %>% 
  fmt_number(columns = "Proportion", decimals = 3) %>% cols_align(align="center")
classes_gt


# bagging models with under sampling method
bag_under <- bagging(home_run ~., data = data_balanced_under, coob=TRUE)

# predict
pred_under <- predict(bag_under, newdata = test, type="prob")
test$pred_under <- pred_under

# variable importance
bag_imp <- varImp(hr_bag)
# rename variables
rownames(bag_imp) <- c("Count", "Home Win Exp", "Run Exp", "FlyBall% Hitter", "FlyBall% Pitcher", "HR% Pitcher", "HR% Hitter", "Outs", "Pitcher Handedness", "Hitter Handedness")

# plot var imp
ggplot(bag_imp, aes(x=Overall, y = reorder(rownames(bag_imp), Overall))) + geom_point() + 
  geom_segment(aes(x=0,xend=Overall, y=rownames(bag_imp), yend=rownames(bag_imp)), size = 1, color="#454545") + theme_minimal() + 
  labs(title="Figure 4. Variable Importance", y="Variable Name") + theme(plot.title = element_text(size=22), axis.title.x = element_text(size=15), axis.title.y = element_text(size=15),
                                                                       text = element_text(size=20))

#############

# Model assessment - Logistic Regression
# decision thresholds
thresholds <- seq(0, 1, by=.01)

# precision and recall lists
prec_list = numeric()
rec_list = numeric()
F_score_list = numeric()

for(i in thresholds) {
  # re-classify observations based on threshold
  # confusion matrix
  conf_mat <- as.matrix(table(factor(log_model_data$hr_prob > i, levels=c(F, T)), log_model_data$home_run))
  
  # precision
  precision = conf_mat[4] / (conf_mat[4] + conf_mat[2])
  # recall
  recall = conf_mat[4] / (conf_mat[4] + conf_mat[3])
  # F score
  F_score = 2 / ((1/recall) + (1/precision))
  
  # append to lists
  prec_list = c(prec_list, precision)
  rec_list = c(rec_list, recall)
  F_score_list = c(F_score_list, F_score)
}

# optimal threshold logistic regression
opt_ind = which.max(F_score_list)

# confusion matrix at optimal threshold (GT package)
as.matrix(table(factor(log_model_data$hr_prob > thresholds[opt_ind], levels=c(F, T)), log_model_data$home_run))
log_tbl = tibble(Predicted = c("Non HR", "HR"), "Non HR"=c(691242,3931), HR=c(1137,4760))
gt(log_tbl) %>% tab_spanner(label="Actual Outcome", columns=c("Non HR", "HR")) %>% 
  tab_header(title = "Table 2. Confusion Matrix", subtitle="Logistic Regression Model with p=.13 Threshold") %>% cols_align(align="center") %>% 
  fmt_number(columns=2:3, use_seps = TRUE, decimals = 0)

# gg plot Precision-Recall Curve Logistic Regression
# used package to find AUC for PR curve
hr <- log_model_data$hr_prob[log_model_data$home_run==1]
no_hr <- log_model_data$hr_prob[log_model_data$home_run==0]
pr <- pr.curve(scores.class0 = hr, scores.class1 = no_hr, curve = T)
plot(pr)

# create data frame 
plot_df <- data.frame(p=prec_list, r=rec_list, f=F_score_list)
# plot PR curve with AUC pulled from above plot
ggplot(plot_df, aes(r, p)) + geom_point(size=3, color = "#454545") + 
  labs(title = "Figure 2. Precision-Recall Curve", x = "Recall", y = "Precision", subtitle = "AUC = .666; optimal threshold = .13") + theme_minimal() + 
  scale_y_continuous(breaks=seq(0,1,.1)) + scale_x_continuous(breaks=seq(0,1,.1)) + 
  geom_point(aes(x=r[opt_ind], y=p[opt_ind]), pch=18, size=8, color='#ff0000') +
  theme(plot.title = element_text(size=22), axis.title.x = element_text(size=15), axis.title.y = element_text(size=15),
        text = element_text(size=17))

#######

# Model assessment - Bagging
# decision thresholds
thresholds <- seq(0, 1, by=.01)

# precision and recall lists
prec_list = numeric()
rec_list = numeric()
F_score_list = numeric()

for(i in thresholds) {
  # re-classify observations based on threshold
  # confusion matrix
  conf_mat <- as.matrix(table(factor(test$pred_under > i, levels=c(F, T)), test$home_run))
  
  # precision
  precision = conf_mat[4] / (conf_mat[4] + conf_mat[2])
  # recall
  recall = conf_mat[4] / (conf_mat[4] + conf_mat[3])
  # F score
  F_score = 2 / ((1/recall) + (1/precision))
  
  # append to lists
  prec_list = c(prec_list, precision)
  rec_list = c(rec_list, recall)
  F_score_list = c(F_score_list, F_score)
}

# optimal threshold bagging
opt_ind_bag = which.max(F_score_list)

# gg plot Precision-Recall Curve Bagging
# used package to find AUC for PR curve
hr <- test$hr_prob[test$home_run==1]
no_hr <- test$hr_prob[test$home_run==0]
pr <- pr.curve(scores.class0 = hr, scores.class1 = no_hr, curve = T)
plot(pr) # AUC = .906

# create data frame 
plot_df <- data.frame(p=prec_list, r=rec_list, f=F_score_list)
ggplot(plot_df, aes(r, p)) + geom_point(size=3, color = "#454545") + 
  labs(title = "Figure 3. Precision-Recall Curve", x = "Recall", y = "Precision", subtitle = "AUC = .906; optimal threshold = .7") + theme_minimal() + 
  scale_y_continuous(breaks=seq(0,1,.1)) + scale_x_continuous(breaks=seq(0,1,.1)) + 
  geom_point(aes(x=r[opt_ind_bag], y=p[opt_ind_bag]), pch=18, size=8, color='#ff0000') +
  theme(plot.title = element_text(size=22), axis.title.x = element_text(size=15), axis.title.y = element_text(size=15),
        text = element_text(size=17))

# confusion matrix bagging normal random sampling
as.matrix(table(factor(test$hr_prob > thresholds[opt_ind_bag], levels=c(F, T)), test$home_run))
log_tbl = tibble(Predicted = c("Non HR", "HR"), "Non HR"=c(138844,153), HR=c(146,1071))
gt(log_tbl) %>% tab_spanner(label="Actual Outcome", columns=c("Non HR", "HR")) %>% 
  tab_header(title = "Table 3. Confusion Matrix", subtitle="Bagging with p=.7 Threshold") %>% cols_align(align="center") %>% 
  fmt_number(columns=2:3, use_seps = TRUE, decimals = 0)

# Repeat for Bagging with under sampling
# optimal threshold bagging
opt_ind_bag_under = which.max(F_score_list)

# confusion matrix at optimal threshold
as.matrix(table(factor(test$pred_under > thresholds[opt_ind_bag_under], levels=c(F, T)), test$home_run))

# gg plot Precision-Recall Curve Bagging undersampling
# used package to find AUC for PR curve
hr <- test$pred_under[test$home_run==1]
no_hr <- test$pred_under[test$home_run==0]
pr <- pr.curve(scores.class0 = hr, scores.class1 = no_hr, curve = T)
plot(pr) # AUC = .4665974

plot_df <- data.frame(p=prec_list, r=rec_list, f=F_score_list)
ggplot(plot_df, aes(r, p)) + geom_point(size=3, color = "#454545") + 
  labs(title = "Figure 2. Precision-Recall Curve", x = "Recall", y = "Precision", subtitle = "AUC = .467; optimal threshold = .4") + theme_minimal() + 
  scale_y_continuous(breaks=seq(0,1,.1)) + scale_x_continuous(breaks=seq(0,1,.1)) + 
  geom_point(aes(x=r[opt_ind_bag_under], y=p[opt_ind_bag_under]), pch=18, size=8, color='#ff0000') +
  theme(plot.title = element_text(size=22), axis.title.x = element_text(size=15), axis.title.y = element_text(size=15),
        text = element_text(size=17))

#### LOG LOSS ALL MODELS ###
# LR
LogLoss(log_model_data$hr_prob, log_model_data$home_run) # .0159

# bagging
LogLoss(test$hr_prob, test$home_run) # .007

# bagging under sampling
LogLoss(test$pred_under, test$home_run) # .045

# LogLoss table display
loss_tbl = tibble("Model" = c("Logistic Regression", "Bagging", "Bagging w/ Under sampling"), "LogLoss"=c(.0159, .007, .045))
gt(loss_tbl) %>%
  tab_header(title = "Table 4. Log Loss by Model") %>% cols_align(align="center") 

#### Get Live data ####

# Run expectancy matrix
re_mat <- matrix(c(.53,.94,1.17,1.43,1.55,1.8,2.04,2.32,.29,.56,.72,1,1,1.23,1.42,1.63,.11,.24,.33,.38,.46,.54,.6,.77), nrow=3, ncol=8, byrow=TRUE)
re_df <- data.frame(re_mat)
colnames(re_df) <- c("000", "100", "020", "003", "120", "103", "023", "123")
re_tbl <- tibble(Outs=c(0,1,2), re_df)
gt(re_tbl) %>% tab_header(title="Expected Runs by Game State") %>% tab_spanner(label="Baserunners", columns=2:9) %>% cols_align(align="center") %>%
  tab_source_note(source_note = "Source: Albert, J. (2015). Beyond runs expectancy. Journal of Sports Analytics, 1(1), 3-18.")


# REPEAT THIS CHUNK EVERY FEW SECONDS, TRACK EVERYTHING 
# access MLB game pitch-by-pitch data from baseballr package
game_date <- "2022-07-30"
game_packs <- baseballr::get_game_pks_mlb(date=game_date, level_ids = 1)
game_packs_live <- game_packs %>% filter(status.detailedState == "In Progress") %>% pull(game_pk)
payload <- purrr::map_df(.x = game_packs_live, ~baseballr::get_pbp_mlb(game_pk = .x))

# get pitches
pitch_events <- payload %>% filter(type=="pitch") %>% dplyr::select(about.inning, details.description, count.balls.start, count.strikes.start, count.outs.start,
                                                                    matchup.batter.fullName, matchup.pitcher.fullName, result.event, 
                                                                    matchup.batSide.code, matchup.pitchHand.code, away_team, home_team,
                                                                    matchup.postOnFirst.id,  matchup.postOnSecond.id,  matchup.postOnThird.id)

# filter for desired game
away_team = "Tigers"
pitch_events_game <- pitch_events %>% filter(away_team %in% away_team)

# get current data
current_inning <- pitch_events_game$about.inning[1]
current_balls <- pitch_events_game$count.balls.start[1]
current_strikes <- pitch_events_game$count.strikes.start[1]
current_outs <- pitch_events_game$count.outs.start[1]
current_batter <- pitch_events_game$matchup.batter.fullName[1]
current_pitcher <- pitch_events_game$matchup.pitcher.fullName[1]
current_bat_hand <- pitch_events_game$matchup.batSide.code[1]
current_pitch_hand <- pitch_events_game$matchup.pitchHand.code[1]
current_on_first <- pitch_events_game$matchup.postOnFirst.id[1]
current_on_second <- pitch_events_game$matchup.postOnSecond.id[1]
current_on_third <- pitch_events_game$matchup.postOnThird.id[1]

current_df <- data.frame(inn=current_inning, balls=current_balls, strikes=current_strikes, outs=current_outs, batter=current_batter,
                         pitcher=current_pitcher, stand=current_bat_hand, p_throws=current_pitch_hand, onFirst=current_on_first,
                         onSecond=current_on_second, onThird=current_on_third)
# runners on base char label
current_df <- current_df %>% dplyr::mutate(baserunners_label = case_when(onFirst>0 & onSecond>0 & onThird>0 ~ "123",
                                                       onFirst>0 & onSecond >0 ~ "120",
                                                       onFirst>0 & onThird>0 ~ "103",
                                                       onSecond>0 & onThird>0 ~ "023",
                                                       onFirst>0 ~ "100",
                                                       onSecond>0 ~ "020",
                                                       onThird>0 ~ "003", TRUE ~ "000"))
# Get run expectation
delta_run_exp <- re_df[current_df$baserunners_label][current_df$outs+1,]
current_df$delta_run_exp <- delta_run_exp

# Merge batter with stats
current_df2 <- left_join(current_df, stats2 %>% dplyr::select("Name", "HR_perc", "FB."), by=c("batter"="Name"))

# and pitcher
current_df3 <- left_join(current_df2, pitcher_stats %>% dplyr::select("Name", "HR.", "FB."), by=c("pitcher"="Name"))
current_df3$FB..x <- as.numeric(sub("%", "", current_df3$FB..x))/100
current_df3$count_label <- paste(current_df3$balls, current_df3$strikes, sep="_")

# test bagging on this new observation
current_pitch_data <- current_df3 %>% dplyr::select(stand, p_throws, outs, delta_run_exp, count_label, HR_perc, FB..x, HR., FB..y)
names(current_pitch_data)[names(current_pitch_data) == 'outs'] <- 'outs_when_up'
names(current_pitch_data)[names(current_pitch_data) == 'count_label'] <- 'count'
# replace invalid count labels
current_pitch_data <- current_pitch_data %>% dplyr::mutate(count = ifelse(count %in% unique(hr_data$count), count, "0_0"))

# Replace NAs with replacement level stats
current_pitch_data$HR_perc[is.na(current_pitch_data$HR_perc)]<-.034
current_pitch_data$FB..x[is.na(current_pitch_data$FB..x)]<-.26
current_pitch_data$FB..y[is.na(current_pitch_data$FB..y)]<-.257
current_pitch_data$HR.[is.na(current_pitch_data$HR.)]<-.031

# scrape live win expectancy from fangraphs.com
current_pitch_data$delta_home_win_exp <- 0
current_pitch_data$delta_run_exp <- 2.1

predict(hr_bag, newdata = current_pitch_data, type="prob")
