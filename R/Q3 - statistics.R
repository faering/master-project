
#load DescTools library (for Dunnett's test)
library(DescTools)

path <- "C:\\Users\\matti\\OneDrive\\Education\\SDC\\MasterThesis\\master-project\\results\\Research Question 3\\Time-frequency Power Analysis - strict labelling\\hippocampus-"
hp_subfield <- "body & tail"
stimulus <- "Picture Placed"
filename <- paste0(path, hp_subfield, "/data/", "3.2 ", stimulus, "_window_subs_clusters.csv")

# load data
data <- read.csv(filename)

boxplot(Power ~ Condition,
        data = data,
        main = "Power In Condition",
        xlab = "Condition",
        ylab = "Average Power",
        col = "darkorange",
        border = "black")

# one-way ANOVA
aov <- aov(Power~factor(Condition), data=data)
summary(aov)

# Tukey's post-hoc test
TukeyHSD(aov, conf.level=0.95)
plot(TukeyHSD(aov, conf.level=.95), las = 2)

# Dunnett's post-hoc test
DunnettTest(data$Power, data$Condition, control=1, method="one.sided", alpha=0.05)
plot(DunnettTest(data$Power, data$Condition, control=1, method="one.sided", alpha=0.05), las = 2)
