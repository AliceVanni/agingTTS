library(glue)

# Loading the data
model_name <- 'gt'
filename <- glue('D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/pitch/mean_pitch_{model_name}_model.txt')
data <- read.table(filename, sep='\t', header = 1)

head(data, 5)

# Boxplots
boxplot(Mean_Pitch ~ Category, data=data)
title('Distribution of Mean Pitch - Ground Truth')
boxplot(Max_Pitch ~ Category, data=data)
title('Distribution of Maximum Pitch - Ground Truth')
boxplot(Min_Pitch ~ Category, data=data)
title('Distribution of Minimum Pitch - Ground Truth')
boxplot(Range ~ Category, data=data)
title('Distribution of Pitch Range - Ground Truth')
boxplot(Variance_Pitch ~ Category, data=data)
title('Distribution of Variance of Pitch - Ground Truth')

# T-test
t.test(Mean_Pitch ~ Category, data=data)
t.test(Range ~ Category, data=data)
t.test(Variance_Pitch ~ Category, data=data)
t.test(Max_Pitch ~ Category, data=data)
t.test(Min_Pitch ~ Category, data=data)

# ANOVA
summary(aov(Mean_Pitch ~ Category, data=data))
summary(aov(Range ~ Category, data=data))
summary(aov(Variance_Pitch ~ Category, data=data))
summary(aov(Max_Pitch ~ Category, data=data))
summary(aov(Min_Pitch ~ Category, data=data))

p_values <- c(0.00689, 0.374, 0.569, 0.765, 0.201)

p.adjust(p_values, method = "holm")

# Speech Rate analysis
model_name <- 'gt_R'
filename <- glue('D:/Alice/Desktop/UNIs/VT-RUG_2023/THESIS/agingTTS/acoustic_analysis/speaking_rate/SyllableNuclei_{model_name}.txt')
data <- read.table(filename, sep=',', header = 1)

head(data, 5)

# Boxplot
boxplot(speechrate_nsyll_dur ~ category, data=data)
title('Speech Rate distribution - Ground Truth')

# ANOVA
summary(aov(speechrate_nsyll_dur ~ category, data=data))

