#gundeathflorida

setwd('C:/Users/pooji/Desktop/George Mason University/STAT 515/mid')
gun_death <- read.csv('Florida Dataset.csv')
head(gun_death)
library(ggplot2)

#linegraph
ggplot(data = gun_death, aes(x = Year, y = Number.of.deaths))+
  geom_line()+
  geom_point()+
  ylim(0,1000)+
  labs(
    title = "Gun Deaths in Florida",
    x = "Year",
    y = "Number of Deaths"
  )

#bargraph
ggplot(data = gun_death, aes(x = Year, y = Number.of.deaths))+
  geom_bar(stat = 'identity', fill = 'violetred2')+
  xlim(1990,2015)+
  ylim(0,1000)+
  labs(
    title = "Gun Deaths in Florida",
    x = "Year",
    y = "Number of Deaths"
  )

#avgmaxtemp
weather_data <- data.frame(
  City = c("London", "Geneva", "Bergerac", "Christchurch", "Wellington"),
  Jan = c(7, 5, 11, 23, 19),
  Feb = c(7, 7, 12, 23, 19.5),
  Mar = c(10, 11, 16, 21, 18.7),
  Apr = c(13, 15, 19, 19, 17),
  May = c(17, 19, 22, 15, 15),
  June = c(20, 22, 26, 13, 13),
  July = c(22, 24, 28, 14, 12.5),
  Aug = c(21, 23, 29, 14, 12.8),
  Sept = c(19, 20, 25, 15, 13.2),
  Oct = c(15, 15, 20, 18, 14.7),
  Nov = c(10, 10, 14, 19, 15.8),
  Dec = c(8, 8, 11, 22, 17.5)
)

#templinegraph
library(reshape2)
library(ggplot2)
data_long <- melt(weather_data, id.vars = "City")
ggplot(data_long, aes(x = variable, y = value, group = City, color = City)) +
  geom_line(linewidth = 1.5) +
  scale_y_continuous(breaks = seq(0, 30, by = 5))+
  labs(title = "Average Maximum Temperature Throughout the Year",
       x = "Month", 
       y = "Temperature (°C)",
       caption = "Source: http://www.worldclimateguide.co.uk/climateguides/"
  )

#populationof50states
library(micromapST)
setwd('C:/Users/pooji/Desktop/George Mason University/STAT 515/mid')
popusdata <- read.csv('popdataus.csv')

str(popusdata)

type = c('map','id', 'bar','bar')
lab1 = c(NA, NA,'Population in 2020','Population in 1920')
lab3 = c(NA, NA, 'Population in Hundred Thousands', 'Population in Hundred Thousands')
col1 = c(NA,NA, 'A2020', 'A1920')

panelDesc <- data.frame(type, lab1,lab3, col1)

fname = 'newfinal.pdf'
pdf(file =fname, width = 7.5, height = 10)

micromapST(popusdata,
           panelDesc,
           rowNamesCol = 'STATE',
           rowNames = 'full',
           sortVar = c('A2020','A1920'), 
           ascend = FALSE,
           title = 'Population of US in 1920 and 2020',
           ignoreNoMatches = TRUE)

dev.off()


