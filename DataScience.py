#important python libraries to handle the datasets and plots
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm

#usable columns from the dataset whcih is to be read
col_list = ["country", "total_vaccinations","people_vaccinated_per_hundred", "vaccines"]

#this file is from https://www.kaggle.com/gpreda/covid-world-vaccination-progress under the name country_vaccinations.csv
df = pd.read_csv("covid_19_vaccination.csv", usecols=col_list)

#reads the first 100 rows of the dataset
df.head(100)

def drop_NaN():
    
    '''
    This line groups the dataset by country after it has removed 
    the NaN values from the "total_vaccination" column and sums 
    the total values. The reason droping the NaN values was chosen 
    over imputing is because the missing values are as a result of
    dates where vaccinations did happen, therefore imputing would 
    have added non existent vaccination numbers giving incorrect 
    information.'''
    
    return None

group1_df = df.dropna(subset=['total_vaccinations'], axis=0).groupby(['country']).sum()

#This counts the the duplicated country names
num_1 = df.groupby(['country']).size() 

#vairable assignment
group2_df = group1_df

#the sum of "total_vaccination" values divided by the number of duplicated names
group2_df['people_vaccinated_per_hundred'] = group2_df.apply(lambda x: x['people_vaccinated_per_hundred']/num_1, axis=1) 

##reads the first 10 rows of the dataset
group2_df.head(10)

#finds the largest twenty sums of duplicated names
largest_df = group1_df.nlargest(20, 'total_vaccinations') 

#finds the smalles twenty sums of duplicated names
smallest_df = group1_df.nsmallest(20, 'total_vaccinations') 

#color selection
color = cm.viridis_r(np.linspace(.4, .8, 20)) 

def total_vac(x):
    
    '''
    This function is to plot those which use "total_vaccinations" column
    '''
    
    plot = x.plot(y='total_vaccinations', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6)) 

    return plot

#plots the twenty countries with higest number of vaccination
total_vac(largest_df)

#plot title
plt.title('20 Countries with Highest Total Vaccination Numbers')

#plots the twenty countries with lowest number of vaccination
total_vac(smallest_df)

#plot title
plt.title('20 Countries with Lowest Total Vaccination Numbers')

#finding the twenty higest number of vaccination per hundred people
largest_df = group2_df.nlargest(20, 'people_vaccinated_per_hundred')

#color selection
color = cm.inferno_r(np.linspace(.4, .8, 20))

def percent_vac(x):
    
    '''
    This function is to plot those which use "people_vaccinated_per_hundred" column
    '''
    
    plot = x.plot(y='people_vaccinated_per_hundred', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6)) 

    return plot

#plots the twenty countries with higest number of vaccination per hundred people
percent_vac(largest_df)

#plot title
plt.title('20 Countries with Highest Vaccination Numbers Per Hundred People')

#assigning variable to a different name
group3_df = df

#replacing vaccine names to more suitable ones
replace_df = group3_df.replace(regex="Pfizer/BioNTech", value="Pfizer", inplace=True)
replace_df = group3_df.replace(regex="Oxford/AstraZeneca", value="AstraZeneca", inplace=True)
replace_df = group3_df.replace(regex="Sinopharm/Beijing", value="SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Sinopharm/Beijing", value="SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Moderna, Oxford/AstraZeneca, Pfizer/BioNTech", value="Moderna, AstraZeneca, Pfizer", inplace=True)
replace_df = group3_df.replace(regex="Pfizer/BioNTech, Sinopharm/Beijing, Sputnik V", value="Pfizer, Sinopharm, Sputnik V", inplace=True)
replace_df = group3_df.replace(regex="Oxford/AstraZeneca, Sinovac", value="AstraZeneca, SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Moderna, Pfizer/BioNTech", value="Moderna, Pfizer", inplace=True)
replace_df = group3_df.replace(regex="Sinopharm/Beijing, Sinopharm/Wuhan, Sinovac", value="SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Pfizer/BioNTech, Sinovac", value="Pfizer, SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Oxford/AstraZeneca, Pfizer/BioNTech", value="AstraZeneca, Pfizer", inplace=True)
replace_df = group3_df.replace(regex="Oxford/AstraZeneca, Sinopharm/Beijing", value="AstraZeneca, SinoVac", inplace=True)
replace_df = group3_df.replace(regex="Sinovac", value="SinoVac", inplace=True)
replace_df = group3_df.replace(regex="SinoVac, SinoVac", value="SinoVac", inplace=True)
replace_df = group3_df.replace(regex="", value="")

#removing null values from "total_vaccination" column
replace_df = replace_df.dropna(subset=['total_vaccinations'], axis=0)

#groups the "vaccines" column by types of vaccines and counts the value removing null values
num_2 = replace_df.groupby(['vaccines']).size()

#divides the number found in num_2 by the total number summed per coutry names in "people_vaccinated_per_hundred" column
replace_df['people_vaccinated_per_hundred'] = replace_df.apply(lambda x: x['people_vaccinated_per_hundred']/num_2, axis=1)

#groups the "vaccines" column by type and sums the number and sorts in descending order
replace_df = replace_df.groupby(['vaccines']).sum().sort_values(by=['total_vaccinations'], ascending=False)

#shows the first 100 rows of the filtered dataset
replace_df.head(100)

#color selection
color = cm.cividis_r(np.linspace(.4, .8, 20))

#plots the total vacciantion number by type of vaccines used
total_vac(replace_df)

#plot title
plt.title('Total Number of Vaccination by Type of Vaccines')

#grrops by type of vaccines and sorts by the sum of people vaccinated per hundred in descending order
vaccines_df = replace_df.groupby(['vaccines']).sum().sort_values(by=['people_vaccinated_per_hundred'], ascending=False)

#color selection
color = cm.magma_r(np.linspace(.4, .8, 20))

#plots the percentage of people vaccinated per type of vaccine
percent_vac(vaccines_df)

#plot title
plt.title('Highest Number in Type of Vaccines Used Per One Hundred People')

#groups first by vaccine type then by country and sortes the number of vaccinated people per hundred in descending order
group4_df = group3_df.groupby(["vaccines", "country"])["people_vaccinated_per_hundred"].size().sort_values(ascending=False)

#prints the result of the dataset
print(group4_df.head)

#locates the the AstraZeneca, Pfizer row from the vaccien types and assigns it to a variable
vaccines_df = group4_df.loc["AstraZeneca, Pfizer"]

#color selection
color = cm.prism_r(np.linspace(.4, .8, 20))

#plots the highest percentage of people per country and vaccine type 
percent_vac(vaccines_df)

#plot title
plt.title('Countries which Got (AstraZeneca, Pfizer) Vaccines per Hundred People')

