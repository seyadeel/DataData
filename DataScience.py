import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm

col_list = ["country", "total_vaccinations", "daily_vaccinations", "people_vaccinated_per_hundred", "vaccines"]

df = pd.read_csv("covid_19_vaccination.csv", usecols=col_list)

#**********************************************************************

group1_df = df.dropna(subset=['total_vaccinations'], axis=0).groupby(['country']).sum()

largest_df = group1_df.nlargest(20, 'total_vaccinations')

smallest_df = group1_df.nsmallest(20, 'total_vaccinations')

color = cm.viridis_r(np.linspace(.4, .8, 20))

largest_df.plot(y='total_vaccinations', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6))

plt.title('COVID-19 Vaccination By The Numbers')

smallest_df.plot(y='total_vaccinations', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6))

plt.title('COVID-19 Vaccination By The Numbers')

#**********************************************************************

group2_df = df.dropna(subset=['people_vaccinated_per_hundred'], axis=0).groupby(['country']).sum()

num = df.groupby(['country']).size()

group2_df['people_vaccinated_per_hundred'] = group2_df.apply(lambda x: x['people_vaccinated_per_hundred']/num, axis=1)

largest_df = group2_df.nlargest(20, 'people_vaccinated_per_hundred')

color = cm.inferno_r(np.linspace(.4, .8, 20))

largest_df.plot(y='people_vaccinated_per_hundred', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6))

plt.title('20 Countries with Highest Vaccination Numbers Per Hundred People')

group2_df.to_csv('per_hundred_vaccinations.csv')


#**********************************************************************
group3_df = df

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

replace_df = replace_df.dropna(subset=['total_vaccinations'], axis=0).groupby(['vaccines']).sum().sort_values(by=['total_vaccinations'], ascending=False)

color = cm.cividis_r(np.linspace(.4, .8, 20))

replace_df.plot(y='total_vaccinations', kind='bar', stacked=True, color=color, legend=True, figsize=(8, 6))

plt.title('Countries with Different Types of Vaccines')
