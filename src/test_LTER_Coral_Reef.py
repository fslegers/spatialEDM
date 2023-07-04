from empirical_dynamic_modeling import *
import pandas as pd

if __name__ == "__main__":

    print("Under construction")

    # Package ID: knb-lter-mcr.7.34 Cataloging System:https://pasta.edirepository.org.
    # Data set title: MCR LTER: Coral Reef: Long-term Population and Community Dynamics: Other Benthic Invertebrates, ongoing since 2005.
    # Data set creator:    - Moorea Coral Reef LTER
    # Data set creator:  Robert Carpenter - Moorea Coral Reef LTER
    # Metadata Provider:    - Moorea Coral Reef LTER
    # Contact:    - Information Manager Moorea Coral Reef LTER  - mcrlter@msi.ucsb.edu
    # Stylesheet v1.0 for metadata conversion into program: John H. Porter, Univ. Virginia, jporter@virginia.edu
    #
    # This program creates numbered PANDA dataframes named dt1,dt2,dt3...,
    # one for each data table in the dataset. It also provides some basic
    # summaries of their contents. NumPy and Pandas modules need to be installed
    # for the program to run.

    infile1 = "https://pasta.lternet.edu/package/data/eml/knb-lter-mcr/7/34/668de777b44a8f656a19c3ca25b737c5".strip()
    infile1 = infile1.replace("https://", "http://")

    dt1 = pd.read_csv(infile1
                      , skiprows=1
                      , sep=","
                      , quotechar='"'
                      , names=[
                            "Year",
                            "Date",
                            "Location",
                            "Site",
                            "Habitat",
                            "Transect",
                            "Quadrat",
                            "Taxonomy",
                            "Count"]
                      , parse_dates=[
                            "Date", ]
                      )

    # Coerce the data into the types specified in the metadata
    dt1.Year = dt1.Year.astype('category')
    # Since date conversions are tricky, the coerced dates will go into a new column with _datetime appended
    # This new column is added to the dataframe but does not show up in automated summaries below.
    dt1 = dt1.assign(Date_datetime=pd.to_datetime(dt1.Date, errors='coerce'))
    dt1.Location = dt1.Location.astype('category')
    dt1.Site = dt1.Site.astype('category')
    dt1.Habitat = dt1.Habitat.astype('category')
    dt1.Transect = dt1.Transect.astype('category')
    dt1.Quadrat = dt1.Quadrat.astype('category')
    dt1.Taxonomy = dt1.Taxonomy.astype('category')
    dt1.Count = pd.to_numeric(dt1.Count, errors='coerce', downcast='integer')

    # Filter Echinometra Mathaei and Echinostrephus aciculatus)
    selected_species = "Echinometra mathaei"
    #selected_species = "Echinostrephus aciculatus"
    df_species = dt1[dt1['Taxonomy'] == selected_species]

    # Turn count data into relative abundance data
    for location in pd.unique(df_species.Location):
        dt_loc = df_species[df_species['Location'] == location]
        total_abundance = sum(dt_loc['Count'])
        if total_abundance > 0:
            df_species.loc[df_species['Location'] == location, 'Count'] = dt_loc['Count'].values/total_abundance
        else:
            print("Site " + str(location) + " is empty.")

    # TODO: this is where I left off
    multivariate_ts = []
    for site in pd.unique(dt1.Location):
        temp = dt1[(dt1.Taxonomy == selected_species) & (dt1.Location == site)]
        time_series = dt1[(dt1.Taxonomy == selected_species) & (dt1.Location == site)].Count.values
        try:
            multivariate_ts.append(time_series)
        except:
            print("Species " + str(selected_species) + " not found at " + str(site))
    print("Finished with species " + str(selected_species))










