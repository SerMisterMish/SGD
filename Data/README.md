# Data from: Trends in plant cover derived from vegetation-plot data  using ordinal zero-augmented beta regression

All data files with the extension '.csv' are colon-separated text files.

Visits table: one row per visit

* Site_id:          key variable identifying location
* Visit_id:         key variable identifying a location-date combination
* Period:           four-year visiting period

Observations table: one row per observation (species seen at a specific time at a specific place)

* Site_id:          key variable identifying location
* Visit_id:         key variable identifying a location-date combination
* Species_id:       key variable identifying species
* Cover:            plant cover percentage (calculated as described in the publication)
* Cover_class:      plant cover class (as documented in the field)

Species table: one row per species

* Species_id:       key variable identifying species
* Scientific_name:  scientific name

