# pandemic_response
Repository focused on solving the Pandemic Response challenge from XPrize with our TBSI team.

## Folders
1. The Predictor folder contain all the files used to run the SIR Particle Filter on top of the Oxford Data
2. The Xprize_Github_Files contains the challenge main explanation files for the basic templates

## Predictor
1. The file Models.py contains the library built using scientific articles, we will have the Particle Filter and SIR model implemented
2. The Predict_clean.ipynb jupyter file have the main code used to iterate between the methods in Models to reach the main results
3. The data is caught on the Oxford Dataset fountain , more information can be seen on the files database_connector_italy and database_connector_brazil.
4. After we catch the data from the Oxford Database , we save it as CSV on the data folder.
