This project was built by Veronica Latzinnik and Almog Elharar at:

latzinnik.veronica@gmail.com

alicranck@gmail.com


This project was done as part of our B.Sc studies at Tel-Aviv University,
Sagol school of neuroscience. It is a decision support system for ICU, that uses
XGBoost to predict mortality rates of patients based on different measured features
as well as clustering the patients, potentially aiding in choosing treatment.

The project structure is as follows:

  '/' contains 'app.py' script to deploy flask server, as well as all python files necessary 
  to serve requests from the client, and handle data on a regular basis.

  '/static' contains all front-end code i.e. all HTML templates, JavaScript files and static content

  '/backend' contains all data files - archives and daily recieved data from the hospital

  '/templates/index.html is the main template and the root of the web-app
