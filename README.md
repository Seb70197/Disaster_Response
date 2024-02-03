__### Disaster Response Pipeline with Udacity__

Implementation of the Machine Learning Pipeline aiming to analyze entered text within a Web Application and showing a classification report.
The application aims to identify what kind of disaster or help request users might have.

The webapplication is showing at first as well 2 visualization 
- One for the Genre of message available within the data available within the training set
- One focusing on the Top 10 categories as well in use within the Dataset.

__How to use the Application__ :
- Within the text area, enter a text related to flood, weather or other categories
- click on classify message
- A result of the related categories will be displayed within the application

Enjoy playing around !!

__### Installation__

to run the application, the relevant python packages needs to be installed.
Download the application and start a terminal
Go to the app directory and run "python run.py"

If using the terminal, once the program run, start your webbrowser and call the page http://localhost:3000/
Best work under the Udacity Workspace IDE


__### Repository Content__

- __App__ :
    - templates : contains the html files necessary to run the web application
    - run.py : file necessary to run the application and displaying the first visualization of our trained data
- __Data__ :
    - disaster_categories.csv : the existing categories relevant for the classification model
    - disaster_messages.csv : the test data used to train the model. Contains the text/messages to categorized
    - process_data.py : process all activities around data load, cleaning and preparation for initiating the ML Model

- __models__ :
    - disasted_model.pkl : the train model used for our classification web application
    - train_classifier.py : the Machine Learning model built based on the data collected and cleaned

- __README.md__ : This file :-)
