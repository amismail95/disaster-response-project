Installation
Python 3 should run the code with no problems. No libraries are necessary beyoed the Anaconda distribution of Python.

Project Motivation
To help label text messages in times of natural disaster to expedite the process of categorizing the messages.

Running Scripts

1. Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

Results

Running the web app will show the results of the work.

Licensing, Authors, Acknowledgments
I'd like to thank Udacity and FigureEight for providing the data.