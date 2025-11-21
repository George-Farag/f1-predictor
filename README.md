#Praedictionem

This project predicts a driver's finishing position in a Formula 1 race using historical race data from 1950-2024. 
Our model uses features such as qualifying results, grid position, team position, and driver history.

The app  is built with Streamlit, processes CSV data, and displays.
1. Predicted race order
2. Actual race order
3. Predicted vs Actual scatter plot (Random Forest Regression)


Dataset

We use official Formula 1 datasets downloaded by Kaggle (1950-2024):

1.24 races for 2024 reason

2.Drivers,contructions, results, qualifying, and lap time.

3.Potentially pit stops, i.e, still in progress of adding to our project.




How to use:

pip install -r requirements.txt

run to host the server- streamlit run  app.py





