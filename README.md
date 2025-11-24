# Car Price Prediction Using Machine Learning

## Project Overview
This project predicts used car prices from real-world advert data. I built it to demonstrate a complete and professional machine learning workflow that includes data cleaning, exploratory analysis, feature engineering, baseline modelling, advanced modelling and evaluation. Accurate price prediction supports valuation tools, dealership strategies and online marketplaces where consistent pricing and fair valuation are essential.

The dataset contains more than 400,000 vehicle listings with information such as mileage, make, model, body type, fuel type, registration year and condition. These attributes strongly influence vehicle value, and the goal of this project is to learn these pricing patterns using modern machine learning.

## Dataset Description
The dataset includes advert-level information such as mileage, registration code, standard colour, make, model, vehicle condition, year of registration, body type and fuel type. The price column is the target variable and represents the advertised price of each car.

Several fields contain missing values, especially year_of_registration and reg_code. Cleaning these values and constructing meaningful features is an important part of the workflow.

## Data Cleaning
I created advert_year based on the public_reference column and then derived vehicle_age as the time difference between advert_year and year_of_registration. Missing registration years were imputed using advert_year for new vehicles and the median year for each make–model group for used cars. I filled missing mileage values using body type medians and completed gaps in categorical fields such as standard_colour, fuel_type and body_type using grouped modes. I removed public_reference because it acts only as an identifier and does not contribute predictive value.

## Exploratory Analysis
I explored key numerical and categorical variables to understand their influence on car price. Price showed a strong right-skewed distribution, confirming the need for log transformation. Both mileage and vehicle_age displayed clear negative relationships with price, while body type and fuel type showed meaningful variations across segments. These patterns match real automotive market behaviour and help validate the quality of the cleaned dataset.

### Price Distribution Plot(Before Log Transform)
![pricedistribution]("images/price_distribution.png")

## Feature Engineering
I created a combined make_model feature to capture the effect of brand and model variations on price. I applied frequency encoding to this feature so that common variants receive higher values. The final feature set includes mileage, vehicle_age, year_of_registration, make_model_freq, body_type, fuel_type and vehicle_condition. I applied a log transformation to the target variable to stabilise variance and improve model performance.


## Baseline Model
Before selecting a final model, I trained a linear regression model as a baseline. Linear regression provides a simple, interpretable reference point and helps determine whether more complex models are justified. The baseline captured broad pricing trends but struggled with the non-linear structure of the data, especially interactions between mileage, age and the combined make–model feature. This resulted in higher error and lower explanatory power compared to more advanced methods.

## Final Model: XGBoost Regressor
I selected XGBoost as the final model because it handles non-linear relationships, works well on structured datasets and captures complex interactions between features. I trained XGBoost through a preprocessing pipeline that standardised numeric variables and one-hot encoded categorical features. The model achieved a significantly lower mean absolute error and a higher R² score than the baseline, confirming that the additional complexity is justified.

## Model Comparison
Both models were evaluated on the same test set using MAE and R². Linear regression produced higher errors and explained less variance, while XGBoost delivered strong, stable performance. The comparison verifies that XGBoost is the most appropriate model for this problem and that the final model choice is based on evidence rather than assumption.

## Feature Importance
I extracted feature importance from the trained XGBoost model. The most influential predictors were vehicle_age, mileage and the frequency-encoded make_model variable. These results align with real-world car valuation practices, where age, mileage and brand-model combinations significantly affect price. Body type and fuel type also contributed meaningfully, reinforcing the model’s interpretability and domain alignment.

### Feature Importance Plot (Top 15 Features)
![featureImp]('images/feature.png')

## Business Value
This project demonstrates how machine learning can support pricing decisions for dealerships, valuation platforms and online marketplaces. A reliable pricing model improves consistency, helps identify underpriced or overpriced vehicles and strengthens customer trust. The workflow also shows how data science practices can convert raw advert data into meaningful predictive insights.

## Future Work
Enhance the model by tuning XGBoost hyperparameters using GridSearchCV or Bayesian optimisation, adding location data to capture geographic price variation and introducing seasonal patterns based on advert dates. Deploying the model as an API or interactive tool would enable real-time pricing for users. Evaluating performance over time would also help maintain reliability as market conditions change.
