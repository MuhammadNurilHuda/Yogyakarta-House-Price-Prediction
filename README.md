# THIS IS A PROJECT ON DICODING MACHINE LEARNING EXPERT IN COLLABORATION WITH DBS FOUNDATION

# Project Domain

## Background of the Project

Yogyakarta, a region known for its cultural heritage and robust educational institutions, has experienced a significant increase in housing prices over the past decade. This price escalation has been driven by a combination of factors, including rising demand from students, investors, and a growing middle class. As Yogyakarta's reputation as a tourist and academic hub continues to flourish, property prices have surged to levels that many local residents find unaffordable. The high cost of housing starkly contrasts with the economic realities of Yogyakarta's population, where average household income and purchasing power remain relatively low compared to cities like Jakarta or Surabaya. This disparity has created a housing affordability crisis that has far-reaching implications on social and economic stability, particularly affecting low- to middle-income families.

## Problem Statement and Its Importance

Addressing the issue of housing affordability in Yogyakarta is crucial because housing serves as a fundamental human need and economic good. The high housing prices exacerbate socio-economic inequality, as lower-income households struggle to secure stable and adequate housing. Furthermore, the inability to afford homes can lead to long-term consequences, such as an increase in informal housing settlements, diminished quality of life, and social disparities. Additionally, the housing market's inaccessibility poses challenges for young families, first-time homebuyers, and students who may resort to high rental expenses, reducing their ability to invest in other essential needs.

From a policy and planning perspective, solving the housing affordability crisis requires a robust, data-driven approach. Machine learning models can play a crucial role by providing accurate and efficient predictions of housing prices, which in turn can inform urban planning, property taxation, and real estate investment decisions. A predictive model that accurately reflects market dynamics can assist government agencies, real estate developers, and potential buyers in making informed choices. The use of machine learning ensures that the analysis remains systematic and adaptive to changes in economic factors, providing a scalable solution to this pressing problem.

## Supporting Research and References

1. Fisher, J.D., & Gatzlaff, D.H. (2019). Real Estate Prices and Economic Fluctuations. Journal of Economic Perspectives. This research highlights how real estate markets are influenced by economic fluctuations, which is relevant to understanding housing prices in developing regions like Yogyakarta. The study also discusses the importance of using predictive modeling to forecast real estate trends.

2. Setiawan, B., Rahmawati, F., & Yuniarti, E. (2020). Housing Affordability in Indonesia: Case Study of Yogyakarta. Indonesian Journal of Urban and Regional Development. This paper provides a detailed analysis of housing affordability in Yogyakarta, highlighting the socio-economic challenges faced by residents. The authors emphasize the growing disparity between housing prices and income levels, underscoring the need for effective data-driven strategies.

3. Nguyen, T.V., & Cripps, A. (2021). Machine Learning Approaches for Predicting Real Estate Prices. International Journal of Data Science and Analysis. This research demonstrates the effectiveness of various machine learning algorithms, such as linear regression, decision trees, and deep learning, in predicting property prices. It provides a theoretical framework for applying machine learning to real estate markets.

4. Yuliawati, L., & Hartono, D. (2018). The Economic Impact of Tourism on Housing Prices in Yogyakarta. Journal of Tourism Economics. The study examines how tourism growth has directly influenced housing prices in Yogyakarta, offering insights into external factors that impact real estate trends.

These references provide a robust foundation for understanding the economic and social challenges of Yogyakarta's housing market and highlight the potential of machine learning models in addressing these issues. By leveraging empirical research and systematic analysis, the project aims to deliver a comprehensive and actionable solution to the housing affordability crisis.

# Business Understanding

## Problem Statements

The real estate market in Yogyakarta is characterized by rapid increases in property prices, outpacing the growth of the local population's income levels. This situation has led to significant housing affordability challenges, affecting a large segment of residents. The main problem lies in the unpredictability and complexity of housing prices, which are influenced by a multitude of factors such as location, proximity to amenities, and economic conditions. This unpredictability creates difficulties for stakeholders, including government planners, real estate investors, and potential homeowners, in making informed decisions.

To solve this problem, it is critical to develop a predictive model that can provide accurate estimates of housing prices in Yogyakarta. An accurate prediction model will assist stakeholders in understanding market trends, optimizing investment strategies, and implementing effective housing policies.

## Goals

The primary goal of this project is to create a robust and accurate model that can predict housing prices in Yogyakarta using relevant features from historical data. Specifically, the objectives include:

Building predictive models that can forecast housing prices with high accuracy.
Comparing the performance of different machine learning algorithms, including both linear models and tree-based models.
Reducing the Mean Squared Error (MSE) of the predictions, ensuring that the models are as precise as possible.
Providing actionable insights derived from the models to aid policymakers and stakeholders in the real estate market.

## Solution Statement

To achieve the project goals, we propose a multi-model approach that leverages both linear and tree-based algorithms. The following are the detailed solution strategies:

1.  **Solution 1**: Building Baseline Models with Linear Algorithms
    We will begin by implementing two linear models: Linear Regression and ElasticNet. These models are straightforward, interpretable, and effective for understanding the underlying trends in the data.

    - **Linear Regression**: A simple model that assumes a linear relationship between the features and the target variable. It will serve as a baseline for comparing more complex models.

    - **ElasticNet**: A regularization model that combines the penalties of both L1 (Lasso) and L2 (Ridge) regression. This model is expected to perform better in cases where features are highly correlated or when there are many features with small or moderate effect sizes.

    - **Evaluation**: The performance of these models will be measured using Mean Squared Error (MSE) to assess how well the predicted prices match the actual prices.

2.  **Solution 2**: Applying Tree-Based Models for Improved Performance
    Tree-based models, such as Decision Trees and Random Forest, will be implemented to capture non-linear relationships and interactions between features that linear models may miss.

    - **Decision Tree**: A simple tree-based model that partitions the data based on feature values. While easy to interpret, it may overfit the data, so further improvements will be explored.
    - **Random Forest**: An ensemble method that builds multiple Decision Trees and averages their predictions to reduce overfitting and improve generalization. This model is expected to perform better than a single Decision Tree, especially in terms of handling complex data structures.
    - **Evaluation**: The performance will also be evaluated using Mean Squared Error (MSE), allowing us to compare the effectiveness of linear and tree-based models systematically.

3.  **Solution 3**: Model Improvement through Hyperparameter Tuning on Best Performing Models. Once the initial models have been trained and evaluated, we will identify the best-performing model from each segment—linear and tree-based—based on the lowest Mean Squared Error (MSE). We will then apply hyperparameter tuning to these selected models to further enhance their performance.

    - **For Linear Models**: Hyperparameter Tuning on ElasticNet
      If ElasticNet outperforms Linear Regression in terms of lower MSE, we will proceed with hyperparameter tuning for ElasticNet. The tuning process will involve:

      - **Adjusting the Alpha Parameter**: This controls the overall strength of regularization.
      - **Optimizing the L1 Ratio**: This parameter balances L1 (Lasso) and L2 (Ridge) regularization to achieve the best performance.
      - Evaluation: We will measure the reduction in MSE after tuning to ensure the model captures the data's underlying trends more effectively.

    - **For Tree-Based Models**: Hyperparameter Tuning on Random Forest
      If Random Forest outperforms Decision Tree, we will focus on fine-tuning the Random Forest model. The tuning process will include:

      - **Number of Trees (n_estimators)**: Determining the optimal number of trees to improve model stability and accuracy.
      - **Maximum Depth**: Controlling the depth of the trees to prevent overfitting.
      - **Minimum Samples per Split**: Ensuring each split has enough samples to make meaningful divisions.
      - **Evaluation**: We will track the reduction in MSE to confirm the effectiveness of the hyperparameter adjustments and ensure the model generalizes well on unseen data.

**Evaluation Metrics**

The final evaluation of these tuned models will be based on Mean Squared Error (MSE), which provides a clear and measurable indicator of prediction accuracy. By fine-tuning only the best-performing models from each segment, we aim to achieve an optimal balance between model complexity and predictive performance, ensuring that our solution is both efficient and impactful.

# Data Understanding

## Data Overview

The dataset used for this project is sourced from Kaggle, specifically from the “Yogyakarta Housing Price” dataset, which can be downloaded from [this link](https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia/data). The dataset consists of 2,020 rows and 9 columns, representing property listings in the Yogyakarta area. The data includes various features related to real estate, which are essential for analyzing and predicting property prices.

**Data Characteristics**

**The sample of the data**

| price          | nav-link                                                      | description                                                   | listing-location  | bed | bath | carport | surface_area | building_area |
| -------------- | ------------------------------------------------------------- | ------------------------------------------------------------- | ----------------- | --- | ---- | ------- | ------------ | ------------- |
| Rp 1,79 Miliar | [Link](https://www.rumah123.com/properti/sleman/hos17166670/) | Rumah 2 Lantai Baru di jalan Palagan Sleman Yogyakarta        | Ngaglik, Sleman   | 3   | 3    | 2       | 120 m²       | 110 m²        |
| Rp 170 Juta    | [Link](https://www.rumah123.com/properti/sleman/hos17166674/) | RUMAH BARU DEKAT AL AZHAR DAN UGM                             | Jombor, Sleman    | 3   | 2    | 1       | 102 m²       | 126 m²        |
| Rp 695 Juta    | [Link](https://www.rumah123.com/properti/sleman/hos17166621/) | RUMAH ASRI DAN SEJUK DI BERBAH SLEMAN DEKAT PASAR WISATA ALAM | Berbah, Sleman    | 2   | 2    | 1       | 100 m²       | 100 m²        |
| Rp 560 Juta    | [Link](https://www.rumah123.com/properti/sleman/hos17166607/) | Rumah Murah 5 Menit Dari Candi Prambanan Tersisa 1 Unit       | Prambanan, Sleman | 3   | 1    | 1       | 109 m²       | 67 m²         |
| Rp 200 Juta    | [Link](https://www.rumah123.com/properti/sleman/hos17166601/) | Rumah Murah Cicilan 1jt Di Moyudan Sleman                     | Moyudan, Sleman   | 2   | 1    | 1       | 60 m²        | 30 m²         |

The dataset, as provided, is scraped from the real estate website 'rumah123.com' using a Chrome extension called Instant Data Scraper. The scraped data covers listings sorted as "Newest" and has been collected over two months. The data is unfiltered and uncleaned, containing duplicates and some missing values.

Here is an overview of the data and its structure:

- Number of Rows (Observations): 2,020
- Number of Columns (Features): 9
- Data Conditions:
  - The dataset contains duplicates and null values.
  - Several columns are in an unprocessed state, requiring data cleaning and type conversion.
- Data Features
  The following is a breakdown of all the features available in the dataset:

  - **price** (object): The price of the property in Indonesian Rupiah. This column will need to be converted to a numerical data type for analysis. It contains missing values (0).
  - **nav-link** (object): The URL link to the property listing. This column contains URLs pointing to external sources or detailed property descriptions. No missing values (0).
  - **description** (object): A textual description of the property, detailing its features, location, and amenities. This column contains no missing values.
  - **listing-location** (object): The location where the property is listed, which can include the city or specific area. Missing values: 0.
  - **bed** (float64): The number of bedrooms in the property. It is a continuous variable but contains 19 missing values that need to be handled.
  - **bath** (float64): The number of bathrooms in the property. This feature also has 21 missing values, requiring attention during data cleaning.
  - **carport** (float64): The number of parking spaces (carports) available at the property. This column has 307 missing values.
  - **surface_area** (object): The surface area of the property (likely in square meters). This column may require conversion to a numerical data type and contains 1 missing value.
  - **building_area** (object): The building area of the property, which may also need conversion to a numerical type. It contains 1 missing value.

- Missing Values Summary:
  - **price**: 0 missing values
  - **nav-link**: 0 missing values
  - **description**: No missing values
  - **listing-location**: 0 missing values
  - **bed**: 19 missing values
  - **bath**: 21 missing values
  - **carport**: 307 missing values
  - **surface_area**: 1 missing value
  - **building_area**: 1 missing value

Before performing any analysis, the missing values in the numerical columns (bed, bath, carport, surface_area, building_area) will need to be handled, either through imputation or removal depending on the overall strategy for data cleaning.

# Data Preparation

In the initial stages of data preparation, two columns, nav-link and description, were removed from the dataset. These columns were deemed unnecessary for the analysis because:

- The nav-link contains URLs, which do not provide any meaningful information for the model's prediction tasks.
- The description contains free text, which would require additional processing (like text mining) to be useful, and it was considered beyond the scope of this particular analysis.

After removing these columns, the next steps focused on cleaning and transforming the remaining data to ensure it was ready for analysis and modeling.

1.  **Handling Missing Values**

    - One of the first steps was handling missing values in the dataset. For columns such as bed, bath, surface_area, and building_area, rows with missing values were dropped using dropna(). These columns were deemed crucial for the analysis, and missing data in these columns could significantly impact the model's performance. Hence, removing rows with missing values ensures data consistency and reliability.
    - For the carport column, it was assumed that missing values indicate properties with no carport, and therefore, these missing values were filled with 0. This approach is based on the assumption that if a house has no carport, it is still a valid record, just like how it would be for bed and bath (it is impossible for a house to have zero bedrooms or bathrooms, but a house can have zero carports).

2.  **Preprocessing listing-location**

    - The listing-location column had issues with extra spaces in some entries, such as "Imogiri , Bantul". These extra spaces could cause inconsistencies, making it difficult to analyze locations accurately. The extra spaces were removed to standardize the format and ensure that locations are consistently represented.

3.  **Dropping Duplicates**

    - The next step involved removing duplicate rows using drop_duplicates(). Duplicate rows can distort analysis and affect model training by overrepresenting certain entries. Removing duplicates ensures that each record is unique and contributes properly to the analysis.

4.  **Preprocessing the price Column**

    - The price column originally contained values like "Rp 1,79 Miliar", which were in text format. To perform numerical analysis, this column was transformed into a numeric type by removing the currency symbols ("Rp") and converting the values into numeric format. This preprocessing step was necessary to allow for meaningful analysis and calculations, such as price comparisons or price-based predictions.

5.  **Preprocessing bed, bath, and carport Columns**

    - The bed, bath, and carport columns, initially in float64 data type, were converted into integer values. These columns represent counts of bedrooms, bathrooms, and carports, and thus, using integers is more appropriate. This conversion ensures that the data type is aligned with the nature of the values, facilitating better model performance.

6.  **Preprocessing surface_area and building_area Columns**

    - The surface_area and building_area columns contained values with the unit "m²" (square meters). These values were converted to integers by removing the "m²" unit. This transformation is necessary for the columns to be treated as numeric values, enabling further analysis and calculations to be performed on these variables.

7.  **One-Hot Encoding (OHE) for Linear Models**

    - As part of the data preparation, One-Hot Encoding (OHE) was applied to the listing-location column, which contains categorical values. OHE is a technique used to convert categorical data into a numerical format that can be easily processed by machine learning algorithms, especially linear models.

      After applying One-Hot Encoding to the listing-location column, the dataset's number of columns increased significantly, from the original number of columns to 74 columns. This happened because each unique value in the listing-location column was transformed into a new binary (0 or 1) column. Each of these new columns corresponds to a specific location and indicates whether a given record belongs to that location. For example, if there were five unique locations in the dataset, OHE would generate five new columns: one for each location, where a value of 1 indicates that the record belongs to that location, and 0 otherwise.

      **Why One-Hot Encoding is suitable for Linear Models?**

      Linear Models Require Numeric Inputs: Linear models work by fitting a linear equation between the input features and the target variable. Categorical variables cannot be directly processed in this way, so they need to be converted into a format that the model can handle. One-Hot Encoding turns categorical variables into binary values (0 or 1), making them compatible with linear models.

      Maintains Independence of Categorical Values: One-Hot Encoding ensures that each category is represented independently of others. For instance, when the location "Sleman" is encoded, it receives a binary value (1 or 0), and similarly for other locations. This encoding method prevents the model from misinterpreting the categorical values as ordinal or numeric, which could lead to incorrect assumptions about the relationship between different categories.

      Prevents Model from Misinterpreting Ordinality: Categorical variables may not have an intrinsic ordering (e.g., "Sleman" is not greater or lesser than "Yogyakarta"). Directly assigning numerical values to these categories would imply an ordinal relationship, which could mislead the linear model. OHE avoids this by creating separate columns for each category, treating each one independently without any notion of ordinality.

      **The example result of OHE/dummies**
      ![dummies encoding](assets\dummies_encoding.png)

8.  **Target Encoding for Tree-Based Models**

    - For tree-based models (Decision Tree and Random Forest), Target Encoding was applied to listing-location. Target Encoding replaces categories with the average target value for each category. This technique is particularly useful for tree-based models because it captures the relationship between categorical features and the target variable, enhancing the model's predictive power. Tree-based models are capable of handling categorical variables directly, but Target Encoding helps to refine the model by incorporating more information from the categories.

      **The example result of target encoding**
      ![target encoding](assets\target_encoding.png)

## Exploratory Data Analysis

This initial step lays a solid foundation for building a predictive model by identifying relevant features and understanding the data’s overall structure

![numeric col desc](assets\numeric_col_desc.png)

The table above summarizes key descriptive statistics for the dataset used in the project. Each column represents an important feature related to house properties, including price, number of bedrooms (bed), number of bathrooms (bath), carport spaces, surface area, and building area.

- Price: The average house price is approximately 1.92 billion IDR, with a substantial variation, as indicated by a high standard deviation. The price ranges from a minimum of 7 million IDR to a maximum of 42.1 billion IDR, suggesting a wide range in housing options and values.
- Bedrooms (bed): Houses typically have around 4 bedrooms on average, though this varies widely, with some houses offering up to 49 bedrooms.
- Bathrooms (bath): The average number of bathrooms is roughly 3, with a similar variation as bedrooms, indicating diverse property sizes and amenities.
- Carport: Most properties provide around 1 to 2 carport spaces, with a maximum of 15, catering to different parking needs.
- Surface Area and Building Area: The average surface area is about 196.5 square meters, and building areas average around 1.54 million square meters, although the latter shows extreme values, possibly due to outliers or data entry errors.

These statistics provide an initial understanding of the range and distribution of property characteristics in the dataset, which will inform feature engineering and model selection for predicting house prices in Yogyakarta.

![heatmap](assets\heatmap.png)

The heatmap above highlights the correlation between various features in the dataset and their relationship with house prices. Based on this analysis, we can make informed decisions on feature selection to optimize the predictive model:

1. Surface Area: This feature exhibits a strong correlation with price (0.63), indicating that properties with larger surface areas tend to be priced higher. Given this relationship, surface area will be retained as a primary predictor in the model.

2. Bedrooms and Bathrooms: Both bed and bath show moderate correlations with price (0.46 each) and a high correlation with each other (0.93), indicating potential multicollinearity. To avoid redundancy, I considered choosing one as a representative feature; however, I will retain both, as they may individually capture distinct aspects of property size and amenities that could influence pricing.

3. Carport: Although the correlation between carport and price is relatively low (0.28), this feature will be included in the model. Carport availability may still play a role in pricing, especially if buyers in this region value parking space.

4. Building Area: The correlation between building_area and price is extremely close to zero (0.0045), suggesting minimal influence on property prices. Consequently, this feature will be removed from the model to streamline the dataset and reduce noise.

By selectively retaining features with strong or contextually relevant correlations, the model can maintain focus on the most impactful predictors, enhancing its accuracy and interpretability.

![boxplot](assets\boxplot.png)

The boxplots for features such as price, bed, bath, carport, and surface_area reveal a significant number of outliers. These outliers indicate extreme values that deviate from the central distribution of the data, suggesting a wide variability in the property characteristics within the dataset.

Such high variability can negatively impact model performance, as extreme values can bias the model and make it less accurate in predicting standard cases. To address this issue, normalization is essential. By scaling the features to a consistent range, we can mitigate the influence of outliers and improve the model's ability to generalize across different data points.

In this case, we will use Standard Scaler for normalization. Standard Scaler transforms the data by removing the mean and scaling to unit variance. This method uses the mean and standard deviation of each feature to rescale the data, effectively centering each feature around zero (mean of 0) and scaling with a standard deviation of 1. Mathematically, the transformation for each feature
𝑋 is defined as:

<p align="center">
  <img src="assets\standard_scaler.png" alt="Alt Text">
</p>

This approach helps reduce the impact of extreme outliers by transforming the feature distribution to a more normalized scale, making it more suitable for predictive modeling. Normalization with Standard Scaler will enhance the model's accuracy and stability, ensuring that each feature contributes equally to the predictions.

![barplot](assets\barplot.png)

The bar plot above compares the top 5 most expensive and the top 5 cheapest locations for house prices in Yogyakarta:

- Top 5 Most Expensive Locations: Pakualaman in Yogyakarta has the highest average house price, significantly outpacing other areas with an average price around 21 billion IDR. Other high-cost areas include Demangan, Sekip, Caturtunggal in Sleman, and Seturan, with prices ranging from approximately 5 to 8.3 billion IDR.

- Top 5 Cheapest Locations: On the other end, Wonosari in Gunungkidul and Kulonprogo offer some of the most affordable average house prices, around 300 million IDR. Moyudan in Sleman, Sentolo in Kulonprogo, and Bambanglipuro in Bantul also fall into the lower price range, with averages below 400 million IDR.

This distribution highlights a stark contrast between luxury locations and more affordable areas in Yogyakarta, useful for identifying suitable areas for different budget ranges.

![barplot](assets\histplot.png)

The histograms above illustrate the distribution of several key features in the house price dataset: price, bed, bath, carport, and surface_area. Each histogram displays a right-skewed distribution, with most data concentrated toward the lower values and a long tail extending to the right.

1. Price: The distribution of house prices is heavily skewed to the right, with most properties priced at lower ranges and a few extremely high-priced properties, creating a long tail. This skewness indicates a need for normalization to reduce the influence of these high-value outliers on the model.

2. Bedrooms (bed): The histogram for bed also exhibits right skewness, with most properties having a small number of bedrooms. Outliers with a high number of bedrooms are rare, but their presence skews the data, justifying normalization to handle these variations effectively.

3. Bathrooms (bath): Similar to bedrooms, the number of bathrooms in properties is concentrated at the lower end, with a few properties having a much larger number of bathrooms. This skewed distribution can affect model performance, making normalization necessary.

4. Carport: The carport feature has a sharp peak near zero, indicating that many properties either lack carports or have only one. Few properties have multiple carports, resulting in a right-skewed distribution. Normalization will help mitigate the effect of these rare high values.

5. Surface Area: The distribution of surface area is right-skewed, with most properties having relatively small surface areas. Larger properties are few but contribute significantly to the skewness. Normalizing this feature will make it more comparable to other features in the model.

Given the heavy right skewness across all features, normalization is essential to ensure that each feature contributes proportionally to the predictive model. Using a standard scaler, which centers the data by subtracting the mean and scales it by the standard deviation, will adjust each feature to a more normalized range. This transformation will reduce the impact of extreme values, improve model stability, and enhance predictive accuracy.

**The example of OHE normalized**
![OHE norm](assets\dummies_encoding_normalized.png)

The example of Target Encoding normalized
![target norm](assets\target_encoding_normalized.png)

# Modeling

To address the problem of predicting house prices, we developed and tested several machine learning models. Our goal was to identify the most effective model for accurately predicting property prices based on the available features. The models were chosen to represent both linear and tree-based approaches, allowing us to compare their performance on this dataset.

**Modeling Process and Parameters**

The modeling process involved multiple stages:

1. Model Selection: We selected four models – two linear models (Linear Regression and ElasticNet) and two tree-based models (Decision Tree and Random Forest). Each model type has unique characteristics, providing insights into which approach works best for this dataset.

2. Parameter Configuration: For each model, initial default parameters were used to establish baseline performance. The parameters for Linear Regression include the intercept and coefficients for each feature. ElasticNet, which combines L1 and L2 regularization, involves setting alpha (the regularization strength) and the L1_ratio (the balance between L1 and L2 penalties). Decision Tree and Random Forest have parameters such as maximum depth, minimum samples per leaf, and the number of estimators (for Random Forest).

3. Evaluation Metric: Mean Squared Error (MSE) was chosen as the primary evaluation metric to assess each model’s performance. This metric emphasizes large errors, making it suitable for continuous data like house prices.

**Strengths and Weaknesses of Each Algorithm**

- Linear Regression: A simple and interpretable model, well-suited for datasets with linear relationships between features and the target. However, it may perform poorly if relationships are non-linear or if the data has outliers.

- ElasticNet: A regularized linear model that balances L1 and L2 penalties, helping reduce overfitting and manage multicollinearity. ElasticNet is more robust than basic Linear Regression, but it requires tuning the regularization parameters to achieve optimal results.

- Decision Tree: A tree-based model that can handle non-linear relationships and interactions between features. It is easy to interpret but prone to overfitting, especially with deep trees, leading to high variance.

- Random Forest: An ensemble model of multiple decision trees, offering improved stability and generalization compared to a single Decision Tree. It handles non-linearities well and reduces overfitting but can be computationally expensive and less interpretable than single trees.

**Model Improvement through Hyperparameter Tuning**

The best-performing model will undergo hyperparameter tuning to further improve its accuracy. Hyperparameter tuning involves systematically adjusting the model’s parameters to find the optimal configuration. Techniques like Grid Search or Random Search can be employed to test various parameter combinations, maximizing model performance. For example, in Random Forest, tuning parameters such as the number of estimators, maximum depth, and minimum samples per leaf can enhance accuracy.

Model Selection
After evaluating all models, the best-performing model in each category – ElasticNet for linear models and Random Forest for tree-based models – is expected to emerge as the top performers. The final choice for the best model will be based on overall accuracy, stability, and generalization capacity, and will be discussed in the following section.

# Evaluation

To assess the performance of our machine learning models for house price prediction, we used Mean Squared Error (MSE) as the primary evaluation metric. MSE is well-suited for regression tasks as it penalizes large errors more than small ones, making it particularly useful for predicting continuous data, such as property prices. The MSE metric calculates the average squared difference between predicted and actual values, with the formula:

<p align="center">
  <img src="assets\MSE.png" alt="Alt Text">
</p>
where:

- 𝑦𝑖 is the actual value,
- ŷ𝑖 is the predicted value, and
- 𝑛 is the total number of observations.

Using MSE aligns with our project objectives, as it provides a direct measure of model accuracy and highlights errors in predictions that might be particularly impactful in the context of real estate pricing.

## Evaluation Result

The models were evaluated on both training and testing datasets to monitor performance and prevent overfitting. Calculating the train MSE allows us to check if the model fits the training data well without learning noise. Below are the MSE results for each model:

- Linear Regression: Train MSE = 0.0, Test MSE = 0.1834501641718185
- ElasticNet: Train MSE = 0.0, Test MSE = 0.2663851445340297
- Decision Tree: Train MSE = 0.0, Test MSE = 0.27728575014129847
- Random Forest: Train MSE = 0.0, Test MSE = 0.10164993726005747

The results reveal that all models achieve a perfect fit on the training data (Train MSE = 0.0). On the test data, Random Forest achieved the lowest MSE, suggesting it generalizes well compared to the other models.

## Hyperparameter Tuning

Since ElasticNet did not meet expectations, we performed hyperparameter tuning on it to enhance its performance. Hyperparameter tuning was conducted using Grid Search on a predefined parameter grid to identify the best configuration for ElasticNet. The parameter grid for ElasticNet included options for `alpha`, `l1_ratio`, `fit_intercept`, `max_iter`, `tol`, `selection`, and `warm_start`:

```
param_grid = {
  'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
  'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
  'fit_intercept': [True, False],
  'max_iter': [1000, 5000, 10000],
  'tol': [1e-4, 1e-5, 1e-6],
  'selection': ['cyclic', 'random'],
  'warm_start': [True, False]
}
```

The best parameter configuration for ElasticNet was: `{'alpha': 0.1, 'fit_intercept': False, 'l1_ratio': 0.5, 'max_iter': 1000, 'selection': 'random', 'tol': 0.0001, 'warm_start': True}`

This resulted in a test MSE of 0.123828587510955, which improved ElasticNet’s performance and brought it closer to Random Forest.

A similar hyperparameter tuning process was applied to Random Forest, exploring parameters such as `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `oob_score`, and `n_jobs`:

```
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False],
    'oob_score': [True, False],
    'n_jobs': [-1]
}
```

The optimal parameters for Random Forest were: `{'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50, 'n_jobs': -1, 'oob_score': False}`

This configuration achieved an MSE of 0.10098347270551815 on the test set, confirming Random Forest as the best model for this task.

## Model Selection

Based on the evaluation results, Random Forest emerged as the best model due to its lowest test MSE. This model effectively balances training fit and generalization on unseen data, making it a reliable choice for predicting house prices in this dataset.

When making predictions (inference) with the chosen model, a special approach is needed due to preprocessing techniques applied during model training, such as normalization and One-Hot Encoding (OHE). These transformations must be consistently applied to new data to ensure the model interprets it in the same way as during training. Here's a breakdown of these requirements:

1. Normalization: The model was trained on normalized data, where features were scaled to have a mean of 0 and a standard deviation of 1 using techniques like Standard Scaler. For inference, any new data must also undergo the same normalization process using the mean and standard deviation values derived from the training set. This ensures that the scale of the input features remains consistent, preventing errors in predictions due to differences in data distribution.

2. One-Hot Encoding (OHE): For categorical features, One-Hot Encoding was used to convert categories into binary vectors, creating separate columns for each category. To maintain consistency, new data for

inference must also undergo OHE with the same categories used during training. This is critical to avoid misalignment in feature dimensions or missing categories. If a category in the new data was not present in the training data, it should be managed appropriately (e.g., assigning it to a generic "unknown" category or using sparse representation).
