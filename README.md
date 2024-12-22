# Project Overview

## Research Question

This project is aimed at answering the question: **'How might we use machine learning to understand the role of social pressure in influencing voting behaviour?'.**

This question can be broken down further into 3 sub-questions: 

1. **Does social pressure influence voting behaviour in terms of changing voting rates?**
2. **If so, by how much?**
3. **For whom, in particular?**

## Motivation and Impact

Understanding how social pressure influences voting behavior can help in designing more effective voter mobilization campaigns. In particular identifying which subgroups in a population are particularly responsive to social pressure can help policymakers and political advocacy groups develop targeted interventions to increase total participation in elections and political representation of historically underrepresented voters, thereby strengthening the democratic process.

## Data

The dataset I have used comes from a paper published in 2008 by Donald P. Green, Alan S. Gerber and Christopher W. Larimer called "Social Pressure and Voter Turnout: Evidence from a Large-Scale Field Experiment". The paper contains a sample size of 344,084 individuals (registered voters) and 180,002 households in the U.S. state of Michigan.

Paper reports results of a field experiment whereby the experimenters sent mail to households that were randomly assigned to one of 4 treatment group. Each treatment group received mail containing one of 4 different prompts with each prompt escalating in the degree of social pressure they exert on the recipient to vote in the upcoming 2006 Michigan Primary Election. The paper then measures whether each individual voted in the election. The dataset also contains data on each individual's previous voting behaviour and census data from the United States Census Bureau that describes demographic, employment and educational information for each ZIP code studied in the paper. The combined replication data and census datasets come from [this GitHub repo](https://github.com/itamarcaspi/experimentdatar) containing datasets used in a [course](https://www.aeaweb.org/conference/cont-ed/2018-webcasts) by Susan Athey and Guido Imbens. 

**Each row in my dataset contains the following information about a single individual:**
1. Outcome/target Variable: Whether or not they voted in the Michigan 2006 Primary Election
2. Treatment Variable: Which treatment group they were in (or whether they were in the control group)
3. Feature Variables: Demographic information at the individual level (e.g. household size, age, sex) and at the ZIP code level (percentage of people in ZIP code between ages of 18 and 64). 

I have included a data dictionary in my accompanying Jupyter Notebooks but chose not to include it here as it is quite lengthy. 

All datasets used in my project are accessible from this [Google Drive](https://drive.google.com/drive/folders/1UqjmiWH4Zvxt3b-Ve2yyaAsAzuwhuxI4?usp=sharing). 

## Project Organisation

My project is organised according to the following components:

1. Initial EDA: This notebook contains the first stage of the project, focusing on conducting initial exploratory data analysis on my dataset. I also identify potential data quality issues and other concerns to prepare my dataset for the preprocessing required before modelling. 

2. Advanced EDA and Baseline Modelling: This notebook contains the second stage of my project, focusing on establishing some baseline modelling approaches. I explore, evaluate and compare 5 different modelling approaches. I also perform some feature engineering and advanced EDA using clustering to identify subgroups in my dataset. 

3. Causal Inference Modelling: This notebooks contains the third stage of my project, focusing on exploring combined machine learning and causal inference models. In particular, I used the [Causal Forest model](https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html#econml.dml.CausalForestDML) to estimate the conditional average treatment effect (CATE) and identify subgroups in the population that significantly differ in their responsiveness to the treatment. 

## Modelling Approaches

I have chosen to explore two distinct modelling approaches in this project - a **'Pure' ML** Approach and a **'Causal Inference'** approach. Here is a quick summary of the objectives, methodology, strengths and weaknesses of each approach. 

### Pure ML Approach

**Objective:** Primarily aims to predict outcomes based on input variables. The focus is on maximizing the accuracy of predictions without necessarily understanding the relationships or causal mechanisms between variables.

**Methodology:** Uses standard supervised learning methods to learn relationships between features and outcomes. These models include decision trees, random forests, neural networks, etc., which optimize for prediction accuracy and are often evaluated on metrics like f1, RMSE, accuracy, or AUC.

**Strengths:** Highly effective at making accurate predictions, which is valuable for applications like image recognition, language processing, or market forecasting.

**Weaknesses:** Cannot determine causal relationships; models may be biased by spurious correlations if not carefully designed. Also, interpreting marginal impacts of changes in one variable on another is difficult (i.e. there is a lack of interpretability).

### Causal Inference Approach

**Objective:** Aims to estimate the causal effect of an intervention on an outcome. This approach not only predicts but also explains how changes in the inputs (i.e. treatment, features) causally affect the outcome.

**Methodology:** Utilizes statistical methods that focus on estimating the conditional average treatment effect (CATE) and other causal impacts. This includes using potential outcomes and counterfactual reasoning, which consider what would happen both with and without the treatment. Often evaluated using metrics like MSE of the estimated vs. actual treatment effects and other statistical inference metrics like R-squared or standard errors. 

**Strengths:** Provides insights into causal relationships, not just correlations. This is crucial for policy-making and understanding the effects of interventions. Also allows for the identification of heterogeneity in treatment effects, which allow for the identification of particularly responsive subgroups in a population.

**Weaknesses:** Requires careful setup and strong assumptions about the data and model, such as assumptions about no unmeasured confounders. Could also be more computationally expensive, making hyperparameter tuning to pick the best models more difficult. 

## Results 

- 'Pure' ML models prove inadequate in answering our specific research questions about the causal impact of social pressure on voting behavior. The key issues include a lack of consensus among models on the importance of social pressure as a predictive feature, the inability to interpret feature importance as causal magnitude, and the incapacity to uncover heterogeneity in treatment effects across different subgroups. While some models find social pressure influential, others do not, or even suggest a negative effect. Furthermore, most models provide outputs like feature importances that, unlike logistic regression's probability outputs, cannot directly measure the effect of treatment on voting. Consequently, these models fail to offer clear answers about the presence, magnitude, or demographic specifics of social pressure’s effect on voting, often averaging out any underlying variabilities.

- The combined causal inference and machine learning model (Causal Forest) provides a more robust analysis of the impact of social pressure on voting behavior, addressing the limitations observed in standard machine learning approaches. These models confirm that social pressure does indeed increase voting rates by an average of 4.1%, offering a quantifiable measure of the treatment's effectiveness, unlike the ambiguous findings from pure ML models. Moreover, they enable the identification of specific subgroups that are more or less responsive to social pressure, highlighting age, unemployment, and racial/ethnic demographics as key differentiators. This capability to discern and quantify treatment effects and to pinpoint their variability across different demographics provides a more nuanced understanding of the causal relationships at play, which pure ML models fail to achieve.


## Limitations 

1. **Lack of Interpretability of 'Pure' ML Models**: Pure machine learning models, such as decision trees, random forests, gradient boosting classifiers, and neural networks, suffer from a lack of interpretability. This opaqueness is a significant challenge in sectors like public policy, where understanding the rationale behind predictions is essential for trust and implementation. Despite extensive hyperparameter tuning, these models showed only minor improvements in accuracy and F1 scores over baseline models. Moreover, their outputs are generally not interpretable in a causal framework, which further complicates their use in applications where understanding the impact of variables is critical, like in the case of voting. This inability to elucidate model reasoning limits stakeholders' confidence in the applicability and reliability of the insights these models provide.

2. **Large Standard Errors for Average Treatment Effect (ATE) of Causal Forest Model**: The average treatment effect (ATE) estimates from causal forest models tend to have large standard errors, indicating a high degree of uncertainty in these estimates. This variability can undermine confidence in the models' predictions and limit their practical applicability in policy-making where precise effect estimates are necessary. The large standard error suggests that the model may not perform consistently across different subsets of the data or under slightly modified conditions.

3. **Lack of External Validity**: The findings from this analysis are highly specific to the dataset and context in which they were developed. Specifically, they tell us about the impact of mailings that exert social pressure on voters in Michigan in the 2006 Primary Election. These results may not necessarily apply to other regions, different methods of exerting social pressure, different electoral conditions, or demographic setups. This limited external validity restricts the generalizability of the conclusions, making it risky to apply the insights to broader contexts without additional validation.

## Next Steps

1. **Hyperparameter tuning of Causal Forest model**: Hyperparameter tuning is essential to optimize the performance of causal forest models. By adjusting parameters like the number of trees, depth of trees, or minimum samples per leaf, one can potentially reduce the variance of the ATE estimates and improve the model's predictive accuracy. Techniques such as grid search, random search, or Bayesian optimization could be employed to methodically explore the hyperparameter space. This systematic approach not only refines the model's efficiency but also enhances its reliability and the robustness of its output. However, due to computational and time constraints, this was out of the scope of this analysis. 

2. **Seek new data sources**: Expanding the data sources to include more diverse demographics, various types of elections, and different communication methods for delivering treatments can significantly enhance the external validity of the model. By applying the model to new datasets that mirror a broader range of electoral scenarios and voter behaviors would allow us to extend this analysis further. 

3. **Use Bayesian Networks to improve causal interpretation**: Bayesian networks offer a robust statistical tool for modeling complex relationships between variables and enhancing causal interpretation. By explicitly modeling conditional dependencies and integrating prior knowledge, Bayesian networks can provide a clearer picture of the influence of causal mechanisms in our data. 