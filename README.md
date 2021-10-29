Development and Validation of AKI Predictive Models Based on Personalized Model with Transfer Learning
=====================================================================

This is the working directory for building and validating a Personalized Model with Transfer Learning (PMTL) for Acute Kidney Injury (AKI) prediction based on electronic health records (EHRs) from University of Kansas Medical Center (KUMC).

by Kang Liu, with Yong Hu and Mei Liu
[Big Data Decision Institute, Jinan University][BDDI]
[Medical Informatics Division, Univeristy of Kansas Medical Center][MI]

[BDDI]: https://bddi.jnu.edu.cn/
[MI]: http://informatics.kumc.edu/

Copyright (c) 2021 Jinan University  
Share and Enjoy according to the terms of the MIT Open Source License.

***

## Background

Acute Kidney Injury (**AKI**) is a heterogeneous syndrome, affecting 10-15% of all hospitalized patients and >50% of the intensive care unit (ICU) patients. In this application, we propose to build personalized predictive models to identify patients at risk for hospital-acquired AKI as well as their risk factors, and externally validate the models in different heterogeneous subgroups of patients. The project was carried out with the following aims:

* **Aim 1 - Development of PMTL**: Personalized AKI prediction modeling approach was developed and internally cross-validated using electronic medical record (EMR) data from the University of Kansas Medical Center’s (KUMC) de-identified clinical data repository called [HERON] (Health Enterprise Repository for Ontological Narration). 
      * Task 1.1: data extraction and quality check       
      * Task 1.2: exploratory data analysis (e.g. strategies for data cleaning and representation, feature engineering)     
      * Task 1.3: developing Similar Sample Matching module and Similarity Measure Optimization module
      * Task 1.4: addressing diminishing sample size after similar sample matching (Transfer learning module is developed)
      * Task 1.5: developing proposed personalized models (PMTL)    
 
* **Aim 2 -  Validation of PMTL**: Validating PMTL in heterogeneous patients. We implemented an automated package to develop PMTL for each general patient. Prediction performance of PMTL was validated in general patients, high-risk subgroups, low-risk patients, and subgroups studied by previous AKI prediction literature and compared with global, subgroup and previous model accordingly. Prediction result of PMTL for a patient will not change in different experiments; performance change of PMTL in different population caused by the change of test sample selected.
      * Task 2.1: testing of PMTL in all test samples        
      * Task 2.2: Comparing  PMTL with global model in general and low-risk patients
      * Task 2.3: Comparing  PMTL, global and subgroup model in high-risk patients
      * Task 2.4: Comparing  PMTL, global, subgroup and previous model in subgroups from previous literature

* **Aim 3 - Interaction analysis of risk factors**: Analyzing and visualizing the effect change of top important predictors in different sub-population, and exploring interaction of predictors related to the effect change.
      * Task 3.1: rank predictors based on their importance in improving model performance for general patients
      * Task 3.2: evaluate influence changes of predictors in different subgroups and persons. 
      * Task 3.3: analyze interaction of predictors based on meta-regression and subgroup analysis.

[HERON]:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3243191/

***

## Data Preprocessing

For each hospital admissions (encounters) in the data set, we extracted all demographic information, vitals data, medications, past medical diagnoses, and admission diagnosis from EMR. For test laboratory, we extracted a selected list of laboratory variables that may represent potential presence of a comorbidity correlated with AKI ([Matheny, M.E., et al., 2010]). SCr and eGFR were not included as predictors because they were used to determine the occurrence of AKI.

Variables are time-stamped and every encounter in the dataset was represented by a sequence of clinical events construed by clinical observation vectors aggregated on daily basis. The prediction point was 1-day prior to AKI onset for AKI patients and 1-day prior to the last SCr record for non-AKI patients. We performed a data preprocessing process as follows: 
1) Medication exposure included inpatient (i.e. drug used during hospitalization) and outpatient medications (i.e. medication reconciliation and prior outpatient prescriptions). Medication names were normalized by mapping to RxNorm ingredient. Medication exposure was defined as true if it is taken within 7-days before a prediction point.
2) Admission diagnoses were represented using the All Patients Refined Diagnosis Related Group ([APR-DRG]),  collected from the University Health System Consortium ([UHC]) data source in [HERON]. We performed one-hot-coding on admission diagnosis to convert them into binary representations.
3) Patient medical history was captured as major diagnoses (ICD-9 codes grouped according to the Clinical Classifications Software ([CCS]) diagnosis categories by the Agency for Healthcare Research and Quality). We considered the presence/absence of each major diagnosis before the prediction point.
4) Vitals were categorized according to commonly used standards and missing values were treated as a unique category. The last recorded value before a prediction point was used.
5) Labs were categorized as “unknown”, “present and normal”, or “present and abnormal”. The last recorded value before a prediction point was used.
6) Demographics were converted into binary variables based on one-hot-coding.

[APR-DRG]: https://www.health.ny.gov/facilities/hospital/reimbursement/apr-drg/weights/docs/siw_alos_2014.xls
[Matheny, M.E., et al., 2010]: https://pubmed.ncbi.nlm.nih.gov/20354229/
[UHC]: https://www.vizientinc.com
[CCS]: [https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)

***

## Requirements
In order to run predictive models and generate final report, the following infrastructure requirement must be satisfied:

- **Python**: version >=3.7.4 is required.    
- **[Scikit-learn]**: A widely used package for machine learning in python. The version 0.19.2 is used to perform most of our experiment, but the experiment of calibration need version 0.24.2.
- **[ PyMARE]**: A python package used for  meta-regression.

[Scikit-learn]: https://scikit-learn.org/stable/
[ PyMARE]: https://pymare.readthedocs.io/en/latest/
***

## Model Validation
The following instructions are for generating final report from our study cohort.

### Part I: Data preparation
 1. Please make sure class label of patients (AKI or not AKI) are placed in the last column of the data sheet used for model training and testing.
 2. Edit the file path of input and output in our python code.

### Part II: Development of PMTL
1. Comparing different approaches for personalized modeling
- **Aim**: Identify suitable methods for personalized modeling by comparing different approach for similar sample matching, similarity measure optimization and addressing diminishing sample size in validation set. 
-  **Code**: Main codes used for these experiments are saved in folder "[Model_development]". For example, "PM-KNN_PCA.py" means similar sample matching are based on "k-nearest neighbor" algorithm, PCA is use for feature selection, similarity measure optimization is not performed.  All the codes can output prediction of models with and without transfer learning.
2. Developing PMTL
 -  **Code**: The final code used for PMTL training is "PMTL_training.py". It outputs the optimized similarity measure for  similar sample matching.

[Model_development]: https://github.com/BDAII/PMTL/tree/main/Model_development

### Part III: Validation of PMTL
1. Validation in general patients
- **Save**  the similarity measure learned by  "PMTL_training.py" from training data.
- **Edit** "PMTL_testing.py"  in folder "[Model_validation]" by specifying the file path where you saved the similarity measure as well as path of training set and test set.
- **Run** "PMTL_testing.py" , and it will output prediction of PMTL (with specific number of similar sample) and global model (built with 100% sample) for each patients as well as logistic regression coefficients of predictors for each patients in PMTL. 
- **Run** "Model_comparison.py" (in folder "[Model_validation]") to compare model discrimination between PMTL and global model, or use software like [Medcalc] (proposed for AUROC).
- **Run** "Calibration_analysis.py" (in folder "[Model_validation]") to compare model calibration

2. Validation in high-risk and low-risk patients
- **Edit** "Subgroup_modeling_in_subgroups_of_our_data.py"  in folder "[Model_validation]" by specifying the file path you save the list of features used for determined high-risk subgroups as well as path of training set and test set.
- **Run** "Subgroup_modeling_in_subgroups_of_our_data.py" , and it will output prediction of subgroup models for each patient in these high-risk subgroups, and the prediction probability of remaining low-risk patients is set to 0, you can filter them out easily with EXCEL. You can perform model comparison similar to the case in  general patients.

3. Validation in subgroups from previous studies
- **Edit** "Subgroup_modeling_in_subgroups_in_previous_study.py"  in folder "[Model_validation]" by specifying the file path you save the list of features used for determined subgroups in previous study, path of training set and test set, and path of prediction result of PMTL for general patients.
- **Run** "AUC_of_model_in_subgroups_in_previous_study.py"  , and it will output AUROC of subgroup, global and personalized model.
- **Edit** "AUC_std_in_subgroups_in_previous_study_PMTL.py"  in folder "[Model_validation]" by specifying the file path you save the list of features used for determined subgroups in previous study and path of prediction result of PMTL for general patients.
- **Run** "AUC_std_in_subgroups_in_previous_study_PMTL.py", and it will output standard deviation of AUROC  of PMTL, and we can use it to compare PMTL with models in previous study.


[Model_validation]: https://github.com/BDAII/PMTL/tree/main/Model_validation
[Medcalc]: https://www.medcalc.org/

### Part IV: Analysis of predictor interaction
- **Edit** "AUC_gain_of_predictors_in_global_model_and_PMTL.py" and  "AUC_gain_change_in_subgroups.py" in folder "[Interaction_analysis]" by specifying the file path of training set, test set, prediction result of PMTL for general patients (saving the intercept of each PMTL), and coefficients of PMTL for each general patient. The file path saves the list of features used for determined high-risk subgroups is also needed for "AUC_gain_change_in_subgroups.py".
- **Run** "AUC_gain_of_predictors_in_global_model_and_PMTL.py". It outputs AUROC gain of each predictor (i.e. AUROC change of model when a predictor is removed) for PMTL and global model in predicting general patients. Then we can rank the importance of predictors using EXCEL.
- **Run** "AUC_gain_change_in_subgroups.py". It outputs AUROC gain of each predictor for PMTL and global model in predicting different subgroups. 
- **Edit** "AUC_with_top_predictors_PMTL.py" and "AUC_with_top_predictors_global_model.py" in folder "[Interaction_analysis]" by specifying the file path of training set, test set, list of important features you select and list of features used for determined high-risk subgroups. The file path saves coefficients of PMTL for each general patient is also needed for "AUC_with_top_predictors_PMTL.py".
- **Run** "AUC_with_top_predictors_PMTL.py" and  "AUC_with_top_predictors_global_model.py".  It outputs AUROC of PMTL and global model  in predicting different subgroups when only the important predictors is considered.
- **Edit** "SE_analysis_for_PMTL.py" by specifying the file path of training set, test set and similarity measure learned in training set. 
- **Run** "SE_analysis_for_PMTL.py", and it output standard error of coefficient estimated by PMTL. This script is not proposed to used when sample size cannot significantly larger than potential predictors; in such a case, such as we estimated standard error of coefficient in subgroup model, we sampling train set with replacement and rebuilt model multiple times, but this approach spend a quite long time in PMTL.
-  **Edit** "Interaction_discover_by_meta_regression.py" by specifying the file path of a list of target predictors, test set, coefficients of PMTL for each general patient, and standard error of coefficient estimation in PMTL.
-  **Run** "Interaction_discover_by_meta_regression.py" and the return results can help us analyze which factor may interact with target predictors.

[Interaction_analysis]: https://github.com/BDAII/PMTL/tree/main/Interaction_analysis

***
*updated 10/29/2021*


