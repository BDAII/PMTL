Development and Validation of AKI Predictive Models Based on Personalized Model with Transfer Learning
=====================================================================

This is the working directory building and validating Personalized Model with Transfer Learning (PMTL) for Acute Kidney Injury (AKI) prediction based on electronic health records (EHRs) from University of Kansas Medical Center (KUMC).

by Kang Liu, with Yong Hu, Mei Liu
[Big Data Decision Institute, Jinan University][BDDI]
[Medical Informatics Division, Univeristy of Kansas Medical Center][MI]

[BDDI]: https://bddi.jnu.edu.cn/
[MI]: http://informatics.kumc.edu/

Copyright (c) 2021 Jinan University  
Share and Enjoy according to the terms of the MIT Open Source License.

***

## Background

Acute Kidney Injury (**AKI**) is a heterogeneous syndrome, affecting 10-15% of all hospitalized patients and >50% of the intensive care unit (ICU) patients. In this application, we propose to build personalized predictive models to identify patients at risk for hospital-acquired AKI as well as their risk factors, and externally validate the models in different heterogeneous subgroups of patients. The project will be carried out with the following aims:

* **Aim 1 - Development of PMTL**: Personalized AKI prediction modeling approach was develop and internally cross-validate using electronic medical record (EMR) data from the University of Kansas Medical Center’s (KUMC) de-identified clinical data repository called [HERON] (Health Enterprise Repository for Ontological Narration). 
      * Task 1.1: data extraction and quality check       
      * Task 1.2: exploratory data analysis (e.g. strategies for data cleaning and representation, feature engineering)     
      * Task 1.3: developing Similar Sample Matching module and Similarity Measure Optimization module
      * Task 1.4: addressing diminishing sample size after similar sample matching (Transfer learning module is developed)
      * Task 1.5: developing proposed personalized models (PMTL)    
 
* **Aim 2 -  Validation of PMTL**: Validating PMTL in heterogeneous patients. We implement an automated package to develop PMTL for each general patient. Prediction performance of PMTL was validated in general patients, high-risk subgroups, low-risk patients, and subgroups studied by previous AKI prediction literature and compared with global, subgroup and previous model accordingly. Prediction result of PMTL for a patient will not change in different experiments; performance change of PMTL in different population caused by the change of test sample we selected.
      * Task 2.1: testing of PMTL in all test samples        
      * Task 2.2: Comparing  PMTL with global model in general and low-risk patients
      * Task 2.3: Comparing  PMTL, global and subgroup model in high-risk patients
      * Task 2.4: Comparing  PMTL, global, subgroup and previous model in subgroups from previous literature

* **Aim 3 - Interaction analysis of risk factors**: Analyzing and visualizing the effect change of top important predictors in different sub-population, and exploring interaction of predictors related to the effect change.
      * Task 3.1: rank predictors based on their importance in improving model performance for general patients
      * Task 3.2: evaluate influence changes of predictors in different subgroups and persons. 
      * Task 3.3: analyze interaction of predictors based on meta-regression and subgroup analysis.

[HERON]:https://pubmed.ncbi.nlm.nih.gov/20190053/

***

## Data Preprocessing

For each hospital admissions (encounters) in the data set, we extracted all demographic information, vitals data, medications, past medical diagnoses, and admission diagnosis from EMR. For test laboratory, we extracted a selected list of laboratory variables that may represent potential presence of a comorbidity correlated with AKI ([Matheny, M.E., et al., 2010]). SCr and eGFR were not included as predictors because they were used to determine the occurrence of AKI.

Variables are time-stamped and every encounter in the dataset was represented by a sequence of clinical events construed by clinical observation vectors aggregated on daily basis. We performed a data preprocessing process as follows: 
1) Medication exposure included inpatient (i.e. drug used during hospitalization) and outpatient medications (i.e. medication reconciliation and prior outpatient prescriptions). Medication names were normalized by mapping to RxNorm ingredient. Medication exposure was defined as true if it is taken within 7-days before a prediction point.
2) Admission diagnoses were represented using the All Patients Refined Diagnosis Related Group (APR-DRG),  collected from the University Health System Consortium ([UHC]) data source in [HERON]. We performed one-hot-coding on admission diagnosis to convert them into binary representations.
3) Patient medical history was captured as major diagnoses (ICD-9 codes grouped according to the Clinical Classifications Software ([CCS]) diagnosis categories by the Agency for Healthcare Research and Quality). We considered the presence/absence of each major diagnosis before the prediction point.
4) Vitals were categorized according to commonly used standards and missing values were treated as a unique category. The last recorded value before a prediction point was used.
5) Labs were categorized as “unknown”, “present and normal”, or “present and abnormal”. The last recorded value before a prediction point was used.
6) Demographics were converted into binary variables based on one-hot-coding.

[Matheny, M.E., et al., 2010]: https://pubmed.ncbi.nlm.nih.gov/20354229/
[UHC]: https://www.vizientinc.com
[CCS]: [https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp)

***
## Previous text

All variables are time-stamped and every patient in the dataset was represented by a sequence of clinical events construed by clinical observation vectors aggregated on daily basis.

The initial feature set contained more than 30,000 distinct features. We performed an automated curation process as follows: 
1) systematically identified extreme values of numerical variables (e.g., lab test results and vital signs) that are beyond 1st and 99th percentile as outliers and removed them; 
2) performed one-hot-coding on categorical variables (e.g., diagnosis and procedure codes) to  convert them into binary representations; 
3) used the cumulative-exposure-days of medications as predictors instead of a binary indicator for the sheer existence of that medication; 
4) when repeated measurements presented within certain time interval, we chose the most recent value; 
5) when measurements are missing for a certain time interval, we performed a common sampling practice called sample-and-hold which carried the earlier available observation over; 
6) introduced additional features such as lab value changes since last observation or daily blood pressure trends, which have been shown to be predictive of AKI10.

The discrete-time survival model ([DTSA]) required converting the encounter-level data into an Encounter-Period data set with discrete time interval indicator (i.e. day1, day2, day3,...). More details about this conversion can be found in the `format_data()` and `get_dsurv_temporal()` functions from `/R/util.R`. As shown in the figure below: ![Figure1-Data Preprocess.](/figure/preproc_demo.png), 

AKI patient at days of AKI onset contributed to positive outcomes, while earlier non-AKI days of AKI patients as well as daily outcomes of truely non-AKI patients (i.e. who never progressed to any stage of AKI) contributed to nefative outcomes. 


[DTSA]: https://www.jstor.org/stable/1165085?seq=1#metadata_info_tab_contents
***

## Requirements
In order to run predictive models and generate final report, the following infrastructure requirement must be satisfied:

**R program**: [R Program] (>=3.3.0) is required and [R studio] (>= 1.0.136) is preferred to be installed as well for convenient report generation.    
**DBMS connection**: Valid channel should also be established between R and DBMS so that communication between R and CDM database can be supported.    
**Dependencies**: A list of core R packages as well as their dependencies are required. However, their installations have been included in the codes. 
* [DBI] (>=0.2-5): for communication between R and relational database    
* [ROracle] (>=1.3-1): an Oracle JDBC driver    
* [RJDBC]: a SQL sever driver    
* [RPostgres]: a Postgres driver    
* [rmarkdown] (>=1.10): for rendering report from .Rmd file (*Note: installation may trip over dependencies [digest] and [htmltools] (>=0.3.5), when manually installation is required*).     
* [dplyr] (>=0.7.5): for efficient data manipulation    
* [tidyr] (>=0.8.1): for efficient data manipulation    
* [magrittr] (>=1.5): to enable pipeline operation    
* [stringr] (>=1.3.1): for handling strings     
* [knitr] (>=1.11): help generate reports
* [kableExtra]: for generating nice tables
* [ggplot2] (>=2.2.1): for generating nice plots    
* [ggrepel]: to avoid overlapping labels in plots   
* [openxlsx] (>=4.1.0): to save tables into multiple sheets within a single .xlsx file      
* [RCurl]: for linkable descriptions (when uploading giant mapping tables are not feasible)
* [XML]: for linkable descriptions (when uploading giant mapping tables are not feasible)
* [xgboost]: for effectively training the gradient boosting machine   
* [pROC]: for calculating receiver operating curve 
* [PRROC]: for calculating precision recall curve
* [ParBayesianOptimization]: a parallel implementation for baysian optimizaion for xgboost
* [doParallel]: provide backend for parallelization


[R Program]: https://www.r-project.org/
[R studio]: https://www.rstudio.com/
[DBI]: https://cran.r-project.org/web/packages/DBI/DBI.pdf
[ROracle]: https://cran.r-project.org/web/packages/ROracle/ROracle.pdf
[RJDBC]: https://cran.r-project.org/web/packages/RJDBC/RJDBC.pdf
[RPostgres]: https://cran.r-project.org/web/packages/RPostgres/RPostgres.pdf
[rmarkdown]: https://cran.r-project.org/web/packages/rmarkdown/rmarkdown.pdf
[dplyr]: https://cran.r-project.org/web/packages/dplyr/dplyr.pdf
[tidyr]: https://cran.r-project.org/web/packages/tidyr/tidyr.pdf
[magrittr]: https://cran.r-project.org/web/packages/magrittr/magrittr.pdf
[stringr]: https://cran.r-project.org/web/packages/stringr/stringr.pdf
[knitr]: https://cran.r-project.org/web/packages/knitr/knitr.pdf
[kableExtra]: http://haozhu233.github.io/kableExtra/awesome_table_in_html.html
[ggplot2]: https://cran.r-project.org/web/packages/ggplot2/ggplot2.pdf
[ggrepel]: https://github.com/slowkow/ggrepel
[openxlsx]: https://cran.r-project.org/web/packages/openxlsx/openxlsx.pdf
[digest]: https://cran.r-project.org/web/packages/digest/digest.pdf
[htmltools]:  https://cran.r-project.org/web/packages/htmltools/htmltools.pdf
[RCurl]: https://cran.r-project.org/web/packages/RCurl/RCurl.pdf
[XML]: https://cran.r-project.org/web/packages/XML/XML.pdf
[xgboost]:https://xgboost.readthedocs.io/en/latest/   
[pROC]: https://cran.r-project.org/web/packages/pROC/pROC.pdf
[PRROC]: https://cran.r-project.org/web/packages/PRROC/PRROC.pdf
[ParBayesianOptimization]: https://cran.r-project.org/web/packages/ParBayesianOptimization/ParBayesianOptimization.pdf 
[doParallel]: https://cran.r-project.org/web/packages/doParallel/doParallel.pdf

***


## Site Usage for Model Validation
The following instructions are for extracting cohort and generating final report from a `DBMS` data source (specified by `DBMS_type`) (available options are: Oracle, tSQL, PostgreSQL(not yet)) 

### Part I: Study Cohort and Variable Extraction

1. Get `AKI_CDM` code
  - **download** the [AKI_CDM] repository as a .zip file, unzip and save folder as `path-to-dir/AKI_CDM`    
  *OR*  
  - **clone** [AKI_CDM] repository (using [git command]):   
      i) navigate to the local directory `path-to-dir` where you want to save the project repository and     
      type command line: `$ cd <path-to-dir>`   
      ii) clone the AKI_CDM repository by typing command line: `$ git clone https://github.com/kumc-bmi/AKI_CDM`  


2. Prepare configeration file `config.csv` and save in the AKI_CDM project folder    
      i) **download** the `config_<DBMS_type>_example.csv` file according to `DBMS_type`      
      ii) **fill in** the content accordingly (or you can manually create the file using the following format)      
    
    |username     |password    |access         |cdm_db_name/sid                 |cdm_db_schema      |temp_db_schema |   
    |:------------|:-----------|:--------------|:-------------------------------|:------------------|:--------------|    
    |your_username|your_passwd |host:port    |database name(tSQL)/SID(Oracle) |current CDM schema |default schema |   
    
      iii) **save as** `config.csv` under the same directory         
      

[AKI_CDM]: https://github.com/kumc-bmi/AKI_CDM
[git command]: https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository



3. Extract AKI cohort and generate final report   
      i) **setup** working directory    
        - In *r-studio* environment, simply **open** R project `AKI_CDM.Rproj` within the folder
        *OR*    
        - In *plain r* environment, **set working directory** to where `AKI_CDM` locates by runing `setwd("path-to-dir/AKI_CDM")`
            
      ii) **edit** r script `render_report.R` by specifying the following parameters:   
        - `which_report`: which report you want to render (default is `./report/AKI_CDM_EXT_VALID_p1_QA.Rmd`, but there will be more options in the future)   
        - `DBMS_type`: what type of database the current CDM is built on (available options are: `Oracle`(default), `tSQL`)        
        - `driver_type`: what type of database connection driver is available (available options are: `OCI` or `JDBC` for oracle; `JDBC` for sql server)      
        - `start_date`, `end_date`: the start and end date of inclusion period, in `"yyyy-mm-dd"` format (e.g. "2010-01-01", with the quotes)             
      
      iii) **run** *Part I* of r script `render_report.R` after assigning correct values to the parameters in ii)        
      
      iv) **collect and report** all output files from `/output` folder   
        -- a. AKI_CDM_EXT_VALID_p1_QA.html - html report with description, figures and partial tables    
        -- b. AKI_CDM_EXT_VALID_p1_QA_TBL.xlsx - excel with full summary tables    

*Remark*: all the counts (patient, encounter, record) are masked as "<11" if the number is below 11


### Part II: Validate Existing Predictive Models and Retrain Predictive Models with Local Data (Not fully tested yet)

1. Validate the given predictive model trained on KUMC's data   

    i) **download** the predictive model package, "AKI_model_kumc.zip", from the securefile link shared by KUMC. Unzip the file and save everything under `./data/model_kumc` (remark: make sure to save the files under the correct directory, as they will be called later using the corresponding path)  
    
    ii) **continue to run** *Part II.0* of the r script `render_report.R` after completing *Part I*. *Part II.0* will only depend on tables already extracted from *Part I* (saved locally in the folder `./data/raw/...`), no parameter needs to be set up.        
    
    iii) **continue to run** *Part II.1* of the r script `render_report.R` after completing *Part II.0*. *Part II.1* will only depend on tables already extracted from *Part II.0* (saved locally in the folder `./data/preproc/...`), no parameter needs to be set up.     

    iv) **collect and report** the two new output files from `/output` folder           
      -- a. AKI_CDM_EXT_VALID_p2_1_Benchmark.html - html report with description, figures and partial tables       
      -- b. AKI_CDM_EXT_VALID_p2_1_Benchmark_TBL.xlsx - excel with full summary tables          

2. Retrain the model using local data and validate on holdout set 

    i) **download** the data dictionary, "feature_dict.csv", from the securefile link shared by KUMC and save the file under "./ref/" (remark: make sure to save the file under the correct directory, as it will be called later using the corresponding path)   

    ii) **continue to run** (optionally, if already run for Part II.1) *Part II.0* of the r script `render_report.R` after completing *Part I*. *Part II.0* will only depend on tables already extracted from *Part I* (saved locally in the folder `./data/raw/...`), no parameter needs to be set up.        

    iii) **continue to run** *Part II.2* of the r script `render_report.R` after completing *Part I*. *Part II.2* will only depend on tables already extracted from *Part II.0* (saved locally in the folder `./data/preproc/...`), no parameter needs to be set up.     

    iv) **collect and report** the two new output files from `/output` folder           
      -- a. AKI_CDM_EXT_VALID_p2_2_Retrain.html - html report with description, figures and partial tables       
      -- b. AKI_CDM_EXT_VALID_p2_2_Retrain_TBL.xlsx - excel with full summary tables          
      
*Remark: As along as Part I is completed, Part II.1 and Part II.2 can be run independently, based on each site's memory and disk availability.   


Run the `distribution_analysis.R` script to calculate the adjMMD and joint KL-divergence of distribution hetergenity among top important variables of each model. adjMMD is an effective metric which can be used to assess and explain model transportability. 


***

### Benchmarking
a. It takes about **2 ~ 3 hours** to complete Part I (AKI_CDM_EXT_VALID_p1_QA.Rmd). At peak time, it will use about **30 ~ 40GB memory**, especially when large tables like Precribing or Lab tables are loaded in. Total size of output for Part I is about **6MB**.

b. It takes about **60 ~ 70 hours** (6hr/task) to complete Part II.0 (AKI_CDM_EXT_VALID_p2_0_Preprocess.Rmd). At peak time, it will use about **50 ~ 60GB memory**, especially at the preprocessing stage. Total size of intermediate tables and output for Part II.0 is about **2GB**.

c. It takes about **25 ~ 30 hours** to complete Part II.1 (AKI_CDM_EXT_VALID_p2_Benchmark.Rmd). At peak time, it will use about **30 ~ 40GB memory**, especially at the preprocessing stage. Total size of intermediate tables and output for Part II.1 is about **600MB**.

d. It takes about **40 ~ 50 hours** to complete Part II.2 (AKI_CDM_EXT_VALID_p2_Retrain.Rmd). At peak time, it will use about **30 ~ 40GB memory**, especially at the preprocessing stage. Total size of intermediate tables and output for Part II.2 is about **800MB**.

***


## SHAP Value Interpretation
We used [Shapely Additive exPlanations (SHAP)] values to evaluate the marginal effects of the shared top important variables of interests 34. Specifically, the SHAP values evaluated how the logarithmic odds ratio changed by including a factor of certain value for each individual patient. The SHAP values not only captured the global patterns of effects of each factor but also demonstrated the patient-level variations of the effects. We also estimated 95% bootstrapped confidence intervals of SHAP values for each selected feature based on 100 bootstrapped samples. Visit our [SHAP value dashboard] for more details. 

[Shapely Additive exPlanations (SHAP)]: https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf 
[SHAP value dashboard]: https://sxinger.shinyapps.io/AKI_shap_dashbd/

## Adjusted Maxinum Mean Discrepancy (adjMMD)
The [Maximum Mean Discrepancy (MMD)] has been widely used in transfer learning studies for maximizing the similarity among distributions of different domains. Here we modified the classic MMD by taking the missing pattern and feature importance into consideration, which is used to measure the similarities of distributions for the same feature between training and validation sites. Visit our paper [Cross-Site Transportability of an Explainable Artificial Intelligence Model for Acute Kidney Injury Prediction] (DOI:10.1038/s41467-020-19551-w) for more details. 

[Maximum Mean Discrepancy (MMD)]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6751384 

[Cross-Site Transportability of an Explainable Artificial Intelligence Model for Acute Kidney Injury Prediction]: http://www.nature.com/ncomm/10.1038/s41467-020-19551-w

*updated 11/10/2020*


