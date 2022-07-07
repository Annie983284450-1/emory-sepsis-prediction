import os
import pandas as pd
import numpy as np
from itertools import chain
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from time import time
# sep_index = ['BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST',
#              'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine',
#              'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
#              'Bilirubin_total', 'Hct', 'Hgb', 'PTT', 'WBC', 'Platelets']
# con_index = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']

sep_index =  ['hourly_total',
'bicarb_(hco3)', 
'ph',
'partial_pressure_of_carbon_dioxide_(paco2)',
'saturation_of_oxygen_(sao2)',
'partial_pressure_of_oxygen_(pao2)',
'aspartate_aminotransferase_(ast)',
'blood_urea_nitrogen_(bun)',
'calcium',
'chloride',
'creatinine',
'glucose',
# 'lactate_dehydrogenase',
'magnesium',
'potassium',
'bilirubin_total', 
'carboxy_hgb',
'met_hgb',
'partial_prothrombin_time_(ptt)',
'platelets']


con_index = ['pulse', 
             # 'O2Sat', 
             'temperature', # 'Temp',
             'sbp_cuff',# 'SBP', 
             'map_cuff',# 'MAP', 
             'dbp_cuff',# 'DBP', 
             'unassisted_resp_rate',# 'Resp', 
             'end_tidal_co2'# 'EtCO2'
            ]

 
 
# summarize the number of rows with missing values for each column
def summarize_dataset(dataframe):
    for i in range(dataframe.shape[1]):
        # count number of rows with missing values
        n_miss = dataframe[[i]].isnull().sum()
        perc = n_miss / dataframe.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))

# sep_columns = con_index + sep_index
def feature_informative_missingness(case, sep_columns):
    """
    informative missingness features reflecting measurement frequency
        or time interval of raw variables
    differential features, defined by calculating the difference between
        the current record and the previous measurement value
    :param case: one patient's EHR data
    :param sep_columns: selected variables
    :return: calculated features
    """
    # i=0
    # for col in case.columns:
    #     print(col)
    #     print(i)
    #     i=i+1   
    #### 
    temp_data = np.array(case)
    



    for sep_column in sep_columns:
        # print('sep_column:',sep_column)
        sep_data = np.array(case[sep_column])
        nan_pos = np.where(~np.isnan(sep_data))[0]
        # print('nan_pos:',nan_pos)
        # Measurement frequency sequence
        interval_f1 = sep_data.copy()
        # Measurement time interval

        interval_f2 = sep_data.copy()


        ### all columns to indicate the position of the NAN values
           ## if all the values in this column is NAN
        if len(nan_pos) == 0:
            interval_f1[:] = 0
            temp_data = np.column_stack((temp_data, interval_f1))
            interval_f2[:] = -1
            temp_data = np.column_stack((temp_data, interval_f2))
        else:
            interval_f1[: nan_pos[0]] = 0
            for p in range(len(nan_pos)-1):
                interval_f1[nan_pos[p]: nan_pos[p+1]] = p + 1
            interval_f1[nan_pos[-1]:] = len(nan_pos)
            
            temp_data = np.column_stack((temp_data, interval_f1))

            interval_f2[:nan_pos[0]] = -1
            for q in range(len(nan_pos) - 1):
                length = nan_pos[q+1] - nan_pos[q]
                for l in range(length):
                    interval_f2[nan_pos[q] + l] = l

            length = len(case) - nan_pos[-1]
            for l in range(length):
                interval_f2[nan_pos[-1] + l] = l
            
            temp_data = np.column_stack((temp_data, interval_f2))
        # print('interval_f1:', interval_f1 )
        # print('interval_f2:', interval_f2 )



        # Differential features
        ## store the gap between each time interval (i.e., 1hr)

        # diff_f = sep_data.copy()
        # diff_f = diff_f.astype(float)
        # if len(nan_pos) <= 1:
        #     diff_f[:] = np.NaN
        #     temp_data = np.column_stack((temp_data, diff_f))
        # else:
        #     diff_f[:nan_pos[1]] = np.NaN
        #     for p in range(1, len(nan_pos)-1):
        #         diff_f[nan_pos[p] : nan_pos[p+1]] = sep_data[nan_pos[p]] - sep_data[nan_pos[p-1]]
        #     diff_f[nan_pos[-1]:] = sep_data[nan_pos[-1]] - sep_data[nan_pos[-2]]
        #     temp_data = np.column_stack((temp_data, diff_f))

    return temp_data

def feature_slide_window(temp, con_index):
    """
    Calculate dynamic statistics in a six-hour sliding window
    :param temp: data after using a forward-filling strategy, numpy array
    :param con_index: selected variables indexs in con_index
    :return: time-series features
    """

    ##  sepdata: selected features in con_index:
    # 'pulse', 
    #          # 'O2Sat', 
    #          'temperature', # 'Temp',
    #          'sbp_line',# 'SBP', 
    #          'map_line',# 'MAP', 
    #          'dbp_line',# 'DBP', 
    #          'unassisted_resp_rate',# 'Resp', 
    #          'end_tidal_co2'# 'EtCO2'
    sepdata = temp[:, con_index]

    ## create 0s list:
    #e.g., [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]] 
    max_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    min_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    mean_values = [[0 for col in range(len(sepdata))]
                   for row in range(len(con_index))]
    median_values = [[0 for col in range(len(sepdata))]
                     for row in range(len(con_index))]
    std_values = [[0 for col in range(len(sepdata))]
                  for row in range(len(con_index))]
    diff_std_values = [[0 for col in range(len(sepdata))]
                       for row in range(len(con_index))]
    # len(sepdata); num of rows
    for i in range(len(sepdata)):
        ## the first sliding window in the time series for this patient
        if i < 6:
            ## shift the window one hour 
            ## [0:5] -> [1:6]
            win_data = sepdata[0:i + 1]
            for ii in range(6 - i):
                # First array x:
                # [1 2 3 4]
                # Second array y:
                # [5 6 7 8]
                # Row stacked array:
                # [[1 2 3 4]
                # [5 6 7 8]]
                win_data = np.row_stack((win_data, sepdata[i]))
        else:
            win_data = sepdata[i - 6: i + 1]


        ## loop each column in the sliding window
        for j in range(len(con_index)):

            dat = win_data[:, j]
            # if len(np.where(~np.isnan(dat))[0]) == 0:
            # if not np.isnan(np.sum(dat)):
            # if pd.isna(dat):

            # dat_df = pd.DataFrame(dat)
            # print(type(dat_df.values[0]))
            # if j==1: 
            #     print(type(dat[0]))
            #     # print(pd.isna(dat_df).all())
            #     print('=========')
            #     break

            # if all the values in this column is Nan, set all the statisitics to nan
            # if dat_df.isna().all():
            # if dat_df.isnull().all():
            # if not np.isnan(np.sum(dat)):
            # if np.isnan(np.sum(dat)):
            
            ### ********** note: 
            ## np.nan is a float, we have to convert all the elements to float
            ## otherwise will get a type error
            dat = dat.astype('float')

            if len(np.where(~np.isnan(dat))[0]) == 0:
                
                # print('Nan column:', con_index[j])
                max_values[j][i] = np.nan
                min_values[j][i] = np.nan
                mean_values[j][i] = np.nan
                median_values[j][i] = np.nan
                std_values[j][i] = np.nan
                diff_std_values[j][i] = np.nan
            else:
                max_values[j][i] = np.nanmax(dat)
                min_values[j][i] = np.nanmin(dat)
                mean_values[j][i] = np.nanmean(dat)
                median_values[j][i] = np.nanmedian(dat)
                std_values[j][i] = np.nanstd(dat)
                diff_std_values[j][i] = np.std(np.diff(dat))

    win_features = list(chain(max_values, min_values, mean_values,
                              median_values, std_values, diff_std_values))
    win_features = (np.array(win_features)).T

    return win_features


## modified the feature index because we do not have several features in emory dataset
def feature_empiric_score(dat,pulse_index,sbp_index, map_index, resp_index,temp_index,creatinine_index,platelets_index,bilirubin_index):
    """
    empiric features scoring for
    heart rate (HR), systolic blood pressure (SBP), mean arterial pressure (MAP),
    respiration rate (Resp), temperature (Temp), creatinine, platelets and total bilirubin
    according to the scoring systems of NEWS, SOFA and qSOFA
    """


    scores = np.zeros((len(dat), 8))
    for ii in range(len(dat)):
        ## pulse
        HR = dat[ii, pulse_index]
        if HR == np.nan:
            HR_score = np.nan
        elif (HR <= 40) | (HR >= 131):
            HR_score = 3
        elif 111 <= HR <= 130:
            HR_score = 2
        elif (41 <= HR <= 50) | (91 <= HR <= 110):
            HR_score = 1
        else:
            HR_score = 0
        scores[ii, 0] = HR_score

        # Temp = dat[ii, 2]
        Temp = dat[ii, temp_index]
        if Temp == np.nan:
            Temp_score = np.nan
        elif Temp <= 35:
            Temp_score = 3
        elif Temp >= 39.1:
            Temp_score = 2
        elif (35.1 <= Temp <= 36.0) | (38.1 <= Temp <= 39.0):
            Temp_score = 1
        else:
            Temp_score = 0
        scores[ii, 1] = Temp_score

        # Resp = dat[ii, 6]
        #  'unassisted_resp_rate'
        Resp = dat[ii, resp_index]
        if Resp == np.nan:
            Resp_score = np.nan
        elif (Resp < 8) | (Resp > 25):
            Resp_score = 3
        elif 21 <= Resp <= 24:
            Resp_score = 2
        elif 9 <= Resp <= 11:
            Resp_score = 1
        else:
            Resp_score = 0
        scores[ii, 2] = Resp_score
        

        # ''creatinine''
        Creatinine = dat[ii, creatinine_index]
        if Creatinine == np.nan:
            Creatinine_score = np.nan
        elif Creatinine < 1.2:
            Creatinine_score = 0
        elif Creatinine < 2:
            Creatinine_score = 1
        elif Creatinine < 3.5:
            Creatinine_score = 2
        else:
            Creatinine_score = 3
        scores[ii, 3] = Creatinine_score

        ## MAP
        # MAP = dat[ii, 4]
        MAP = dat[ii, map_index]
        if MAP == np.nan:
            MAP_score = np.nan
        elif MAP >= 70:
            MAP_score = 0
        else:
            MAP_score = 1
        scores[ii, 4] = MAP_score
        # 'SBP'
        # SBP = dat[ii, 3]
        SBP = dat[ii, sbp_index]
        # Resp = dat[ii, 6]
        Resp = dat[ii, resp_index]
        if SBP + Resp == np.nan:
            qsofa = np.nan
        elif (SBP <= 100) & (Resp >= 22):
            qsofa = 1
        else:
            qsofa = 0
        scores[ii, 5] = qsofa
        # platelets
        # Platelets = dat[ii, 30]
        Platelets = dat[ii, platelets_index]
        if Platelets == np.nan:
            Platelets_score = np.nan
        elif Platelets <= 50:
            Platelets_score = 3
        elif Platelets <= 100:
            Platelets_score = 2
        elif Platelets <= 150:
            Platelets_score = 1
        else:
            Platelets_score = 0
        scores[ii, 6] = Platelets_score

        # bilirubin_total
        # Bilirubin = dat[ii, 25]
        Bilirubin = dat[ii, bilirubin_index]
        if Bilirubin == np.nan:
            Bilirubin_score = np.nan
        elif Bilirubin < 1.2:
            Bilirubin_score = 0
        elif Bilirubin < 2:
            Bilirubin_score = 1
        elif Bilirubin < 6:
            Bilirubin_score = 2
        else:
            Bilirubin_score = 3
        scores[ii, 7] = Bilirubin_score

    return scores


'''
input: case 
pandas dataframe
data from individual patient
'''
def feature_extraction(case,cols_to_remove):
    labels = np.array(case['sepsis_onset_time'])
    # drop three variables due to their massive missing values
    # pid = case.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'sepsis_onset_time'])


# keep only the columns in df that do not contain string
    case1 = case[[col for col in case.columns if col not in cols_to_remove]]
    # pid = case1.drop(columns=['sepsis_onset_time','pat_id','hospital_discharge_date_time','charttime','hospital_admission_date_time' ,'best_map','procedure','icu','imc','ed'])
    # pid = case1.drop(columns=['sepsis_onset_time','pat_id' , 'best_map','procedure','icu','imc','ed'])
    pid = case1.drop(columns=['sepsis_onset_time','pat_id','csn'])
    pid['stayHrs'] = list(range(0, len(pid)))
    # print(pid.head())
    # for col in pid.columns:
    #     print(col)


    pulse_index = pid.columns.get_loc('pulse')
    sbp_index = pid.columns.get_loc('sbp_cuff')
    map_index = pid.columns.get_loc('map_cuff')
    resp_index = pid.columns.get_loc('unassisted_resp_rate')
    
    # print('Getting informative_missingness ......')
    temp_data = feature_informative_missingness(pid, con_index + sep_index)
    temp = pd.DataFrame(temp_data)
    # Missing values used a forward-filling strategy
    temp = temp.fillna(method='ffill')
    
    

 
    feature_A = np.array(temp)
    # Statistics in a six-hour window for the selected measurements
    # [0, 1, 3, 4, 6] = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    # 30 statistical features in the window
    # feature_B = feature_slide_window(feature_A, [0, 1, 3, 4, 6])

    ## modified because we do not have O2Sat
    temp_index = pid.columns.get_loc('temperature')
    creatinine_index = pid.columns.get_loc('creatinine')
    platelets_index = pid.columns.get_loc('platelets')
    bilirubin_index = pid.columns.get_loc('bilirubin_total')

    # print('Getting sliding window ......')
    feature_B = feature_slide_window(feature_A, [pulse_index,sbp_index, map_index, resp_index])
    
    # print('Getting empirical scores ......')
    # 8 empiric features
    feature_C = feature_empiric_score(feature_A, pulse_index,sbp_index, map_index, resp_index,temp_index,creatinine_index,platelets_index,bilirubin_index)

    #A total of 168 features were obtained
    features = np.column_stack((feature_A, feature_B, feature_C))

    return  features, labels

## added by Annie Zhou on June 28th, 2022 1:05PM
def knn_imputation(X):

    imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean')
    # fit on the dataset
    imputer.fit(X)
    Xtrans = imputer.transform(X)
    return Xtrans
def data_process(data_set, data_path_dir,cols_to_remove):
    """
    Feature matrix across all patients in the data_set
    """
    frames_features = []
    frames_labels = []

    num_psv = 0
    for psv in data_set:
        if num_psv > 3000:
            break
        # print('Start processing', psv,'...........')
        # patient = pd.read_csv(os.path.join(data_path_dir, psv), sep='|')
        patient = pd.read_csv(os.path.join(data_path_dir, psv), sep=',')
        # print('Start feature extraction ......')
        features, labels = feature_extraction(patient,cols_to_remove)
        features = pd.DataFrame(features)
        labels = pd.DataFrame(labels)
        frames_features.append(features)
        frames_labels.append(labels)
        num_psv=num_psv+1
        print('{} preprocessing done!'.format(psv))
    
    print('Concat all the patients data into one numpy array .......')

    concat_start = time()
    dat_features = np.array(pd.concat(frames_features))
    dat_labels = (np.array(pd.concat(frames_labels)))[:, 0]
    print('Time taken to concat all the csv files: {} seconds ###########'.format(time()-concat_start))
    index = [i for i in range(len(dat_labels))]
    np.random.shuffle(index)
    dat_features = dat_features[index]
    dat_labels = dat_labels[index]
    
    print('data processing done!')
    # print(dat_labels)

    return dat_features, dat_labels



## testing 
if __name__ == "__main__":
    data_path =  os.getcwd() + "/merged/"
    train_nosepsis = np.load('train_nosepsis.npy')
    train_sepsis = np.load('train_sepsis.npy')
    # kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed(12306))


#     # for (k, (train0_index, val0_index)), (k, (train1_index, val1_index)) in \
#     #         zip(enumerate(kfold.split(train_nosepsis)), enumerate(kfold.split(train_sepsis))):
#     #     # print(k)
#     #     # print(train0_index[0:4])
#     #     # print(val0_index)

#     #     # training set
#     #     if k==0:
#     #         train_set = np.append(train_nosepsis[train0_index[0:20]], train_sepsis[train1_index[0:4]])
#     #     else:
#     #         break
#     # print(train_set)


#     # frames_features = []
#     # frames_labels = []

#     # ## testing function data_process(data_set, data_path_dir)
#     # data_set =  train_set
#     # data_path_dir = data_path
 
#     # for psv in data_set:
        
   
#         # if n>=3:
#         #     break
#     csv = os.getcwd()+'/merged/201716431667289.csv'
#     patient = pd.read_csv(csv, sep=',')



#     # labels = np.array(patient['sepsis_onset_time'])
#     # # drop three variables due to their massive missing values
#     # # pid = case.drop(columns=['Bilirubin_direct', 'TroponinI', 'Fibrinogen', 'sepsis_onset_time'])
#     # pid = patient.drop(columns=['sepsis_onset_time','pat_id','hospital_discharge_date_time','charttime','hospital_admission_date_time'])
#     # pid['stayHrs'] = list(range(0, len(pid)))
#     # for col in pid.columns:
#     #     print(col)
#     # # print(pid.iloc[0,:])
#     # print(pid.iloc[0].values)

#     # features, labels = feature_extraction(patient)
#     # print('features:',features[0])
 
    dat_features, dat_labels = data_process(train_sepsis[0:1], data_path)
    print(dat_labels)
    print(dat_features)