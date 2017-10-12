import pandas as pd
import numpy as np

from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

import urllib,os,zipfile

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## load data

def read_data_loading_table(filename): 
    data_loading_table = pd.read_csv(filename)
    return(data_loading_table)


def detect_delimiter(infile):
    opin = file(infile,'r')
    for line in opin:
        ### only read the first line
        arr_tab   = len(line.split('\t'))
        arr_comma = len(line.split(','))
        arr_line  = len(line.split('|'))
        seps = [arr_tab,arr_comma,arr_line]
        ind = seps.index(max(seps))
        sep = ['\t',',','|'][ind]
        break
    opin.close()
    return(sep)

def load_raw_data(data_loading_table,sep=None,
                  na_values=[' '],keep_default_na=True):
    raw_data = {}
    for ind, row in data_loading_table.iterrows():
        file_name,table_name = row.file_name, row.table_name
        if sep is None:
            sep = detect_delimiter(file_name)
        print ind+1, 'loading',table_name,
        raw_data[table_name] = pd.read_csv(file_name,sep=sep,na_values=na_values)
        print ' done',raw_data[table_name].shape
    return(raw_data)


#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## check data

def check_data(infile):
    pass
    ### check format: time
    ### check encoding


def detect_extra_new_line_character(infile,sep=None,outfile=None):
    if sep is None: sep = detect_delimiter(infile)
    opin = file(infile, 'r')
    i = 0
    strings = []
    nsplits_arr = []
    for line in opin:
        i = i + 1
        arr = line.split(sep)
        nsplits = len(arr)
        ### read the header line
        ### assume the splits of the header line are correct
        if i == 1:
            header_nsplits = nsplits
            strings.append(line)
            nsplits_arr.append(nsplits)
            continue
        ### if the nsplits are correct, save the line
        if nsplits == header_nsplits:
            strings.append(line)
            nsplits_arr.append(nsplits)
            continue
        ### if the current nsplits are not correct
        else:
            print 'find incorrect splitted line =', i,'nsplits =',nsplits,'header splits =',header_nsplits 
            ### check the previous nsplits
            ### if the previous nsplits are correct, just save the current line
            if nsplits_arr[-1] == header_nsplits:
                print "##"*40
                print line
                print "##"*40
                strings.append(line)
                nsplits_arr.append(nsplits)
                continue
            ### if the previous nsplits is not correct
            else:
                if nsplits_arr[-1] + (nsplits-1) == header_nsplits:
                    ### there will be extra split, so use nsplits-1
                    print "##"*10
                    print "## first part, line",i-1
                    print strings[-1]
                    ### remove the previous line and concat the current line
                    strings.append(strings.pop().rstrip('\n')+line)
                    nsplits_arr.append(header_nsplits)
                    print "## second part, line",i
                    print line
                    print "## merged"
                    print strings[-1]
                    print "##"*40
                    continue
                else:
                    print "cound not find matched line"
                    print "##"*40
                    print "\n\n"
                    strings.append(line)
                    nsplits_arr.append(nsplits)
                    continue
    opin.close()
    
    if outfile is not None:
        opt = file(outfile, 'w')
        opt.write('%s' % ("").join(strings))
        opt.close()
    
    return(None)

#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## prepare data for loading






    
    
    
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## detect variable type

def extract_features(var_col):
    #### remove the spaces
    if var_col.dtype == 'object': var_col = var_col.str.strip()
    ####
    
    sample_rows = 10000
    if var_col.size > sample_rows:
        var_col = var_col.sample(n=sample_rows,random_state=1)
        #var_col_sample = var_col.sample(n=sample_rows,random_state=1)
        ### sampled rows are all null values
        #if var_col_sample.isnull().sum() == sample_rows:
        #    pass
        #else:
        #    var_col = var_col_sample
        
    features = {}
    features['dtype_object'] = int(var_col.dtype == 'object')
    features['dtype_float']  = int(var_col.dtype == 'float')
    features['dtype_int']    = int(var_col.dtype == 'int')
    features['n'] = var_col.size    
    features['n_level'] = var_col.nunique()
    features['n_level_per'] = features['n_level']*1.0 / features['n']
    features['n_missing'] = var_col.isnull().sum()
    features['n_missing_per'] = features['n_missing']*1.0 / features['n']
    ####
    ## !!!!!!!!!!!!!!!!!!!!!!
    ## need to replace Nan with empty string, otherwise it will be counted as length 3 string
    string = var_col.fillna('').astype('string').str
    ####
    #string_len = var_col[var_col.notnull()].sample(n=1000,random_state=1).astype('string').str.len()
    string_len = string.len()
    features['min_length'] = string_len.min()
    features['max_length'] = string_len.max()
    features['mean_length'] = string_len.mean()
    features['std_length'] = string_len.std()
    features['min_length_exclude_missing'] = string_len[string_len>0].min()
    features['max_length_exclude_missing'] = string_len[string_len>0].max()
    features['mean_length_exclude_missing'] = string_len[string_len>0].mean()
    features['std_length_exclude_missing'] = string_len[string_len>0].std()
    #####
    
    dash_num = string.count('-')
    features['n_dash_count'] = (dash_num>0).sum()
    features['all_have_dash'] = int((features['n_dash_count']>0) and ((features['n_dash_count'] + features['n_missing']) == features['n']))
    features['min_dash_count'] = dash_num.min()
    features['max_dash_count'] = dash_num.max()
    features['mean_dash_count']= dash_num.mean()
    features['std_dash_count'] = dash_num.std()
    ## if all Nans, 
    features['min_dash_count_exclude_missing'] = dash_num[string_len>0].min()
    features['max_dash_count_exclude_missing'] = dash_num[string_len>0].max()
    features['mean_dash_count_exclude_missing']= dash_num[string_len>0].mean()
    features['std_dash_count_exclude_missing'] = dash_num[string_len>0].std()
    #####backslash
    backslash_num = string.count('/')
    features['n_backslash_count'] = (backslash_num>0).sum()
    features['all_have_backslash'] = int((features['n_backslash_count']>0) and ((features['n_backslash_count'] + features['n_missing']) == features['n']))
    features['min_backslash_count'] = backslash_num.min()
    features['max_backslash_count'] = backslash_num.max()
    features['mean_backslash_count']= backslash_num.mean()
    features['std_backslash_count'] = backslash_num.std()
    features['min_backslash_count_exclude_missing'] = backslash_num[string_len>0].min()
    features['max_backslash_count_exclude_missing'] = backslash_num[string_len>0].max()
    features['mean_backslash_count_exclude_missing']= backslash_num[string_len>0].mean()
    features['std_backslash_count_exclude_missing'] = backslash_num[string_len>0].std()
    
    del(string_len)
    del(dash_num)
    del(backslash_num)
    #features['n_space'] = string_len.std()
    
    features = pd.Series(features)
    features = features.fillna(0)
    return(features)



def predict_var_type(var_col):
    ###
    clf = joblib.load('variable_type_detector.pkl')
    ###
    features = extract_features(var_col)
    features = features.values.reshape(1, -1)
    pred = clf.predict(features)
    ###
    var_type_prediction = ['categorical', 'date', 'numeric', 'text'][np.argmax(pred)]
    return(var_type_prediction)



#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## summary 


def generate_table_summary(raw_data):
    #table_summary = pd.DataFrame(columns=['table_name','n_samples','n_variables'])
    table_name  = []
    n_samples   = []
    n_variables = []
    for tn in raw_data.keys():
        table_name.append(tn)
        n_samples.append(raw_data[tn].shape[0])
        n_variables.append(raw_data[tn].shape[1])
    table_summary = pd.DataFrame({
                  'table_name':table_name,
                  'n_samples':n_samples,
                  'n_variables':n_variables},
    columns=['table_name','n_samples','n_variables'])
    table_summary = table_summary.sort_values(by=['n_samples'],ascending=False).reset_index(drop=True)
    return(table_summary)



def generate_var_summary(raw_data,table_summary=None,infer_var_type=True):
    ##############################################
    ### init the var_summary
    if table_summary is None:
        table_summary = generate_table_summary(raw_data)
    table_names = []
    var_names   = []
    var_types   = []
    n_samples   = []
    n_levels    = []
    n_missing   = []
    predicted_var_type = []
    ###
    column_list = ['min','max','mean','median','std',
                  'earliest_date','latest_date',
                  'largest_category','largest_category_count','largest_category_percent']
    df = pd.DataFrame(index=np.arange(table_summary.n_variables.sum()),columns=column_list)
    ###
    i = -1
    for ind, row in table_summary.iterrows():
        print 'processing ->',row.table_name
        ### use arr.extend(), faster!
        table_names = table_names + [row.table_name] * row.n_variables
        var_names = var_names + raw_data[row.table_name].columns.tolist()
        var_types = var_types + raw_data[row.table_name].dtypes.tolist()
        n_samples = n_samples + [row.n_samples] * row.n_variables
        n_levels  = n_levels + raw_data[row.table_name].apply(pd.Series.nunique).values.tolist()
        n_missing = n_missing + raw_data[row.table_name].isnull().sum().tolist()
        
        ## predict var type
        ## the if statement is necessary when changing the feature extraction function and re-traning the model
        if infer_var_type:
            for var_ind, var_col in enumerate(raw_data[row.table_name].columns):
                i = i + 1
                vtype = predict_var_type(raw_data[row.table_name][var_col])
                predicted_var_type.append(vtype)
                if vtype == 'numeric':
                    df.loc[i,'min'] = raw_data[row.table_name][var_col].min()
                    df.loc[i,'max'] = raw_data[row.table_name][var_col].max()
                    df.loc[i,'mean'] = raw_data[row.table_name][var_col].mean()
                    df.loc[i,'std'] = raw_data[row.table_name][var_col].std()
                    df.loc[i,'median'] = raw_data[row.table_name][var_col].median()
                if vtype == 'date':
                    #print row.table_name, var_col
                    df_time = (pd.to_datetime(raw_data[row.table_name][var_col],infer_datetime_format=True))
                    df.loc[i,'earliest_date'] = df_time.min()
                    df.loc[i,'latest_date'] = df_time.max()
                    del df_time
                if vtype == 'categorical' or vtype == 'text':
                    freq = raw_data[row.table_name][var_col].value_counts()
                    df.loc[i,'largest_category'] = freq.idxmax()
                    df.loc[i,'largest_category_count'] = freq[df.loc[i,'largest_category']]
                    del freq
              
        else:
            predicted_var_type = predicted_var_type + [np.nan]*row.n_variables
            
        #from IPython.core.debugger import Tracer; Tracer()() 
    ####
    var_summary = pd.DataFrame({
        'table_name':table_names,
        'var_name':var_names,
        'var_type':var_types,
        'predicted_var_type':predicted_var_type,
        'n_samples':n_samples,
        'n_levels':n_levels,
        'n_missing':n_missing
    })
    if infer_var_type:
        var_summary = pd.concat([var_summary,df], axis=1)
    var_summary['missing_percent'] = var_summary['n_missing'] / var_summary['n_samples'] 
    var_summary['largest_category_percent'] = var_summary['largest_category_count'] / var_summary['n_samples'] 
    ################################################
    ### correct the var type
    
    ################################################
    col_ordered = ['table_name','var_name','var_type','predicted_var_type',
                   'n_samples','n_levels','n_missing','missing_percent'] + column_list
    var_summary = var_summary[col_ordered]
    return(var_summary)


def find_common_variables(raw_data):
    all_vars = raw_data.keys()
    n = len(all_vars)
    common_vars = pd.DataFrame(index=np.arange(n*(n-1)/2),columns=['table1','table2','common_vars'])
    ind = -1
    for i in range(n):
        for j in range(i+1,n):
            #if i==j:continue
            #print i,j
            ind = ind + 1
            table1 = all_vars[i]
            table2 = all_vars[j]
            common_vars.loc[ind,'table1'] = table1
            common_vars.loc[ind,'table2'] = table2
            common_vars.loc[ind,'common_vars'] = (' | ').join(list(set(raw_data[table1].columns).intersection(set(raw_data[table2].columns))))
    return(common_vars)
  
    
    
def generate_var_disbution(raw_data,outfile, table_summary=None,infer_var_type=True):
    writer = pd.ExcelWriter(outfile)
    #df1.to_excel(writer,'Sheet1')
    #df2.to_excel(writer,'Sheet2')
    ##############################################
    ### init the var_summary
    if table_summary is None:
        table_summary = generate_table_summary(raw_data)
    ##############################################
    predicted_var_type = []
    i = 0
    for ind, row in table_summary.iterrows():
        if infer_var_type:
            for var_ind, var_col in enumerate(raw_data[row.table_name].columns):
                i = i + 1
                print i,row.table_name,var_col
                vtype = predict_var_type(raw_data[row.table_name][var_col])
                predicted_var_type.append(vtype)
                if vtype == 'categorical':
                    var_dist = pd.DataFrame(raw_data[row.table_name][var_col].value_counts(dropna=False)).reset_index()
                    var_dist.columns = [row.table_name+'__'+var_col,'Freq']
                    var_dist['Percent'] = var_dist['Freq'] /  var_dist['Freq'].sum()
                    pd.DataFrame(var_dist).to_excel(writer,(row.table_name+'__'+var_col)[0:30])
                    #break
                if vtype == 'date':
                    pass
    writer.save()
    return(None)    
    

    
def get_ICD_table():
    
    
    #import urllib2
    #import requests
    ### ICD10
    ### https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html
    ### https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-Code-Descriptions.zip
    ### https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-PCS-Order-File.zip
    ### ICD9
    ### https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html
    ### https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip
    
    if not os.path.exists('./download_icd_code_table/'):
        os.makedirs('./download_icd_code_table/')
    
    url = 'https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-Code-Descriptions.zip'
    download_file = urllib.urlretrieve(url, "./download_icd_code_table/icd10_code.zip")[0]
    
    zip_ref = zipfile.ZipFile(download_file,'r')
    zip_ref.extractall('./download_icd_code_table/')
    zip_ref.close()
    
    url = 'https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-PCS-Order-File.zip'
    download_file = urllib.urlretrieve(url, "./download_icd_code_table/icd10_code.zip")[0]
    
    zip_ref = zipfile.ZipFile(download_file,'r')
    zip_ref.extractall('./download_icd_code_table/')
    zip_ref.close()
    
        
    url = 'https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip'
    download_file = urllib.urlretrieve(url, "./download_icd_code_table/icd10_code.zip")[0]
    
    zip_ref = zipfile.ZipFile(download_file,'r')
    zip_ref.extractall('./download_icd_code_table/')
    zip_ref.close()
    
    #ICD-10-CM (diagnosis) and ICD-10-PCS (procedure)
    #icd10cm_order_2018.txt
    #icd10pcs_order_2018.txt
    
    #ICD-9-CM Diagnosis and Procedure Codes
    #CMS32_DESC_LONG_SHORT_DX.xlsx
    #CMS32_DESC_LONG_SHORT_SG.xlsx
    
    icd10_diagnosis = pd.read_fwf('./download_icd_code_table/icd10cm_order_2018.txt',header=None,names=['index','ICD_code','flag','short_name','long_name'])
    icd10_diagnosis['type'] = 'ICD10_diagnosis'
    icd10_diagnosis = icd10_diagnosis[['ICD_code','short_name','long_name']]
    icd10_procedure = pd.read_fwf('./download_icd_code_table/icd10pcs_order_2018.txt',header=None,names=['index','ICD_code','flag','short_name','long_name'])
    icd10_procedure['type'] = 'ICD10_procedure'
    icd10_procedure = icd10_procedure[['ICD_code','short_name','long_name']]
    
    icd9_diagnosis = pd.read_excel('./download_icd_code_table/CMS32_DESC_LONG_SHORT_DX.xlsx',headr=0,names=['ICD_code','short_name','long_name'])
    icd9_diagnosis['type'] = 'ICD9_diagnosis'
    icd9_procedure = pd.read_fwf('./download_icd_code_table/CMS32_DESC_LONG_SHORT_SG.xlsx',headr=0,names=['ICD_code','short_name','long_name'])
    icd9_procedure['type'] = 'ICD9_procedure'
    
    icd_table = pd.concat([icd10_diagnosis,icd10_procedure,icd9_diagnosis,icd9_procedure],axis=0,ignore_index=True)
    
    return(icd_table)
#####################################################################################################
#####################################################################################################
#####################################################################################################
#####################################################################################################
## Feature engineering

def generate_dummy_vars_for_train_and_test(train, test, cat_vars):
    ###############
    categories={} 
    for f in cat_vars:
        train[f] = train[f].astype('category')
        categories[f] = train[f].cat.categories
    train_dummy = pd.get_dummies(train,columns=cat_vars)
    ###############
    for f in cat_vars:
        test[f] = test[f].astype('category')    
        test[f].cat.set_categories(categories[f],inplace=True)
    test_dummy = pd.get_dummies(test,columns = cat_vars)
    ####
    return(train_dummy,test_dummy)


### ICD10
### https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs.html
### https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2018-ICD-10-Code-Descriptions.zip
### ICD9
### https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/codes.html
### https://www.cms.gov/Medicare/Coding/ICD9ProviderDiagnosticCodes/Downloads/ICD-9-CM-v32-master-descriptions.zip