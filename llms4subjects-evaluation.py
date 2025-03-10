import os
import json
import pandas as pd

def read_json_file(filepath: str, encoding: str = 'utf-8'):
    """
    Reads the JSON file and return it's content

    :param str file_path: The path to the JSON file
    :param str encoding: The encoder to use for reading the file
    
    :returns json_dict: The JSON file content
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print('Cannot parse JSON file: {}'.format(filepath))
    except FileNotFoundError:
        print('File Not Found: {}'.format(filepath))
    except Exception as e:
        print('Exception Occured: {}'.format(e))
        
def precision(true_labels: list, pred_labels: list, k: int):
    """
    Calculates the precision@k for the given true labels and the predicted labels.
    
    Args:
        true_labels (list): The list of true labels
        pred_labels (list): The list of predicted labels
        k (int): The value of K representing the top k values to consider

    Returns:
        float: Precision@k
    """
    true_set = set(true_labels)
    pred_set = set(pred_labels[:k])
    intersection = true_set & pred_set
    return round(len(intersection) / k, 4)

def recall(true_labels: list, pred_labels: list, k: int):
    """
    Calculates the recall@k for the given true labels and the predicted labels.

    Args:
        true_labels (list): The list of true labels
        pred_labels (list): The list of predicted labels
        k (int): The value of K representing the top k values to consider

    Returns:
        float: Recall@k
    """
    # 将true_labels转换为集合
    true_set = set(true_labels)
    # 取出pred_labels的前k个元素，转换为集合
    pred_set = set(pred_labels[:k])
    # 计算true_set和pred_set的交集
    intersection = true_set & pred_set
    # 返回交集的长度除以true_set的长度，保留四位小数
    return round(len(intersection) / len(true_set), 4)

def f1(precision_k: float, recall_k: float):
    """
    Calculates the f1@k for the given precision@k and recall@k.

    Args:
        precision_k (float): The value of Precision@k
        recall_k (float): The value of Recall@k

    Returns:
        float: f1@k
    """
    if precision_k + recall_k == 0: return 0
    return round(2 * (precision_k * recall_k) / (precision_k + recall_k), 4)
    
def evaluate_combined_record_type_language(true_dict: dict, predicted_dict: dict, k: int):
    """
    Calculates the evaluation metrics (Precision@k, Recall@k and F1@k) on the combined granularity level (record type and language) 
    for the given true GND labels and predicted GND labels.

    Args:
        true_dict (dict): The dictionary containing the list of true GND labels
        predicted_dict (dict): The dictionary containing the list of predicted GND labels
        k (int): The value of K for calculating precision and recall

    Returns:
        dict: The resulted dictionary containing the evaluation metrics score
    """
    
    #Dictionary to store evaluation metric scores
    combined_metrics = {}
    
    #Iterating over each record type and language combination
    for record_type, lang_dict in true_dict.items():
        for language, file_data in lang_dict.items():
            
            #Aggregating the recall and precision for each record type and language combination
            total_recall, total_precision = 0, 0
            
            #Total files for each combination
            count = len(file_data.keys())
            
            #Iterating over each file containing the true GND labels 
            for filename, true_labels in file_data.items():
                #Extracting the corresponding predicted GND labels
                # print(record_type, language, predicted_dict[record_type][language])
                # if filename=='3A77074267X.json':
                #     print(record_type, language, filename)
                pred_labels = predicted_dict[record_type][language][filename]
                
                #Calculating the recall and precision at k
                recall_k = recall(true_labels, pred_labels, k)
                precision_k = precision(true_labels, pred_labels, k)
                
                total_recall += recall_k
                total_precision += precision_k
            
            #Averaging recall and precision and calculating the f1 score  
            avg_recall = total_recall / count if count else 0.0
            avg_precision = total_precision / count if count else 0.0
            avg_f1 = f1(avg_recall, avg_precision)
            
            #Saving the metrics score in the dictionary
            if record_type not in combined_metrics:
                combined_metrics[record_type] = {}
            combined_metrics[record_type][language] = {f'precision_{k}': avg_precision, f'recall_{k}': avg_recall, f'f1_{k}': avg_f1}
    
    return combined_metrics

def evaluate_record_type_level(true_dict: dict, predicted_dict: dict, k: int):
    """
    Calculates the evaluation metrics (Precision@k, Recall@k and F1@k) on the record type granularity level for the given true 
    GND labels and predicted GND labels.

    Args:
        true_dict (dict): The dictionary containing the list of true GND labels
        predicted_dict (dict): The dictionary containing the list of predicted GND labels
        k (int): The value of K for calculating precision and recall

    Returns:
        dict: The resulted dictionary containing the evaluation metrics score
    """
    
    #Dictionary to store evaluation metric scores
    metrics_score = {}
    
    #Iterating over each record type and language combination
    for record_type, lang_dict in true_dict.items():
        for language, file_data in lang_dict.items():
            
            #Aggregating the recall and precision for each record type
            if record_type not in metrics_score:
                metrics_score[record_type] = {f'precision_{k}': 0, f'recall_{k}': 0, f'f1_{k}': 0, 'total_files': 0}
            
            #Total files
            metrics_score[record_type]['total_files'] += len(file_data.keys())
            
            #Iterating over each file containing the true GND labels 
            for filename, true_labels in file_data.items():
                #Extracting the corresponding predicted GND labels
                pred_labels = predicted_dict[record_type][language][filename]
                
                #Calculating the recall and precision at k
                recall_k = recall(true_labels, pred_labels, k)
                precision_k = precision(true_labels, pred_labels, k)
                
                metrics_score[record_type][f'recall_{k}'] += recall_k
                metrics_score[record_type][f'precision_{k}'] += precision_k
            
    #Averaging recall and precision and calculating the f1 score
    for record_type, metrics in metrics_score.items():
        total_files = metrics['total_files']
        metrics[f'recall_{k}'] = metrics[f'recall_{k}'] / total_files if total_files else 0.0
        metrics[f'precision_{k}'] = metrics[f'precision_{k}'] / total_files if total_files else 0.0
        metrics[f'f1_{k}'] = f1(metrics[f'recall_{k}'], metrics[f'precision_{k}'])
        
        #Deleting the total files key and value
        del metrics['total_files']
    
    return metrics_score

def evaluate_language_level(true_dict: dict, predicted_dict: dict, k: int):
    """
    Calculates the evaluation metrics (Precision@k, Recall@k and F1@k) on the language granularity level for the given true 
    GND labels and predicted GND labels.

    Args:
        true_dict (dict): The dictionary containing the list of true GND labels
        predicted_dict (dict): The dictionary containing the list of predicted GND labels
        k (int): The value of K for calculating precision and recall

    Returns:
        dict: The resulted dictionary containing the evaluation metrics score
    """
    
    #Dictionary to store evaluation metric scores
    metrics_score = {}
    
    #Iterating over each record type and language combination
    for record_type, lang_dict in true_dict.items():
        for language, file_data in lang_dict.items():
            
            #Aggregating the recall and precision for each record type
            if language not in metrics_score:
                metrics_score[language] = {f'precision_{k}': 0, f'recall_{k}': 0, f'f1_{k}': 0, 'total_files': 0}
            
            #Total files
            metrics_score[language]['total_files'] += len(file_data.keys())
            
            #Iterating over each file containing the true GND labels 
            for filename, true_labels in file_data.items():
                #Extracting the corresponding predicted GND labels
                pred_labels = predicted_dict[record_type][language][filename]
                
                #Calculating the recall and precision at k
                recall_k = recall(true_labels, pred_labels, k)
                precision_k = precision(true_labels, pred_labels, k)
                
                metrics_score[language][f'recall_{k}'] += recall_k
                metrics_score[language][f'precision_{k}'] += precision_k
            
    #Averaging recall and precision and calculating the f1 score
    for language, metrics in metrics_score.items():
        total_files = metrics['total_files']
        metrics[f'recall_{k}'] = metrics[f'recall_{k}'] / total_files if total_files else 0.0
        metrics[f'precision_{k}'] = metrics[f'precision_{k}'] / total_files if total_files else 0.0
        metrics[f'f1_{k}'] = f1(metrics[f'recall_{k}'], metrics[f'precision_{k}'])
        
        #Deleting the total files key and value
        del metrics['total_files']
    
    return metrics_score

def evaluate_and_save_to_excel(dir_path: str, filename: str, true_dict: dict, predicted_dict: dict, list_k: list):
    """
    Calculate the evaluation metrics at each granularity level for the given true labels and predicted labels and save the results
    in an excel file.

    Args:
        dir_path (str): The path to save the excel file
        filename (str): The file name for the excel file
        true_dict (dict): The dictionary containing the list of true GND labels
        predicted_dict (dict): The dictionary containing the list of predicted GND labels
        list_k (list): The list of values of k
    """
    
    #List of dataframes for each granularity level
    combined_df_list, record_type_df_list, language_df_list = [], [], []
    
    #Calculating the evaluation metrics on each value of k
    for k in list_k:
        
        print(f'\nEvaluating GND Subject Codes -- Granularity Level: Combined Language and Record-levels and k: {k}')
        combined_metrics_score = evaluate_combined_record_type_language(true_dict, predicted_dict, k)
        
        #Converting the nested dictionary into the dataframe
        rows = []
        for category, langs in combined_metrics_score.items():
            for lang, metrics in langs.items():
                row = {'Record Type': category, 'Language': lang}
                row.update(metrics)
                rows.append(row)
        combined_df_list.append(pd.DataFrame(rows))

        print(f'Evaluating GND Subject Codes -- Granularity Level: Record Type level and k: {k}')
        record_type_metrics_score = evaluate_record_type_level(true_dict, predicted_dict, k)
        record_type_df_list.append(pd.DataFrame.from_dict(record_type_metrics_score, orient='index'))

        print(f'Evaluating GND Subject Codes -- Granularity Level: Language level and k: {k}')
        language_metrics_score = evaluate_language_level(true_dict, predicted_dict, k)
        language_df_list.append(pd.DataFrame.from_dict(language_metrics_score, orient='index'))
    
    # Concatenate all the DataFrames
    final_combined_df = pd.concat(combined_df_list, axis=1)
    final_combined_df = final_combined_df.loc[:, ~final_combined_df.columns.duplicated()]
    final_combined_df.set_index(['Record Type', 'Language'])
    
    final_record_type_df = pd.concat(record_type_df_list, axis=1)
    final_record_type_df.reset_index(inplace=True)
    final_record_type_df.rename(columns={'index': 'Record Type'}, inplace=True)
    
    final_language_df = pd.concat(language_df_list, axis=1)
    final_language_df.reset_index(inplace=True)
    final_language_df.rename(columns={'index': 'Language'}, inplace=True)
    
    #Saving the results in an excel file
    os.makedirs(dir_path, exist_ok=True)
    with pd.ExcelWriter(f'{dir_path}/{filename}') as writer:
        final_combined_df.to_excel(writer, sheet_name="Record Type and Language", index=False)
        final_record_type_df.to_excel(writer, sheet_name="Record Type", index=False)
        final_language_df.to_excel(writer, sheet_name="Language", index=False)
 
def read_gnd_files(dir_path: str, true_labels: bool):
    """
    Reads the Files containing the TIBKAT records in the JSON format

    Args:
        dir_path (str): The path containing the records
        true_labels (bool): Does the directory contains the true GND labels?

    Returns:
        dict: The dictionary containing the subjects information
    """
    gnd_labels = {'Article': {'de': {}, 'en': {}}, 'Book': {'de': {}, 'en': {}}, 'Conference': {'de': {}, 'en': {}}, 
                   'Report': {'de': {}, 'en': {}}, 'Thesis': {'de': {}, 'en': {}}}
    # gnd_labels = {'Article': { 'en': {}}, 'Book': {'en': {}}, 'Conference': {'en': {}}, 
    #                'Report': {'en': {}}, 'Thesis': {'en': {}}}
    #Iterating over each files in the directory
    for root, _, filenames in os.walk(dir_path):
        if not filenames: continue
        
        #Extracting the record type and language from the root path
        record_type, language = root.replace('\\', '/').split('/')[-2:]
        if language not in ['en','de']: continue
        for fname in filenames:
            gnd_codes = read_json_file(f'{root}/{fname}')
            fname=fname.replace('.jsonld', '.json')
            if not gnd_codes: continue
            if true_labels and 'dcterms:subject' not in gnd_codes['@graph'][-1]: continue
            if not true_labels and 'dcterms:subject' not in gnd_codes: continue
            
            if true_labels:
                dc_subjects = gnd_codes['@graph'][-1]['dcterms:subject']
                dc_subjects = [dc_subjects['@id']] if isinstance(dc_subjects, dict) else [code['@id'] for code in dc_subjects]
            gnd_labels[record_type][language][fname] = dc_subjects if true_labels else gnd_codes['dcterms:subject']
            
    return gnd_labels

if __name__ == "__main__":
    
    print('\nLLMs4Subjects Shared Task -- Evaluations')
    
    print('\nPlease specify the directory containing the true GND labels')
    true_labels_dir = '/media/jh/新加卷1/2024_11_10/llms4subjects-main/shared-task-datasets/TIBKAT/all-subjects/data/dev'#input('Directory path> ')
    
    print('\nPlease specify the directory containing the predicted GND labels')
    # pred_labels_dir = '/media/jh/新加卷/2024_12_04/test2'#input('Directory path> ')
    
    print('\nPlease specify the directory to save the evaluation metrics')
    results_dir = './test'#input('Directory path> ')
    
    true_dict = read_gnd_files(true_labels_dir, True)

    def evaluate1(results_dir, filename, true_dict):
        pred_labels_dir=f'/media/jh/新加卷1/2024_12_04/2025/{filename}'
        print('Reading the Predicted GND labels...')
        predicted_dict = read_gnd_files(pred_labels_dir, False)
        list_k = [k for k in range(5, 55, 5)]
        evaluate_and_save_to_excel(results_dir, f'{filename}.xlsx', true_dict, predicted_dict, list_k)

    f_list=['test6-4']
    for filename in f_list:
        evaluate1(results_dir, filename, true_dict)
    print('\nEvaluation Completed!')
