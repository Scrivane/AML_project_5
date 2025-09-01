###always copy this for logging ::::::::::::::::::::::::::::::::::::::::::::::::::
import os, csv,json
from datetime import datetime
import pandas as pd


    
def log_results(name, rnd,maxround, val_loss, val_acc, test_loss, test_acc, train_loss, train_acc,local_train_mean,local_train_std, csv_path='results_log.csv',csv_path_final='global_results.csv',params={}):
        file_exists = os.path.exists(csv_path)
        if not file_exists:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name','round', 'val_loss', 'val_acc', 'test_loss', 'test_acc','train_loss', 'train_acc','local_train_mean','local_train_std'])
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                name,
                rnd,
                f"{val_loss:.4f}", f"{val_acc:.4f}",
                f"{test_loss:.4f}", f"{test_acc:.4f}",
                 f"{train_loss:.4f}",  f"{train_acc:.4f}",
                 f"{local_train_mean:.4f}",  f"{local_train_std:.4f}",

            ])
        

        if rnd==maxround:
                csv_path_res='results_only_'+name+'.csv'
                clean_results_history(csv_path,csv_path_res,name)
                write_final_results(name,params,csv_path_res,csv_path_final)

def clean_results_history(results_file_name,new_file_name,name):   #da fare eliminare righe che vengono prima della successiva 
    input_file = results_file_name
    output_file=new_file_name


    # Read and clean lines
    with open(input_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    filtered = []
    last_seen_index = float('inf')  # Start with a very large number
    header = lines[0]
    data_lines = lines[1:]

    # Iterate in reverse
    for line in reversed(data_lines):
        current_index = int(line.split(',')[1])
        if current_index < last_seen_index and (line.split(',')[0]==name):
            filtered.append(line)
            last_seen_index = current_index
        else:
            # Skip this line, as its index is higher than the next one
            pass

    # Reverse again to restore original order (except removed lines)
    filtered.reverse()

    file_exists = os.path.exists(output_file)
    # Write to output


    if os.path.exists(new_file_name):
        with open(new_file_name, "r") as f:
            history_lines = [line.strip() for line in f if line.strip()]
        history_header = history_lines[0]
        history_data = history_lines[1:]
    else:
        history_header = header
        history_data = []

    # --- Remove from history any round that will be updated ---
    new_rounds_set = {int(line.split(',')[1]) for line in filtered}
    updated_history_data = [
        line for line in history_data
        if int(line.split(',')[1]) not in new_rounds_set
    ]

    # --- Merge history and new data ---
    merged_data = updated_history_data + filtered

    # --- Write back ---
    with open(new_file_name, "w") as f:
        f.write(history_header + "\n")
        f.write("\n".join(merged_data))
        f.write("\n")

    print(f"Filtered and merged results written to {new_file_name}")

    """ if file_exists:
        with open(output_file, "a") as f:
            f.write('\n'.join(filtered) )


    else:

        with open(output_file, "w") as f:
            f.write(header+"\n")

            f.write('\n'.join(filtered))
            f.write('\n')

    
    

    

    print(f"Filtered results written to {output_file}") """



def get_results(csv_path):
    df = pd.read_csv(csv_path)

    results = {}

    for split in ['train', 'val', 'test']:
        acc_col = f"{split}_acc"
        loss_col = f"{split}_loss"

        max_acc = df[acc_col].max()
        max_idx = df[acc_col].idxmax()

        max_round = df.loc[max_idx, 'round']
        loss_at_max = df.loc[max_idx, loss_col]

        results[split] = {
            'max_acc': max_acc,
            'round': int(max_round),
            'loss_at_max': loss_at_max
        }

    return results





def write_final_results(name, params, csv_path='results_log.csv', results_csv_path='global_results.csv'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    results = get_results(csv_path)

    row = {
        'timestamp': timestamp,
        'model_name': name,
        'parameters': json.dumps(params),
        'train_max_acc': results['train']['max_acc'],
        'train_round': results['train']['round'],
        'train_loss': results['train']['loss_at_max'],
        'val_max_acc': results['val']['max_acc'],
        'val_round': results['val']['round'],
        'val_loss': results['val']['loss_at_max'],
        'test_max_acc': results['test']['max_acc'],
        'test_round': results['test']['round'],
        'test_loss': results['test']['loss_at_max'],
    }

    file_exists = os.path.exists(results_csv_path)

    with open(results_csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


df = pd.read_csv("shakespeare_with_name.csv")
listnames=set(df["name"])
csv_path="shakespeare_with_name.csv"
for name in listnames:
    params={}



    csv_path_res='results_only_'+name+'.csv'
    csv_path_final='global_results.csv'
    clean_results_history(csv_path,csv_path_res,name)
    write_final_results(name,params,csv_path_res,csv_path_final)
