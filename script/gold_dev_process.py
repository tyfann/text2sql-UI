import json

import os

if __name__=='__main__':
    os.chdir('..')

    dev_path = './data/CSgSQL_sp/dev.json'

    with open(dev_path) as f:
        dev_data = json.load(f)
    
    with open('./data/CSgSQL_sp/gold_dev.sql', 'w') as f:
        for index, data in enumerate(dev_data):
            # print(data['question_id'],'\t',data['query'],'\t',data['db_id'],file = f)
            # import pdb; pdb.set_trace()
            new_data_list = [data['question_id'], data['query'], data['db_id']]

            new_data = '\t'.join(new_data_list)
            f.write(new_data)
            if index != len(dev_data) - 1:
                f.write('\n')
        
        # for data in dev_data:
        #     print(data['question_id'],'\t',data['query'],'\t',data['db_id'],file = f)