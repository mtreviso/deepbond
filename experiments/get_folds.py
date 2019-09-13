import numpy as np
import os 
import shutil
np.random.seed(1)
import argparse
import warnings


def check_args(args):
    data = os.path.abspath(args.data)
    output = os.path.abspath(args.output) if args.output else ''
    if not os.path.exists(args.data):
        raise FileNotFoundError(
            'No such file or directory: \'{}\''.format(data))
    if not os.path.isdir(args.data):
        raise ValueError('Path \'{}\' is not a directory'.format(data))
    return data, output


def main(data_dir,folds_dir):
    if not os.path.exists(folds_dir):
        os.mkdir(folds_dir)

    n_folds = 5
    classes_list_dir= os.listdir(data_dir)

    for dir in classes_list_dir:
        class_path = os.path.join(data_dir,dir)
        file_list = os.listdir(class_path)
        np.random.shuffle(file_list)
        num_files = len(file_list)
        test_len = int(num_files/n_folds)
        fold_dir = os.path.join(folds_dir,dir)
        if not os.path.exists(fold_dir):
            os.mkdir(fold_dir)
        
        for id in range(0,num_files,test_len):
            fold_out_dir_name = os.path.join(fold_dir,str(id))
            if not os.path.exists(fold_out_dir_name):
                os.mkdir(fold_out_dir_name) 
            fold_test_out = os.path.join(fold_out_dir_name,'test')
            if not os.path.exists(fold_test_out):
                os.mkdir(fold_test_out) 

            fold_train_out= os.path.join(fold_out_dir_name,'train')
            if not os.path.exists(fold_train_out):
                os.mkdir(fold_train_out) 

            #test files
            test_files = file_list[id:int(id+test_len)]
            for file in test_files:
                shutil.copyfile(os.path.join(class_path,file),os.path.join(fold_test_out,file))

            #train files
            train_files = file_list[0:id]+file_list[int(id+test_len):]
            for file in train_files:
                shutil.copyfile(os.path.join(class_path,file),os.path.join(fold_train_out,file))

            # get for another class to train
            for typ in classes_list_dir:
                if dir == typ:
                    continue
                typ_path = os.path.join(data_dir,typ)
                other_class_file_list =  os.listdir(typ_path)
                for file in other_class_file_list:
                    shutil.copyfile(os.path.join(typ_path,file),os.path.join(fold_train_out,file))

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-c', type=str,
                        required=True,
                        help='Corpus path, Exemplo: ../../transcriptions/ss')
    parser.add_argument('--output', '-o', type=str,
                        required=False,
                        help='Output path, Exemplo: ../../folds')
    ARGS = parser.parse_args()
    data, output = check_args(ARGS)
    main(data, output)
