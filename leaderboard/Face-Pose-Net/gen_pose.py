import os
import pandas as pd
import pose_utils
import renderer_fpn
## To make tensorflow print(less (this can be useful for debug though)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#import ctypes; 
import get_Rts as getRts

"""
生成不同角度的人脸
python gen_pose.py
"""
unique_record_file = "../../data/unique_face_record.ftr"


def load_old():
    outpu_proc = './output_render/output_preproc.csv'
    df_finish = None
    try:
        df_finish = pd.read_csv(outpu_proc, header=None)
    except Exception as e:
        print(e)
    return df_finish


def load_data(remove_old=True):
    global unique_record_file
    df_record =None
    try:
        df_record = pd.read_feather(unique_record_file)
        df_record.rename({'img_path': 'file'}, axis=1, inplace=True)
        df_record['file'] = df_record['file'].str.replace("data/", "../../data/")
        df_record['x'] = 0
        df_record['y'] = 0
        df_record['width'] = 186
        df_record['height'] = 186
        print("load unique data:", len(df_record))
    except Exception as e:
        df_record = None
        print(e)
    if remove_old and df_record is not None:
        df_old = load_old()
        if df_old is not None:
            df_record = df_record[~df_record['img_id'].isin(df_old[0])]
    return df_record


def parse_input(df_record):
    df_record = df_record[['img_id', 'file', 'x', 'y', 'width', 'height']]
    df_record.set_index('img_id', inplace=True)
    data_dict = df_record.to_dict('index')
    return data_dict


if __name__=="__main__":
    ######## TMP FOLDER #####################
    _tmpdir = './tmp/'#os.environ['TMPDIR'] + '/'
    print('> make dir')
    if not os.path.exists( _tmpdir):
        os.makedirs( _tmpdir )
    #########################################
    ##INPUT/OUTPUT
    outpu_proc = './output_render/output_preproc.csv'
    output_pose_db =  './output_pose.lmdb'
    output_render = './output_render'
    #################################################
    print('> network')
    _alexNetSize = 227
    _factor = 0.25 #0.1
    # ***** please download the model in https://www.dropbox.com/s/r38psbq55y2yj4f/fpn_new_model.tar.gz?dl=0 ***** #
    model_folder = './fpn_new_model/'
    model_used = 'model_0_1.0_1.0_1e-07_1_16000.ckpt' #'model_0_1.0_1.0_1e-05_0_6000.ckpt'
    lr_rate_scalar = 1.0
    if_dropout = 0
    keep_rate = 1
    ################################
    df_record = load_data()
    data_dict = parse_input(df_record)
    ## Pre-processing the images 
    print('> preproc')
    pose_utils.preProcessImage( _tmpdir, data_dict, './',\
                                _factor, _alexNetSize, outpu_proc )
    ## Runnin FacePoseNet
    print('> run')
    ## Running the pose estimation
    getRts.esimatePose(model_folder, outpu_proc, output_pose_db, model_used, lr_rate_scalar, if_dropout, keep_rate, use_gpu=False)
    renderer_fpn.render_fpn(outpu_proc, output_pose_db, output_render)
    # 更新记录
    df_unique = pd.read_feather(unique_record_file)
    df_unique['gen_pose'] = True
    df_unique.to_feather(unique_record_file)
