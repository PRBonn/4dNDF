import numpy as np
from tabulate import tabulate

from time import time
import sys, os, math
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..' ))
sys.path.append(BASE_DIR)
from utils.pcdpy3 import load_pcd
from utils import cnt_staticAdynamic, check_file_exists, bc


def run(gt_pcd_path, et_pcd_path):
    gt_pc_ = load_pcd(check_file_exists(gt_pcd_path))
    et_pc_ = load_pcd(check_file_exists(et_pcd_path))
    num_gt = cnt_staticAdynamic(gt_pc_.np_data) 
    num_et = cnt_staticAdynamic(et_pc_.np_data)

    assert et_pc_.np_data.shape[0] == gt_pc_.np_data.shape[0] , \
		        "Error: The number of points in et_pc_ and gt_pc_ do not match.\
		        \nThey must match for evaluation, if not Please run `export_eval_pcd`."

    correct_static = np.count_nonzero((et_pc_.np_data[:,3] == 0) * (gt_pc_.np_data[:,3] == 0))
    missing_static = num_gt['static'] - correct_static
    correct_dynamic = np.count_nonzero((et_pc_.np_data[:,3] == 1) * (gt_pc_.np_data[:,3] == 1))
    missing_dynamic = num_gt['dynamic'] - correct_dynamic
    
    SA = float(correct_static) / float(num_gt['static']) * 100
    DA = float(correct_dynamic) / float(num_gt['dynamic']) * 100
    AA = math.sqrt(SA * DA)
    printed_data = []
    printed_data.append([num_et['static'], num_et['dynamic'], SA, DA,  AA])
    print(tabulate(printed_data, headers=['# static', '# dynamics', 'SA [%] ↑', 'DA [%] ↑', 'AA [%] ↑'], tablefmt='orgtbl'))


if __name__ == "__main__":

    if len(sys.argv) > 1:
        base_folder = sys.argv[1]
    else:
        sys.exit("path not valid.")
    
    gt_pcd = base_folder + '/gt_cloud.pcd'
    est_pcd = base_folder + '/eval/static_points_exportGT.pcd'
    run(gt_pcd, est_pcd)
    
