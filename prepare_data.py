import os
import numpy as np
import scipy.io as sio
import glob
import cv2
import h5py
import json
import sys
import shutil
import svg_utils
from svgpathtools import svg2paths, wsvg
from svgpathtools import svg2paths2


def save_hdf5(fname, d):
    hf = h5py.File(fname, 'w')
    for key in d.keys():
        value = d[key]
        if type(value) is list:
            value = np.array(value)
        dtype = value.dtype.name
        if 'string' in dtype:
            dtype = value.dtype.str.split('|')[1]
            value = [v.encode("ascii", "ignore") for v in value]
            hf.create_dataset(key, (len(value),1), dtype, value)
        else:
            hf.create_dataset(key,
                      dtype=value.dtype.name,
                      data=value)
    hf.close()
    return fname


def load_hdf5(fname):
    hf = h5py.File(fname, 'r')
    d = {key: np.array(hf.get(key)) for key in hf.keys()}
    hf.close()
    return d


def read_info(dataset_folder):
    subset_info = {
        'image_nums': [],
        'class_names': [],
        'num_classes': 0
    }
    data_info = {
        'id': [],
        'class_name': [],
        'class_id': [],
        'image_name': [],
        'image_id': [],
        'instance_id': [],
        'image_data': []
    }
    class_name = dataset_folder.split('/')[-3]
    data_type = dataset_folder.split('/')[-2]
    # class_id_dict = {'shoes': 0, 'chairs': 1}
    class_id_dict = {'shoes': 0, 'chairs': 0}
    id_in_list = 0
    image_id_offset = 0
    class_id = 0

    image_files = os.walk(dataset_folder).next()[2]
    # sort image files
    image_files.sort()
    image_base_names = []
    unique_image_base_names = []
    instance_ids = []
    print "read info for %s in %s" % (class_name, data_type)
    for image_file in image_files:
        if '_' not in image_file:
            raise Exception('Sketch file name wrong')
        image_base_name = '_'.join(image_file.split('_')[:-1])
        instance_id = image_file.split('_')[-1]
        instance_id = instance_id.split('.')[0]
        instance_ids.append(int(instance_id) - 1)

        image_base_names.append(image_base_name)
        if image_base_name not in unique_image_base_names:
            unique_image_base_names.append(image_base_name)

    # this is to avoid the ranking problem that "a56-3002_1.png < a_1.png" but "a.png < a56-3002.png" on chair dataset
    image_files = np.array(image_files)[np.argsort(image_base_names)].tolist()
    image_base_names = np.array(image_base_names)[np.argsort(image_base_names)].tolist()
    instance_ids = np.array(instance_ids)[np.argsort(image_base_names)].tolist()
    unique_image_base_names = np.sort(unique_image_base_names).tolist()
    # image_base_names.sort()

    for idx in range(len(image_files)):
        image_file = image_files[idx]
        data_info['id'].append(id_in_list)
        data_info['class_name'].append(class_name)
        data_info['class_id'].append(class_id_dict[class_name])
        data_info['image_name'].append(image_file)
        data_info['image_id'].append(unique_image_base_names.index(image_base_names[idx]) + image_id_offset)
        data_info['instance_id'].append(instance_ids[idx])
        id_in_list += 1
    image_id_offset += len(unique_image_base_names)
    class_id += 1

    print "\n Data list reading complete"

    num_images = len(data_info['image_name'])
    print "save svg data"
    data_info['image_data'] = []
    data_info['data_offset'] = np.zeros((num_images, 2))
    start_idx = 0
    for idx in range(num_images):
        sys.stdout.write('\x1b[2K\r>> Process svg data, [%d/%d]' % (idx, num_images))
        sys.stdout.flush()
        lines = svg_utils.build_lines(os.path.join(dataset_folder, data_info['image_name'][idx]))
        data_info['image_data'].extend(lines)
        end_idx = start_idx + len(lines)
        data_info['data_offset'][idx, ::] = [start_idx, end_idx]
        start_idx = end_idx
    data_info['data_offset'] = data_info['data_offset'].astype(int)

    if save_png:
        if simplify_flag:
            png_data_dir = dataset_folder.split('sim')[0] + 'png'
        else:
            png_data_dir = dataset_folder + '_png'
        print "save rgb data"
        data_info['png_data'] = np.zeros((num_images, 256, 256), dtype=np.uint8)
        for idx in range(num_images):
            sys.stdout.write('\x1b[2K\r>> Process png data, [%d/%d]' % (idx, num_images))
            sys.stdout.flush()
            im = cv2.imread(os.path.join(png_data_dir, data_info['image_name'][idx]).split('.svg')[0] + '.png')
            im = cv2.resize(im, (256, 256))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            # im = cv2.imread(os.path.join(dataset_folder, test_img.split('.jpg')[0] + '.jpg'))
            data_info['png_data'][idx, ::] = im

    return subset_info, data_info


def prepare_dbs(image_folder, data_type = 'h5'):

    if simplify_flag:
        simplify_str = '_sim'
    else:
        simplify_str = ''

    sketchy_info_train, data_info_train = read_info(os.path.join(image_folder, 'svg_train%s' % simplify_str))
    sketchy_info_test, data_info_test = read_info(os.path.join(image_folder, 'svg_test%s' % simplify_str))

    if save_png:
        simplify_str += '_png'

    save_hdf5(os.path.join(image_folder, 'train_svg%s.%s' % (simplify_str, data_type)), data_info_train)
    save_hdf5(os.path.join(image_folder, 'test_svg%s.%s' % (simplify_str, data_type)), data_info_test)


def generate_db_list(image_folder):
    train_file_list_txt_origin = os.path.join(image_folder, 'train.txt')
    train_file_list_txt = os.path.join(image_folder, 'train_svg.txt')
    test_file_list_txt_origin = os.path.join(image_folder, 'test.txt')
    test_file_list_txt = os.path.join(image_folder, 'test_svg.txt')
    if not os.path.exists(train_file_list_txt) or not os.path.exists(test_file_list_txt):
        with open(train_file_list_txt_origin, 'r') as f:
            train_file_lists_origin = f.read().splitlines()
        with open(test_file_list_txt_origin, 'r') as f:
            test_file_lists_origin = f.read().splitlines()
        train_file_lists = [item.split('png')[0] + 'svg' for item in train_file_lists_origin]
        test_file_lists = [item.split('png')[0] + 'svg' for item in test_file_lists_origin]
        with open(train_file_list_txt, 'w') as f:
            f.writelines("\n".join(train_file_lists))
        with open(test_file_list_txt, 'w') as f:
            f.writelines("\n".join(test_file_lists))


def split_db(image_folder, train_list_txt = 'train_svg.txt', test_list_txt = 'test_svg.txt'):
    if train_list_txt:
        with open(os.path.join(image_folder, train_list_txt)) as f:
            train_list = f.read().splitlines()
        copy_db_files(image_folder, 'svg_all', 'svg_train', train_list)
    if test_list_txt:
        with open(os.path.join(image_folder, test_list_txt)) as f:
            test_list = f.read().splitlines()
        copy_db_files(image_folder, 'svg_all', 'svg_test', test_list)
        # copy_db_files(image_folder, 'all', 'test', test_list)


def copy_db_files(root_folder, src_folder, dst_folder, file_list):
    src_path = os.path.join(root_folder, src_folder)
    dst_path = os.path.join(root_folder, dst_folder)
    print "Copy files from %s/%s to %s/%s" % (root_folder, src_folder, root_folder, dst_folder)
    src_base_names = []
    dst_base_names = []
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
        sub_dirs = os.walk(src_path).next()[1]
        for sub_dir in sub_dirs:
            os.mkdir(os.path.join(dst_path, sub_dir.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')))
        for file in file_list:
            sys.stdout.write('\x1b[2K\r>> Copying subfolder %s ==> %s: %d/%d' % (src_folder, dst_folder, file_list.index(file)+1, len(file_list)))
            sys.stdout.flush()
            src_file = os.path.join(src_path, file)
            dst_file = os.path.join(dst_path, file.replace(' ', '_').replace('-', '_').replace('(', '').replace(')', ''))
            base_name = file.split('_')[0]
            if base_name not in src_base_names:
                src_base_names.append(base_name)
            try:
                shutil.copy2(src_file, dst_file)
                if base_name not in dst_base_names:
                    dst_base_names.append(base_name)
            except:
                print "File not exist: ", src_file
        print "\n Copy finished"
        print "Warning, none of files with below basename is copied"
        print [filename for filename in src_base_names if filename not in dst_base_names]



if __name__ == "__main__":

    datasets = ['shoes', 'chairs']
    simplify_flag = True

    save_png = True

    for dataset in datasets:
        data_dir = 'data/%s/svg' % dataset

        generate_db_list(data_dir)

        split_db(data_dir)

        prepare_dbs(data_dir)
