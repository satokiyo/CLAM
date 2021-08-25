import pickle
import h5py

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer)
	writer.close()

def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file


def save_hdf5(output_path, asset_dict, attr_dict= None, mode='a'):
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    file.close()
    return output_path


def create_hdf5_group(h5py_obj, group_name):
    if group_name in h5py_obj:
        #print(f"already exist group {group_name}")
        return h5py_obj[group_name]
    else:
        print(f"create group {group_name}")
        return h5py_obj.create_group(group_name)


def create_hdf5_dataset(h5py_obj, dset_name, data, data_type=None, append=False):
    if dset_name in h5py_obj:
        if not append:
            del h5py_obj[dset_name]
            print(f"update dataset {dset_name}")
    data_shape = data.shape
    if not data_type:
        data_type = data.dtype
    if append and (dset_name in h5py_obj):
        dset = h5py_obj[dset_name]
        dset.resize(len(dset) + data_shape[0], axis=0)
        dset[-data_shape[0]:] = data
        print(f"append dataset {dset_name}")
    else:
        chunk_shape = (1, ) + data_shape[1:]
        maxshape = (None, ) + data_shape[1:]
        h5py_obj.create_dataset(dset_name, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type, data=data) # maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
        print(f"create dataset {dset_name}")
 

def create_hdf5_attrs(h5py_obj, name, data):
    if name in h5py_obj.attrs.keys():
        #print(f"already exist attr {name}")
        pass
    else:
        h5py_obj.attrs.create(name=name, data=data)
        print(f"create attr {name}")


def open_hdf5_file(path, mode):
    '''
    wait until file ready to open
    '''
    while True:
        count=0
        try:
            f = h5py.File(path, libver='latest', mode=mode)
            return f 
        except:
            #print('not ready')
            count+=1
            if count > 100:
                raise
            pass

