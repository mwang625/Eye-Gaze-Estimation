import sys
import h5py

print(sys.argv)
in_file = sys.argv[1]
if_write = False
if len(sys.argv)>2:
    if_write=True
    write_id = int(sys.argv[2])

with h5py.File(in_file,'r') as h5f:
    objs = list(h5f.keys())
    print(len(objs))
    name = objs[0]
    print(h5f[name].keys())
    left_eye = h5f[name+"/left-eye/"][:]
    print(len(left_eye))
    if if_write:
        write_obj = h5f[objs[write_id]+"/eye-region"][:]
        #print(write_obj.keys())
        #with h5py.File("view_sample.h5","w") as h5fw:
        #    for key, val in write_obj.items():
        #h5fw[key]=val
if if_write:
    with h5py.File("view_sample.h5","w") as h5fw:
        h5fw["data"]=write_obj
        #for key,val in write_obj.items():
        #    h5fw[key]=val
