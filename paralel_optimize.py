import warnings
import json
from m3gnet.models import Relaxer
from pymatgen.core import Lattice, Structure
import os
import multiprocessing as mp

for category in (UserWarning, DeprecationWarning):
    warnings.filterwarnings("ignore", category=category, module="tensorflow")

def optimize_m3gnet(name):
    filename = directory_in_str + name
    if os.path.isfile(directory_in_str+'/'+name+'_optimized.cif'):
        print('File exist')
        return
    try:
        mo = Structure.from_file(filename)
    except:
        print(filename)
        print('Error cif')
        return
    
    relaxer = Relaxer()  # This loads the default pre-trained model
    relax_results = relaxer.relax(mo, verbose=False)
    final_structure = relax_results['final_structure']
    final_structure.to(filename=directory_in_str+'optimized'+'/'+name+'_optimized.cif')
    

# Init a Mo structure with stretched lattice (DFT lattice constant ~ 3.168)

directory_in_str = 'my/dir/path/'
if not os.path.isdir(directory_in_str+'optimized/'):
    os.makedirs(directory_in_str+'optimized/')
filenames = ['top_'+str(i)+'.cif' for i in range(100)]

pool = mp.Pool(mp.cpu_count())
results = pool.map(optimize_m3gnet,filenames) 
pool.close()
