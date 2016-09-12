from __future__ import print_function
import pyximport; pyximport.install()
from interpolate import interpolate
import numpy as np
import h5py

bad_sections = {
    'A': [167],
    'B': [],
    'C': [51, 111],
}

if __name__ == "__main__":

    #for sample in bad_sections.keys():
    for sample in ['C']:

        print("Processing sampe " + sample)

        f = h5py.File('sample_' + sample + '_padded_20160501.aligned.hdf', 'r+')
        neuron_ids = f['volumes/labels/neuron_ids']
        #for bad_section in bad_sections[sample]:
        for bad_section in [111]:

            print("Processing section " + str(bad_section))

            print("Copying previous section to memory")
            a = np.array(neuron_ids[bad_section-1])
            print("Copying next section to memory")
            b = np.array(neuron_ids[bad_section+1])

            print("Interpolating...")
            neuron_ids[bad_section] = interpolate(a, b)

        f.close()
