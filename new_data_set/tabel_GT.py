import xraydb
from xraydb import chemparse, validate_formula, mu_elam, find_material, material_mu
import h5py
import numpy as np
import matplotlib.pyplot as plt

# Import energies i eV
energy = np.linspace(20205.3, 153527.3 , 128)

# Import the materials that xraydb dont already have
xraydb.add_material('dichloromethane', 'CH2Cl2',1.33,categories=['solvent'])
xraydb.add_material('aluminium', 'Al',2.67,categories=['metal'])
xraydb.add_material('pvc','C2H3Cl',1.42,categories=['solid'])



hej=mu_elam('Al',energy)



# water and ethanol was already in the library

# Find the attenuation for each material
mu_dic = material_mu('dichloromethane', energy,density=1.33)
mu_alu = material_mu('aluminium', energy,density=2.67)
mu_pvc = material_mu('pvc', energy,density=1.42)
mu_water = material_mu('water',energy,density=0.9982)
mu_etha = material_mu('ethanol',energy,density=0.78)

print(energy[19]/1000)
print(energy[79]/1000)

# GT curve plot
# Plotting span. Remember 0-idexing
idx_first=19
idx_last=79
extent=extent=np.linspace(energy[idx_first]/1000,energy[idx_last]/1000,8)
extent = ["%.2f" % v for v in extent]
curve_span=np.linspace(idx_first,idx_last,(idx_last-idx_first)+1)
curve_span=curve_span.astype(int)
plt.plot(mu_alu[curve_span],'r',label='Aluminium')
plt.plot(mu_dic[curve_span],'m',label='Dichloromethane')
plt.plot(mu_pvc[curve_span],'g',label='PVC')
plt.plot(mu_etha[curve_span],'c',label='Ethanol')
plt.plot(mu_water[curve_span],'b',label='Water')



plt.legend()
plt.xlabel('Energy [keV]')
plt.ylabel('Attenuation [1/cm]')
plt.ylim(0,10)
plt.title('Ground Truth energy curves between channel '+str(idx_first+1)+' and '+str(idx_last+1)+'\n using the xraydb library')
plt.xticks(ticks=np.linspace(0,idx_last-idx_first,8),labels=extent)
plt.savefig('xraydb_GT_curve_'+str(idx_first+1)+'_'+str(idx_last+1)+'.png')
plt.clf()