import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# load the data
f1_name_prota = "gap_var_pos3"
f2_name_prota = "LB_std"
X1_prota = np.load('X1_PROTA.npy', allow_pickle=True)
X2_prota = np.load('X2_PROTA.npy', allow_pickle=True)
Y1_prota = np.load('Y1_PROTA.npy', allow_pickle=True)
Y2_prota = np.load('Y2_PROTA.npy', allow_pickle=True)



f1_name_flat = "gc_mean_pos3"
f2_name_flat = "inter_len_mean"
X1_flat = np.load('X1_FLAT.npy', allow_pickle=True)
X2_flat = np.load('X2_FLAT.npy', allow_pickle=True)
Y1_flat = np.load('Y1_FLAT.npy', allow_pickle=True)
Y2_flat = np.load('Y2_FLAT.npy', allow_pickle=True)






# make subplots with 
# if there were not interactions
# the plots should lie in horizontal or vertical
# gradients

fs = 24
fig, axs = plt.subplots(2,2, figsize=(14, 11))
fig.tight_layout(h_pad=2, w_pad=2)

ax = axs[0,0]
ax.text(-0.1, 1.1, 'A', transform=ax.transAxes, fontsize=fs, fontweight='bold', va='top', ha='right')
im1 = ax.contourf(X1_prota, X2_prota, Y1_prota, )
ax.set_title('PDP for H1 p-value prediction', fontsize=fs)
ax.set_xlabel(f1_name_prota, fontsize=fs)
ax.set_ylabel(f2_name_prota, fontsize=fs)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

# make space betwee two plots
plt.subplots_adjust(wspace=0.2)


ax = axs[0,1]
im2 = ax.contourf(X1_prota, X2_prota, Y2_prota )
ax.set_title('PDP for H2 p-value prediction', fontsize=fs)
ax.set_xlabel(f1_name_prota, fontsize=fs)
ax.set_ylabel(f2_name_prota, fontsize=fs)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

# make space betwee two plots
plt.subplots_adjust(hspace=0.2)

ax = axs[1,0]
ax.text(-0.1, 1.1, 'B', transform=ax.transAxes, fontsize=fs, fontweight='bold', va='top', ha='right')
im3 = ax.contourf(X1_flat, X2_flat, Y1_flat, )
# ax.set_title('PDP for H1 p-value prediction', fontsize=fs)
ax.set_xlabel(f1_name_flat, fontsize=fs)
ax.set_ylabel(f2_name_flat, fontsize=fs)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

# make space betwee two plots
plt.subplots_adjust(wspace=0.25)


ax = axs[1,1]
im4 = ax.contourf(X1_flat, X2_flat, Y2_flat )
# ax.set_title('PDP for H2 p-value prediction', fontsize=fs)
ax.set_xlabel(f1_name_flat, fontsize=fs)
ax.set_ylabel(f2_name_flat, fontsize=fs)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im4, cax=cax, orientation='vertical')

# save the figure as pdf with tight bounding box
fig.savefig("/Users/ulises/Desktop/ABL/writings/figures/pdp.pdf", bbox_inches='tight')