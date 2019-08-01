import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
from matplotlib import gridspec
import viz
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
def generate_video(video_dir,labels_true,labels_pred,xyz_gt,reduced_states,colors,gen_video):
	if gen_video:
		if not os.path.exists(video_dir):
			os.makedirs(video_dir)
	fig = plt.figure(figsize=(8,6))
	gs = gridspec.GridSpec(3,2,height_ratios=[4,1,1])
	# body movement
	ax1 = plt.subplot(gs[0,0],projection='3d')
	ax1.set_yticklabels([])
	ax1.set_xticklabels([])
	ax1.set_zticklabels([])
	ob = viz.Ax3DPose(ax1)
	# predicted labels
	ax2 = plt.subplot(gs[1,:])
	ax2.set_xlim([0,len(labels_pred)])
	ax2.set_ylim([0,0.2])
	ax2.set_ylabel("pred")
	ax2.set_yticks([])
	p = {}
	p[0] = ax2.barh(0,1,color=colors[labels_pred[0]],height=0.4)
	seg = []
	prev_label = labels_pred[0]
	count = 0
	for i in range(len(labels_pred)):
		if prev_label == labels_pred[i]:
			continue
		else:
			seg.append(i)
			count +=1
			p[count] = ax2.barh(0,1,color=colors[labels_pred[i]],height=0.4)
			prev_label = labels_pred[i]
	seg.append(len(labels_pred))
	# ground truth labels
	ax3 = plt.subplot(gs[2,:])
	ax3.set_xlim([0,len(labels_true)])
	ax3.set_ylim([0,0.2])
	ax3.set_ylabel("GT")
	ax3.set_yticks([])
	p_ = {}
	p_[0] = ax3.barh(0,1,color=colors[labels_true[0]],height=0.4)
	seg_ = []
	prev_label_ = labels_true[0]
	count_ = 0
	for i in range(len(labels_true)):
		if prev_label_ == labels_true[i]:
			continue
		else:
			seg_.append(i)
			count_ +=1
			p_[count_] = ax3.barh(0,1,color=colors[labels_true[i]],height=0.4)
			prev_label_ = labels_true[i]
	seg_.append(len(labels_true))

	ax4 = plt.subplot(gs[0,1],projection='3d')
	ax4.set_xticks([])
	ax4.set_yticks([])
	ax4.set_zticks([])

	start_label = labels_pred[0]
	if gen_video:
		moviewriter = animation.FFMpegWriter(fps=120)
		with moviewriter.saving(fig,video_dir+'.mp4',dpi=100):
			# Plot the conditioning ground truth
		    s = 0
		    s_ = 0
		    for i in range(len(labels_true)):
		        
		        ob.update( xyz_gt[i,:], colors[labels_pred] )
		        
		        if i<seg[s] and s==0:
		            p[s].patches[0].set_width(i)
		        elif i<=seg[s]:
		            if i==seg[s]:
		                s+=1
		            if s==len(seg):
		                for j in range(s-1):
		                    if j==0:
		                        p[j].patches[0].set_width(seg[j])
		                    else:
		                        p[j].patches[0].set_width(seg[j]-seg[j-1])
		                    p[j+1].patches[0].set_x(seg[j])
		                    p[j+1].patches[0].set_width(seg[j+1]-seg[j])
		                p[s-1].patches[0].set_x(seg[s-2])
		                p[s-1].patches[0].set_width(i-seg[s-2])
		            else:
		                if s==1:
		                    p[0].patches[0].set_width(seg[0])
		                    p[1].patches[0].set_x(p[0].patches[0].get_width())
		                    p[1].patches[0].set_width(i-seg[0])
		                else:
		                    for j in range(s):
		                        if j==0:
		                            p[j].patches[0].set_width(seg[j])
		                        else:
		                            p[j].patches[0].set_width(seg[j]-seg[j-1])
		                        p[j+1].patches[0].set_x(seg[j])
		                        p[j+1].patches[0].set_width(i-seg[j])
		        
		        if i<seg_[s_] and s_==0:
		            p_[s_].patches[0].set_width(i)
		        elif i<=seg_[s_]:
		            if i==seg_[s_]:
		                s_+=1
		            if s_==len(seg_):
		                for j in range(s_-1):
		                    if j==0:
		                        p_[j].patches[0].set_width(seg_[j])
		                    else:
		                        p_[j].patches[0].set_width(seg_[j]-seg_[j-1])
		                    p_[j+1].patches[0].set_x(seg_[j])
		                    p_[j+1].patches[0].set_width(seg_[j+1]-seg_[j])
		                p_[s_-1].patches[0].set_x(seg_[s_-2])
		                p_[s_-1].patches[0].set_width(i-seg_[s_-2])
		            else:
		                if s_==1:
		                    p_[0].patches[0].set_width(seg_[0])
		                    p_[1].patches[0].set_x(p_[0].patches[0].get_width())
		                    p_[1].patches[0].set_width(i-seg_[0])
		                else:
		                    for j in range(s_):
		                        if j==0:
		                            p_[j].patches[0].set_width(seg_[j])
		                        else:
		                            p_[j].patches[0].set_width(seg_[j]-seg_[j-1])
		                        p_[j+1].patches[0].set_x(seg_[j])
		                        p_[j+1].patches[0].set_width(i-seg_[j])
		        ax4.scatter(reduced_states[i,0],reduced_states[i,1],reduced_states[i,2],c=colors[labels_pred[i]],s=2)
		        moviewriter.grab_frame()
	else:
	    s = 0
	    s_ = 0
	    for i in range(len(labels_true)):
	        #if i%4==0:
        	#update 3D body movement
	        ob.update( xyz_gt[i,:],colors[labels_pred] )
	        #update predicted labels
	        if i<seg[s] and s==0:
	            p[s].patches[0].set_width(i)
	        elif i<=seg[s]:
	            if i==seg[s]:
	                s+=1
	            if s==len(seg):
	                for j in range(s-1):
	                    if j==0:
	                        p[j].patches[0].set_width(seg[j])
	                    else:
	                        p[j].patches[0].set_width(seg[j]-seg[j-1])
	                    p[j+1].patches[0].set_x(seg[j])
	                    p[j+1].patches[0].set_width(seg[j+1]-seg[j])
	                p[s-1].patches[0].set_x(seg[s-2])
	                p[s-1].patches[0].set_width(i-seg[s-2])
	            else:
	                if s==1:
	                    p[0].patches[0].set_width(seg[0])
	                    p[1].patches[0].set_x(p[0].patches[0].get_width())
	                    p[1].patches[0].set_width(i-seg[0])
	                else:
	                    for j in range(s):
	                        if j==0:
	                            p[j].patches[0].set_width(seg[j])
	                        else:
	                            p[j].patches[0].set_width(seg[j]-seg[j-1])
	                        p[j+1].patches[0].set_x(seg[j])
	                        p[j+1].patches[0].set_width(i-seg[j])
	        # update ground truth labels
	        if i<seg_[s_] and s_==0:
	            p_[s_].patches[0].set_width(i)
	        elif i<=seg_[s_]:
	            if i==seg_[s_]:
	                s_+=1
	            if s_==len(seg_):
	                for j in range(s_-1):
	                    if j==0:
	                        p_[j].patches[0].set_width(seg_[j])
	                    else:
	                        p_[j].patches[0].set_width(seg_[j]-seg_[j-1])
	                    p_[j+1].patches[0].set_x(seg_[j])
	                    p_[j+1].patches[0].set_width(seg_[j+1]-seg_[j])
	                p_[s_-1].patches[0].set_x(seg_[s_-2])
	                p_[s_-1].patches[0].set_width(i-seg_[s_-2])
	            else:
	                if s_==1:
	                    p_[0].patches[0].set_width(seg_[0])
	                    p_[1].patches[0].set_x(p_[0].patches[0].get_width())
	                    p_[1].patches[0].set_width(i-seg_[0])
	                else:
	                    for j in range(s_):
	                        if j==0:
	                            p_[j].patches[0].set_width(seg_[j])
	                        else:
	                            p_[j].patches[0].set_width(seg_[j]-seg_[j-1])
	                        p_[j+1].patches[0].set_x(seg_[j])
	                        p_[j+1].patches[0].set_width(i-seg_[j])
	        ax4.scatter(reduced_states[i,0],reduced_states[i,1],reduced_states[i,2],c=colors[labels_pred[i]],s=2)
	        plt.show(block=False)
	        #fig.canvas.draw()
	        plt.pause(0.001)