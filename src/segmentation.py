import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
def find_cuts(sess,model,data):
	cuts = []
	test_states = []
	all_states = []
	start = 0
	total_frames = model.source_seq_len + model.target_seq_len
	while start<data.shape[0]:
		change_start = False
		if start+total_frames > data.shape[0]:
			print(start)
			break
		end = start + total_frames
		encoder_inputs = np.zeros((1,model.source_seq_len,model.features),dtype=float)
		decoder_inputs = np.zeros((1,model.target_seq_len,model.features),dtype=float)
		decoder_outputs = np.zeros((1,model.target_seq_len,model.features),dtype=float)

		data_sel = data[start:end,:]
		encoder_inputs[0,:,:] = data_sel[:model.source_seq_len,:]
		decoder_inputs[0,:,:] = data_sel[model.source_seq_len-1:total_frames-1,:]
		decoder_outputs[0,:,:] = data_sel[model.source_seq_len:,:]
		test_loss,dec_out,dec_states,_, enc_states = model.step(sess,encoder_inputs,decoder_inputs,decoder_outputs,True)
		print("test loss between {0} and {1} is:{2}".format(start,end,test_loss))
		if test_loss>1:
			for i in range(0,len(dec_states)):
				error = np.mean(abs(dec_out[i][0]-decoder_outputs[0][i]))
				print("error at step {0} is {1}".format(start+model.source_seq_len+i,error))
				if i==0:
					max_error = error
					change_i = i
				else:
					if error>max_error:
						max_error = error
						change_i = i
			start = start + model.source_seq_len + change_i
			change_start = True
			dec_states = dec_states[:change_i]
			cuts.append(start)
		if len(all_states)>0 and len(all_states[-1])<model.source_seq_len:
			all_states.append(enc_states)
		all_states.append(dec_states)
		test_states.append(dec_states)
		if not change_start:
			start+=model.source_seq_len
	return test_states, all_states,cuts

def clustering(model,cuts,test_states,all_states,data,nactions):
	true_cuts = []
	accumulate = 0
	for i in data.values():
		accumulate +=i.shape[0]
		true_cuts.append(accumulate)

	labels_true = []
	j = 0
	for i in range(accumulate):
		if i < true_cuts[j]:
			labels_true.append(j)
		elif i==true_cuts[j]:
			j+=1
			labels_true.append(j)

	d = np.vstack(np.reshape(np.asarray(test_states[i]),(-1,model.rnn_size)) for i in range(len(test_states)))
	clustering = AgglomerativeClustering(affinity='cosine',n_clusters=nactions,linkage='single').fit(d)
	pred_labels = clustering.labels_

	a = np.vstack(np.reshape(np.asarray(all_states[i]),(-1,model.rnn_size)) for i in range(len(all_states)))
	reduced_states = PCA(n_components=3).fit_transform(a)

	all_labels = pred_labels
	all_labels = np.insert(all_labels,0,np.repeat(pred_labels[0],model.source_seq_len))
	reduced_states_ = reduced_states
	reduced_states_ = np.insert(reduced_states,0,np.tile(reduced_states[0],[model.source_seq_len,1]),axis=0)
	for i in range(len(cuts)):
		all_labels = np.insert(all_labels,cuts[i],np.repeat(pred_labels[cuts[i]],model.source_seq_len))
		reduced_states_ = np.insert(reduced_states_,cuts[i],np.tile(reduced_states_[cuts[i]],[model.source_seq_len,1]),axis=0)
	if accumulate>len(all_labels)-1:
		all_labels = np.append(all_labels,np.repeat(all_labels[-1],accumulate-len(all_labels)))
		reduced_states_ = np.vstack([reduced_states_,np.tile(reduced_states_[-1],[accumulate-len(reduced_states_),1])])

	print("ARI: {0}".format(metrics.adjusted_rand_score(labels_true, all_labels)))
	print("AMI: {0}".format(metrics.adjusted_mutual_info_score(labels_true, all_labels)))
	print("V-measure: {0}".format(metrics.homogeneity_completeness_v_measure(labels_true, all_labels)))
	#return pred_labels,reduced_states,labels_true,all_labels
	return pred_labels,reduced_states_,labels_true,all_labels

def order_labels(true_labels,pred_labels):
	key = []
	key_ = []
	for i in true_labels:
		if i not in key:
			key.append(i)
	for i in pred_labels:
		if i not in key_:
			key_.append(i)
	labels_true = []
	labels_pred = []
	for i in true_labels:
		labels_true.append(key.index(i))
	for i in pred_labels:
		labels_pred.append(key_.index(i))
	return labels_true, labels_pred
