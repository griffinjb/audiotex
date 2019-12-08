import numpy as np 
import matplotlib.pyplot as plt 
from scipy.io import wavfile
from scipy.interpolate import interp1d

# Reference Material:
	# https://arxiv.org/pdf/1510.02189.pdf - Sparse approximation based on a random overcomplete basis
	# http://dafx16.vutbr.cz/dafxpapers/16-DAFx-16_paper_07-PN.pdf - cross fade
	# http://eeweb.poly.edu/iselesni/EL713/STFT/stft_inverse.pdf - STFT, ISTFT

# Lets start simple.

# Consider signals X and Y

# Both signals are available up until the nth sample

# Consider the derived signals XW and YW, denoting sliding window functions on X and Y corresponsingly.

# For basis selection, consider a ranking system. 
# Let NB denote the number of basis vectors
# The rank of each vector is determined by the sum of their weights in the past (?) time steps
# If the rank of a vector falls below a threshold, it is booted from the list. 
# How can we mitigate lifespan bias?
# How about a sort of "bubble sort" esque method:
	# If the bottom rank vector is better than it's neighbor, swap until not
	# or maybe just swap neighbor once on every iteration

# nonconvexity! This occurs when a vector is internal to the span of other basis vectors.

# Perhaps a custom loss function is in order...
# What I mean by this:
	# It is not a bad thing if the correlated harmonics are present! In fact this is good
	# If it is not corrected, we will be losing power in our estimator... because the bad of gaining harmonics where none is present
	# will balance the good of building the fundamental. Perhaps it can be dampened? Ensure that it is bidirectional.


# Perhaps to improve generality... we can generate pitch shifted base vectors

def X_gen():
	fmod = 523.25
	t = np.linspace(0,10,10*44100)
	# signal = np.sin(fmod*2*np.pi*t) + .7*np.sin(2*fmod*2*np.pi*t) + .7*np.sin(3*fmod*2*np.pi*t)
	signal = np.sin(fmod*2*np.pi*t) + np.sin(2*fmod*2*np.pi*t)
	while 1:
		for sample in signal:
			yield(sample)

def Y_gen():
	fmod = 523.25
	t = np.linspace(0,10,10*44100)
	signal = np.sin(fmod*2*np.pi*t)
	while 1:
		for sample in signal:
			yield(sample)

def window_gen(gen,N,step=1):
	window = np.zeros(N)
	window_width = 1
	ctr = 0
	for sample in gen:

		if 'array' in str(type(sample)):
			if np.prod(sample.shape) == 1:
				window = np.roll(window,1)
				window[0] = sample
				ctr += 1
				if not ctr%step:
					ctr = 0
					yield(window)

			else:
				if window_width != len(sample):
					window_width = len(sample)
					window = np.zeros([N,len(sample)])
				window = np.roll(window,1,axis=0)
				window[0,:] = sample
				ctr += 1
				if not ctr%step:
					yield(window)
					ctr = 0

		else:
			window = np.roll(window,1)
			window[0] = sample
			ctr += 1
			if not ctr%step:
				yield(window)
				ctr = 0

def sin_window_gen(gen,N,step=1):

	noverlap = N-step

	window = np.zeros(N)

	ctr = 0

	for sample in gen:
		window = np.roll(window,1)
		window[0] = sample

		ctr += 1
		if not ctr % step:
			ctr = 0
			shaped_window = window.copy()
			shaped_window[:noverlap] *= sin_half_window(noverlap,'start')
			shaped_window[-noverlap:] *= sin_half_window(noverlap,'end')

			yield(shaped_window)

def sin_half_window(N,start_end='start'):
	if start_end == 'start':
		return(np.sin(np.pi*np.linspace(0,N-1,N)/2/N))

	if start_end == 'end':
		return(np.sin(np.pi*np.linspace(N,2*N,N)/2/N))

def plt_gen(gen,figure_name,skip=20,db=False):
	ctr = 0
	for sample in gen:
		ctr += 1
		if not ctr%skip:
			ctr = 0
			plt.figure(figure_name)
			plt.cla()
			if len(sample.shape) == 1:
				plt.plot(sample)
			elif len(sample.shape) == 2:
				if db:
					plt.imshow(np.log10(sample))
				else:
					plt.imshow(sample)
			plt.show(block=False)
			plt.pause(.001)
		yield(sample)

def rfft_gen(windows):
	for window in windows:
		yield(np.fft.rfft(window))

def abs_gen(data):
	for dat in data:
		yield(np.abs(dat))

def phase_gen(data):
	for dat in data:
		yield(np.angle(dat))

def dictionary_estimator(dictionary,target):
	w = np.linalg.pinv(dictionary.T)@(target[:,None])
	return((dictionary.T@w)[:,0])
	
def out_2dictionary_estimator(dictionary,target,mu=1e-14,max_iter=100):

	# SGD

	# DW = T -> W = inv(D)T

	gradient = np.zeros([dictionary.shape[0],1])
	W = np.zeros([dictionary.shape[0],1])

	gs = []
	gn = []

	for i in range(max_iter):
	
		for col in range(dictionary.shape[1]):
	
			e = W.T@dictionary[:,col] - target[col]
	
			g = 2*e*dictionary[:,col]
	
			gs.append(g)

			# gn.append(np.linalg.norm(g))	
			# liveplot(gn,'gn',1)
	
		gradient = np.mean(gs,axis=0)[:,None]

		W -= .01*gradient

	return(np.sum(W*dictionary,axis=0))

def out_dictionary_estimator(dictionary,target,mu=1e-14,max_iter=1000):

	# 2eX
	# 2(WD-T)D

	errors = np.zeros(max_iter)
	# gradients = np.zeros([dictionary.shape[0],max_iter])
	gradients = np.zeros(max_iter)


	W = np.random.normal(0,1,[dictionary.shape[0],1])

	for i in range(max_iter):

		gradient = np.zeros([dictionary.shape[0],1])

		for col in range(dictionary.shape[1]):

			e = (W.T@dictionary[:,col]-target[col])**2

			X = dictionary[:,col][:,None]

			gradient += 2*e*X

		errors = np.roll(errors,1)
		errors[0] = e

		gradients = np.roll(gradients,1)
		gradients[0] = np.linalg.norm(gradient)

		W -= mu*gradient

	liveplot(errors,'errors',dim=1)
	liveplot(gradients,'gradients',dim=1)
	return(np.sum(W*dictionary,axis=0))

def dictionary_gen(source_signal):
	for window in window_gen(source_signal,32):
		yield(window)

def dictionary_update(new_sample,rank_function,d_in=None):
	print('')
	# Score

	# remove lowest score

def rank_by_recency(d_in):
	return(d_in)

def main():
	N = 512
	codebook_nrows = 32
	codebook_ncols = int(N/2)+1

	codebook_source = 	abs_gen(rfft_gen(window_gen(X_gen(),N,N)))
	tonal_driver 	= 	abs_gen(rfft_gen(window_gen(Y_gen(),N,N)))
	phase_driver	=	phase_gen(rfft_gen(window_gen(Y_gen(),N,N)))

	codebook = np.zeros([codebook_nrows,codebook_ncols])

	# Iterate	
	for codebook_sample,tonal_sample,phase_sample in zip(codebook_source,tonal_driver,phase_driver):

		# Update
		codebook = np.roll(codebook,1,axis=0)
		codebook[0,:] = codebook_sample
		
		# Fit
		reconstruction = dictionary_estimator(codebook,tonal_sample)
		
		# Replace
		signal_out = np.fft.irfft(reconstruction*np.exp(1j*phase_sample))

		yield(signal_out)

def wav_gen(fn):
	Fs,wav = wavfile.read(fn)
	print(Fs)
	if len(wav.shape) == 2:
		mono = wav[0,:]
	else:
		mono = wav
	while 1:
		for sample in mono:
			yield(sample)

def liveplot(data,fig_name,dim=2):

	plt.figure(fig_name)
	plt.cla()
	if dim==2:
		plt.imshow(data)
	if dim==1:
		plt.plot(data)
	plt.pause(.001)

def spectrogram(X,N_samples=512,noverlap=0):
	stft = np.log10(np.array([fft for fft in abs_gen(rfft_gen(window_gen(X,N_samples,N_samples-noverlap)))]))
	return(stft)

def crossfade(x1,x2):
	# window1 = np.sqrt(.5*(1-np.linspace(-1,1,len(x1))))
	# window2 = np.sqrt(.5*(1+np.linspace(-1,1,len(x1))))
	# x1mod = x1 * window1
	# x2mod = x2 * window2
	# return(x1mod+x2mod)

	# return(np.linspace(x1[0],x2[-1],len(x1)))
	
	x_start = np.linspace(1,len(x1)/4,int(len(x1)/4))
	x_end = x_start + .75*len(x1)
	x = np.hstack([x_start,x_end])

	y = np.hstack([x1[:int(len(x1)/4)],x2[-int(len(x1)/4):]])

	interpolator = interp1d(x,y,kind='cubic')

	return(interpolator(np.linspace(1,len(x1),len(x1))))

class codebook:

	def __init__(self,N_codes,N_dim,estimator='LMS',update_mode='recency'):

		self.N_codes = N_codes
		self.N_dim = N_dim
		self.cdbk = np.zeros(N_codes,N_dim)
		self.estimator = estimator
		self.update_mode = update_mode

	def update(self,new_sample):
		
		if self.update_mode == 'recency':
			print('')

		# distances = np.zeros(self.N_codes)
		# for row in range(self.cdbk.shape[0]):
		# 	code = self.cdbk[row,:]
		# 	distances[row] = np.linalg.norm(new_sample-code)

		# Initialize 0

		# If 0, replace with new sample

		# If no 0, 


		# We want to maximize the sum of the angles
			# Between the codes, including new sample

		# Assume codebook size equals the number of notes

		# Then, the optimal codes will maximize thae sum
		# of angles between the codes

		# If the notes/codes form a convex hull, (bold)
		# then a new sample will be assigned as a code
		# if it has non-zero additive estimation error
		# This code will replace it's nearest neighbor (angle)

		# Codes are unit vectors








	def estimate(self,target):
		if self.estimator == 'LMS':
			w = np.linalg.pinv(self.cdbk.T)@(target[:,None])
			return((dictionary.T@w)[:,0])

def test_1():
	# fn_codebook = 'violin_AM.wav'
	# fn_target   = 'nu_beet-002.wav'

	#fn_codebook = 'piano_C.wav'
	fn_codebook = 'flute1_C.wav'
	fn_target   = 'flute2_C.wav'

	# fn_codebook = 'flute_C.wav'
	# fn_target   = 'piano_C.wav'

	N = 512
	codebook_nrows = 1
	codebook_ncols = int(N/2)+1

	codebook_source = 	abs_gen(rfft_gen(window_gen(wav_gen(fn_codebook),N,N)))
	#tonal_driver 	= 	abs_gen(rfft_gen(window_gen(wav_gen(fn_target),N,N)))
	#phase_driver	=	phase_gen(rfft_gen(window_gen(wav_gen(fn_target),N,N)))

	#disp = plt_gen(window_gen(abs_gen(rfft_gen(window_gen(wav_gen(fn_codebook),N,N))),128),'Spectrogram',True)
	#j = [disp.__next__() for i in range(128)]
	
	
	#codebook_source = 	abs_gen(rfft_gen(window_gen(X_gen(),N,N)))
	tonal_driver 	= 	abs_gen(rfft_gen(window_gen(Y_gen(),N,N)))
	phase_driver	=	phase_gen(rfft_gen(window_gen(Y_gen(),N,N)))

	codebook = np.zeros([codebook_nrows,codebook_ncols])

	ctr = 0


	# Iterate	
	for codebook_sample,tonal_sample,phase_sample in zip(codebook_source,tonal_driver,phase_driver):
		

		# Update
		codebook = np.roll(codebook,1,axis=0)
		codebook[0,:] = codebook_sample
	
		# Fit
		reconstruction = dictionary_estimator(codebook,tonal_sample)
		
		if not ctr%20 and 0:
			liveplot(codebook,'cdbk')
			liveplot(reconstruction,'re',1)
			liveplot(tonal_sample,'tonal',1)
			liveplot(tonal_sample-reconstruction,'error',1)
		
		# Replace
		# signal_out = np.fft.irfft(tonal_sample*np.exp(1j*phase_sample))
		signal_out = np.fft.irfft(reconstruction*np.exp(1j*phase_sample))
		#signal_out = reconstruction
		#yield(signal_out)

		yield(np.flip(signal_out))

def test_2():
	# fn_codebook = 'violin_AM.wav'
	# fn_target   = 'nu_beet-002.wav'

	#fn_codebook = 'piano_C.wav'
	fn_codebook = 'flute1_C.wav'
	fn_target   = 'flute2_C.wav'

	# fn_codebook = 'flute_C.wav'
	# fn_target   = 'piano_C.wav'

	N = 256
	codebook_nrows = 4
	codebook_ncols = N


	#codebook_source = 	abs_gen(window_gen(wav_gen(fn_codebook),N,N))
	#tonal_driver 	= 	abs_gen(rfft_gen(window_gen(wav_gen(fn_target),N,N)))
	#phase_driver	=	phase_gen(rfft_gen(window_gen(wav_gen(fn_target),N,N)))
	
	
	codebook_source = 	window_gen(X_gen(),N,N)
	tonal_driver 	= 	window_gen(Y_gen(),N,N)
	phase_driver	=	phase_gen(window_gen(Y_gen(),N,N))

	codebook = np.zeros([codebook_nrows,codebook_ncols])

	ctr = 0


	# Iterate	
	for codebook_sample,tonal_sample in zip(codebook_source,tonal_driver):
		

		# Update
		codebook = np.roll(codebook,1,axis=0)
		codebook[0,:] = codebook_sample
	
		# Fit
		reconstruction = dictionary_estimator(codebook,tonal_sample)
		
		if not ctr%20 and 0:
			liveplot(codebook,'cdbk')
			liveplot(reconstruction,'re',1)
			liveplot(tonal_sample,'tonal',1)
			liveplot(tonal_sample-reconstruction,'error',1)
		
		# Replace
		signal_out = reconstruction

		yield(np.flip(signal_out))

def test_3():

	N = 512
	noverlap = 64
	codebook_nrows = 1
	codebook_ncols = int(N/2)+1

	fn_codebook = 'flute_C.wav'

	codebook_signal = 	abs_gen(rfft_gen(window_gen(wav_gen(fn_codebook),N,N)))
	# codebook_signal = abs_gen(rfft_gen(window_gen(X_gen(),N,N-noverlap)))
	tonal_driver = abs_gen(rfft_gen(window_gen(Y_gen(),N,N-noverlap)))
	phase_driver = phase_gen(rfft_gen(window_gen(Y_gen(),N,N-noverlap)))

	if noverlap:
		overlap_buffer = np.zeros(noverlap)

	codebook = np.zeros([codebook_nrows,codebook_ncols])

	ctr = 0

	# Iterate signals
	for cdbk_sample,tonal_sample,phase_sample in zip(codebook_signal,tonal_driver,phase_driver):

		# Update Codebook

		# Update
		codebook = np.roll(codebook,1,axis=0)
		codebook[0,:] = cdbk_sample
	
		# Fit
		reconstruction = dictionary_estimator(codebook,tonal_sample)
		
		if not ctr%20 and 0:
			liveplot(codebook,'cdbk')
			liveplot(reconstruction,'re',1)
			liveplot(tonal_sample,'tonal',1)
			liveplot(tonal_sample-reconstruction,'error',1)
			ctr = 0
		ctr += 1
		
 		# Replace
		signal_out = np.flip(np.fft.irfft(reconstruction*np.exp(1j*phase_sample)))

		# Overlap crossfade
		if noverlap:
			signal_out[:noverlap] = crossfade(overlap_buffer,signal_out[:noverlap])
			overlap_buffer = signal_out[-noverlap:]

			# yield(signal_out)
			yield(signal_out[:-noverlap])
		else:
			yield(signal_out)

def test_4():

	N = 1024
	noverlap = 64
	codebook_nrows = 1
	codebook_ncols = int(N/2)+1

	fn_codebook = 'flute_C.wav'

	# codebook_signal = 	abs_gen(rfft_gen(sin_window_gen(wav_gen(fn_codebook),N,N-noverlap)))
	codebook_signal = abs_gen(rfft_gen(sin_window_gen(X_gen(),N,N-noverlap)))
	tonal_driver = abs_gen(rfft_gen(sin_window_gen(X_gen(),N,N-noverlap)))
	phase_driver = phase_gen(rfft_gen(sin_window_gen(X_gen(),N,N-noverlap)))

	if noverlap:
		overlap_buffer = np.zeros(noverlap)

	codebook = np.zeros([codebook_nrows,codebook_ncols])

	ctr = 0

	# Iterate signals
	for cdbk_sample,tonal_sample,phase_sample in zip(codebook_signal,tonal_driver,phase_driver):

		# Update Codebook

		# Update
		codebook = np.roll(codebook,1,axis=0)
		codebook[0,:] = cdbk_sample
	
		# Fit
		reconstruction = dictionary_estimator(codebook,tonal_sample)
		
		if not ctr%20 and 0:
			liveplot(codebook,'cdbk')
			liveplot(reconstruction,'re',1)
			liveplot(tonal_sample,'tonal',1)
			liveplot(tonal_sample-reconstruction,'error',1)
			ctr = 0
		ctr += 1
		
 		# Replace
		signal_out = np.flip(np.fft.irfft(reconstruction*np.exp(1j*phase_sample)))

		# Overlap crossfade
		if noverlap:
			# signal_out[:noverlap] = crossfade(overlap_buffer,signal_out[:noverlap])
			signal_out[:noverlap] = sin_half_window(noverlap,'start')*signal_out[:noverlap]+sin_half_window(noverlap,'end')*overlap_buffer
			overlap_buffer = signal_out[-noverlap:]

			# yield(signal_out)
			yield(signal_out[:-noverlap])
		else:
			yield(signal_out)

if __name__ == '__main__':

	# m = plt_gen(abs_gen(rfft_gen(plt_gen(main(),'dd',skip=1))),'ff',skip=1)
	#test_gen = plt_gen(window_gen(test_1(),64),'test_fft',1)
	test_gen = test_4()
	samples = []
	for i in range(200):
		samples.append(test_gen.__next__())
	samples /= max(abs(np.min(samples)),np.max(samples))
	samples *= 22000
	wavfile.write('test3.wav',44100,np.int16(np.vstack([np.hstack(samples),np.hstack(samples)])).T)
	
	wv = wavfile.read('test3.wav')[1]
	plt.figure('wv form')
	plt.plot(wv[:,0])
	plt.figure('Spectrogram')
	plt.imshow(spectrogram(wv[:,0]))

	plt.show()
	














# Future Patch Notes:

# Method to avoid gain reduction due to negative of 

# Overlap and add

# Non-negative pinv

# Sparse Weights

# adaptive codebook estimation

# 

















































