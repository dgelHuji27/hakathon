import numpy as np

class FF(object):
	"""A simple FeedForward neural network"""
	def __init__(self, layerDims):
		super(FF, self).__init__()
		n_weights = len(layerDims)-1
		self.weights = []
		for i in range(n_weights):
			self.weights.append(0.1*np.random.randn(layerDims[i+1], layerDims[i]))



	def sgd(self, X, y, epochs, eta, mb_size, Xtest, ytest):
		N = X.shape[1]
		n_mbs = int(np.ceil(N/mb_size))
		acc = self.eval_test(Xtest, ytest)

		updates = 0
		steps = [updates]
		test_acc = [acc]
		print("Starting training, test accuracy: {0}".format(acc))

		for i in range(epochs):
			perm = np.random.permutation(N);
			for j in range(n_mbs):
				X_mb = X[:,perm[j*mb_size:(j+1)*mb_size]]
				y_mb = y[perm[j*mb_size:(j+1)*mb_size],:]

				grads = self.backprop_mat(X_mb, y_mb.T)

				for k,grad in enumerate(grads):
					self.weights[k] = self.weights[k] - (eta/mb_size)*grad

				updates = updates + 1
				if updates%50 == 0:
					steps.append(updates)
					test_acc.append(self.eval_test(Xtest, ytest))

			acc = self.eval_test(Xtest, ytest)
			print("Done epoch {0}, test accuracy: {1}".format(i+1, acc))

		steps = np.asarray(steps)
		steps = steps/n_mbs

		return steps, test_acc

	def backprop_man_mat(self, X, y):
		Z1_M = self.weights[0].dot(X)
		A1_M = self.activation(Z1_M)
		Z2_M = self.weights[1].dot(A1_M)
		A2_M = self.activation(Z2_M)

		dz2_tot = self.activation_deriv(Z2_M) * self.loss_deriv(A2_M, y)

		dz1_tot = self.weights[1].T.dot(dz2_tot) * self.activation_deriv(Z1_M)

		dw1 = dz1_tot.dot(X.T)
		dw2 = dz2_tot.dot(A1_M.T)

		return [dw1, dw2]


	def backprop_mat(self, X, y):
		Z = []
		A = []

		A.append(X)
		for w in range(len(self.weights)):
			Z.append(self.weights[w].dot(A[w]))
			A.append(self.activation(Z[w]))

		DZ = []
		DZ.append(self.activation_deriv(Z[-1]) * self.loss_deriv(A[-1], y))

		for i in range(len(self.weights) - 1):
			DZ.append(self.weights[-1-i].T.dot(DZ[i]) * self.activation_deriv(Z[-2-i]))

		DW = []
		DZ.reverse()

		for i in range(len(DZ)):
			DW.append(DZ[i].dot(A[i].T))

		return DW

	def backprop_manual(self, X, y):
		avgErr = []
		avgS = []

		for i in range(len(X.T)):
			Z1 = self.weights[0].dot(np.array([X.T[i]]).T)  # the raw calculation
			A1 = self.activation(Z1)
			Z2 = self.weights[1].dot(A1)
			A2 = self.activation(Z2)

			dz2 = self.activation_deriv(Z2)*(A2 - np.array([y.T[i]]).T)  # this is basiclly the error
			dz1 = self.weights[1].T.dot(dz2) * self.activation_deriv(Z1)

			avgErr.append([dz1, dz2])
			avgS.append([np.array([X.T[i]]).T, A1, A2])

		DW = []
		avgErr = np.array(avgErr)
		avgErr = np.average(avgErr, axis=0)
		avgS = np.array(avgS)
		avgS = np.average(avgS, axis=0)
		# Gradients
		for i in range(len(avgErr)):
			DW.append(avgErr[i].dot(avgS[i].T))

		return DW


	def backprop(self, X, y):
		avgErr = []
		avgL = []

		for i in range(len(X[0])):
			Z = []
			A = []
			Errs = []


			A.append(np.array([X.T[i]]).T)
			Z.append(np.array([X.T[i]]).T)

			for w in range(len(self.weights)):
				Z.append(self.weights[w].dot(A[w]))
				A.append(self.activation(Z[w+1]))

			"""
			Z.append(self.weights[0].dot(A[0]))
			A.append(self.activation(Z[1]))

			Z.append(self.weights[1].dot(A[1]))
			A.append(self.activation(Z[2]))
			
			"""
			Errs.append(self.activation_deriv(Z[-1])*(A[-1] - np.array([y.T[i]]).T))

			for i in range(len(self.weights) - 1):
				Errs.append(self.weights[-1-i].T.dot(Errs[i]) * self.activation_deriv(Z[-2-i]))

			"""
			Errs.append(self.activation_deriv(Z[2]) * (A[2] - np.array([y.T[i]]).T))
			Errs.append(self.weights[1].T.dot(Errs[0]) * self.activation_deriv(Z[1]))
			"""
			Errs.reverse()

			avgErr.append(Errs)
			avgL.append(A)

		DW = []
		avgErr = np.array(avgErr)
		avgErr = np.average(avgErr, axis=0)
		avgL = np.array(avgL)
		avgL = np.average(avgL, axis=0)
		# Gradients
		for i in range(len(avgErr)):
			DW.append(avgErr[i].dot(avgL[i].T))

		return DW

	def predict(self,x):
		a = x
		for w in self.weights:
			a = FF.activation(self, np.dot(w,a))

		return a

	def eval_test(self,Xtest, ytest):
		ypred = self.predict(Xtest)
		ypred = ypred==np.max(ypred,axis=0)
		
		return np.mean(np.all((ypred-ytest)**2,axis=0))


	def activation(self, x):
		return np.tanh(x)

	def activation_deriv(self, x):
		return 1-(np.tanh(x)**2)

	def loss_deriv(self, output, target):
		# Derivative of loss function with respect to the activations
		# in the output layer.
		# we use quadratic loss, where L=0.5*||output-target||^2
		
		# YOUR CODE HERE

		return output - target