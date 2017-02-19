class RNN:
    def step(self,x):
        #update hidden state 
        self.h = np.tanh(np.dot(self.W_hh,self.h) + np.dot(self.W_xh,x))
        #compute output vector
        y = np.dot(self.W_hy,self.h)
        return y

rnn = RNN()
y = rnn.step()

