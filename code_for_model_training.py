import commpy
import numpy as np
import sys
import numpy
import binascii
import glob
from keras import optimizers
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt
import math

from OFDM import OFDM_module

msgNum=0
NUM_CHARS=8

# Hamming (16,8) Coding


P = np.array([[0, 1, 0, 0, 1, 1, 0, 1],
              [1, 0, 1, 0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 1, 1, 0, 1, 0, 1],
              [1, 0, 0, 1, 1, 0, 1, 0]])

P = np.array([[1, 1, 0, 1, 0, 1, 0, 0],
              [0, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 1, 1, 0, 1, 0, 1],
              [1, 0, 0, 1, 1, 0, 1, 0],
              [0, 1, 0, 0, 1, 1, 0, 1],
              [1, 0, 1, 0, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 1, 1],
              [1, 0, 1, 0, 1, 0, 0, 1]])

I = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 0, 1]])
'''
G = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0],
              [0, 1, 1, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0],
              [0, 0, 1, 1, 0, 0, 0, 0, 0,0,0,0,0,0,0,0],
              [0, 0, 0, 1, 1, 0, 0, 0, 0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 1, 1, 0, 0, 0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 1, 1, 0, 0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0, 1, 1, 0,0,0,0,0,0,0,0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1,0,0,0,0,0,0,0]])
'''
H = np.concatenate((P.T, I), axis=1)
G = np.concatenate((I, P), axis=1)

np.set_printoptions(threshold=sys.maxsize)

def generateNoise(messageNum, noise_var):
  noise_matrix = np.zeros((messageNum, 16))
  for i in range(messageNum):
      noise_matrix[i] = np.random.normal(0,noise_var,16)
      for j in range(len(noise_matrix[i])):
          noise_matrix[i][j]=int(noise_matrix[i][j])
      #print(noise_matrix[i])
      #noise_matrix[i]=int(noise_matrix[i])
  return noise_matrix

def bpsk_modulation(messageNum, input_matrix):
    result_matrix = np.zeros((messageNum, 16))
    for i in range(messageNum):
        for j in range(16):
            result_matrix[i][j] = (2*(input_matrix[i][j])) - 1
    return result_matrix

def simpleEncoding(input):
    code = np.repeat(input, 2)
    return code

def generateInput(filename):
    #messageBitOld = np.random.randint(2, size=(messageNum,8))
    #print(type(messageBit))
    f=open(filename,"rb")
    f1=f.readlines()
    input=[]
    ''''
    for x in f1:
        input.append(x)
    '''
    for x in f1:
        for y in x:
            input.append((bin(y))[2:].zfill(8))
    #print(input)
    #messageBit=bin(int(binascii.hexlify(input), 16))[2:]
    messageBit=''.join(input)
    print(messageBit)
    #messageBit=messageBit[0]+messageBit[2:]
    #print(len(messageBit))
    reshapedBit=[]
    i=0
    while(i<len(messageBit)):
        count=0
        temp=np.empty([])
        while(count<8):
            temp=np.append(temp, messageBit[i])
            count+=1
            i+=1

        reshapedBit.append(temp[1:])
    global msgNum
    msgNum = int((len(messageBit))/8)
    print(msgNum)
    reshapedBit=np.array(reshapedBit)
    reshapedBit=np.reshape(reshapedBit, (int(msgNum), 8))
    reshapedBit=reshapedBit.astype(int)
    #print(type(reshapedBit))
    #print(reshapedBit)
    return reshapedBit

def de2bi(d, n):
    d = np.array(d)
    power = 2**np.arange(n)
    d = d * np.ones((1,n))
    b = np.floor((d%(2*power))/power)
    return b

def encoding(input, codeword):
    tempword=[]
    for i in range(16):
        result = 0
        for j in range(8):
            result ^= G[j][i] * input[j]

        tempword.append(result)
    #psk = commpy.PSKModem(2)
    codeword.append(tempword)




def findSyndrome(codeword, syndrome):
    # sydrome = np.zeros((1, 8))

    for i in range(0, 8):
        result = 0
        for j in range(0, 16):
            result ^= H[i][j] * codeword[j]
        syndrome.append(result)


def decoding(syndrome, codeword, decodedMessage):
    if (syndrome[0] + syndrome[1] + syndrome[2] + syndrome[3] + syndrome[4] + syndrome[5] + syndrome[6] + syndrome[
        7] == 0):
        decodedMessage.append(codeword[0])
        decodedMessage.append(codeword[1])
        decodedMessage.append(codeword[2])
        decodedMessage.append(codeword[3])
        decodedMessage.append(codeword[4])
        decodedMessage.append(codeword[5])
        decodedMessage.append(codeword[6])
        decodedMessage.append(codeword[7])


    else:
        '''Find error position'''
        pos = 0
        j = 1
        for i in range(0, 8):
            if (syndrome[i]):
                pos += j
            j = j << 1

        '''Apply error correction'''
        if (pos < 16):
            codeword[pos - 1] = codeword[pos - 1] ^ 1

        decodedMessage.append(codeword[2])
        decodedMessage.append(codeword[4])
        decodedMessage.append(codeword[5])
        decodedMessage.append(codeword[6])
        decodedMessage.append(codeword[8])
        decodedMessage.append(codeword[9])
        decodedMessage.append(codeword[10])
        decodedMessage.append(codeword[11])

def baseline_model():
    # Create model
    model = Sequential()
    model.add(Dense(16, input_dim=16, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(16, activation='elu'))
    model.add(Dropout(0.3))


    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.2))
    '''
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.1))
    '''
    model.add(Dense(NUM_CHARS, activation='softmax'))

    # Compile model
    #Adadelta = optimizers.Adadelta(lr = 1)
    #Adadelta = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
    #Adadelta=optimizers.RMSprop(lr=0.005, rho=0.9, epsilon=None, decay=0.0)
    Adadelta=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer=Adadelta, metrics=['accuracy'])
    return model

def calculateAccuracy(messageNum, input, output):
    count = 0
    accuracy = 0
    countTotal = messageNum * 8
    for i in range(messageNum):
        for j in range(8):
            if (output[i][j] == input[i][j]):
                count+=1
    accuracy = count / countTotal
    return accuracy


'''
#Code for regular demodulation
input = np.array([1, 1, 1, 1, 0, 0, 0, 0])

codeword = []
encoding(input, codeword)
print("codeword: ",codeword)
#modulation
psk=commpy.PSKModem(256)

modulated_arr=psk.modulate(codeword)
print("Modulated array: ",modulated_arr)

#demodulation
demodulated_arr=psk.demodulate(input_symbols=modulated_arr, demod_type='hard', noise_var = 1)
print("Demodulated array: ", demodulated_arr)

syndrome = []
findSyndrome(demodulated_arr, syndrome)
decodedMessage = []
decoding(syndrome, demodulated_arr, decodedMessage)

print("decoded message: ",decodedMessage)

'''
'''
input_matrix=[[1,1],[2,2]]
encoded_matrix=input_matrix[0]
for i in range(1,len(input_matrix)):
    encoded_matrix=np.stack((encoded_matrix,input_matrix[i]), axis=0)
print (encoded_matrix)
'''

print(msgNum)
input_matrix=generateInput("Input_50_002.txt")
print(msgNum)
#print(input_matrix)


codeword = []

for i in range(msgNum):
    #encoding(input_matrix[i], codeword)
    #print(input_matrix[i])
    codeword.append(simpleEncoding(input_matrix[i]))

codeword=bpsk_modulation(msgNum, codeword)
codeword=np.array(codeword)
#input = np.reshape(codeword,(msgNum,16)).copy()

codeword=np.reshape(codeword,(msgNum,16)) # input to NN

ofdm_output = []
for j in range(msgNum):
    ofdm_output.append(OFDM_module(codeword[j]).OFDM_run())
print(ofdm_output)

noiseNum=250

noise_var = [0.1+i*0.0036 for i in range(noiseNum)]  #0.1-1
SNR = [ 20 * math.log10(1/noise_var[i]) for i in range(len(noise_var))] #0-20 dB
result_ber = []
epoch_num = [10, 100, 1000]



for s in range (len(epoch_num)):
    for index in range (len(noise_var)):
        #ofdm_output=ofdm_output+generateNoise(msgNum, noise_var[index]*noise_var[index])
        ofdm_output = ofdm_output + generateNoise(msgNum, noise_var[index] * noise_var[index])
        #print(ofdm_output)
        #print(calculateAccuracy(msgNum,input,codeword))

        model = baseline_model()

        history = model.fit(x = ofdm_output, y = input_matrix, validation_split=0.1, shuffle=True, epochs=epoch_num[s], batch_size=32)

        prediction=model.predict(x=ofdm_output, batch_size=32, verbose=0, steps=None)

        prediction=(prediction>0.05).astype(int)

        #print(prediction)

        print ("calculated accuracy: ",calculateAccuracy(msgNum, input_matrix, prediction))
        result_ber.append(1-(calculateAccuracy(msgNum, input_matrix, prediction)))
        
print(SNR)
print(result_ber)
for k in range(len(epoch_num)):
    plt.plot(SNR,result_ber[k*noiseNum:k*noiseNum+noiseNum])
#plt.plot(history.history['val_acc'])
plt.title('BER vs SNR')
plt.ylabel('BER')
plt.xlabel('SNR/dB')
#plt.legend(['train', 'test'], loc='upper left')
plt.show()


'''
noise_var=[1]
SNR = [ 20 * math.log10(1/noise_var[i]) for i in range(len(noise_var))]
vs_split=[0.4, 0.3, 0.2, 0.1]
epoch_num = [10, 100, 1000]
result_ber = []

for s in range(len(epoch_num)):
    temp_res=[]
    for vs in vs_split:
        ofdm_output = ofdm_output + generateNoise(msgNum, noise_var[0] * noise_var[0])
        # print(ofdm_output)
        # print(calculateAccuracy(msgNum,input,codeword))

        model = baseline_model()

        history = model.fit(x=ofdm_output, y=input_matrix, validation_split=vs, shuffle=True, epochs=epoch_num[s],
                            batch_size=32)

        prediction = model.predict(x=ofdm_output, batch_size=32, verbose=0, steps=None)

        prediction = (prediction > 0.05).astype(int)

        # print(prediction)

        print(calculateAccuracy(msgNum, input_matrix, prediction))
        temp_res.append(1 - (calculateAccuracy(msgNum, input_matrix, prediction)))
    temp_res=[ x/temp_res[0] for x in temp_res]
    result_ber.append((1/len(vs_split)) *sum(temp_res))
print(result_ber)

'''

