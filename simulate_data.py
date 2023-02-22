import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
def plot_dirac(tk, ak, length, color='red', marker='*', ax=None, label=''):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    markerline211_1, stemlines211_1, baseline211_1 = \
        ax.stem(tk, ak, label=label)
    plt.setp(stemlines211_1, linewidth=1.5, color=color)
    plt.setp(markerline211_1, marker=marker, linewidth=1.5, markersize=8,
             markerfacecolor=color, mec=color)
    plt.setp(baseline211_1, linewidth=0)
    plt.xlabel('t')
    plt.ylabel('amplitude')
    plt.xlim([0, length]
'''

def simulate_dirac(k, length, amplitude_max, cutoff_freq, specificity, tau):
    # Generate random parameter to express diracs
    parameter_pair = []

    for i in range(k):
        random.seed(random.random())
        position = random.random() * length
        amplitude = random.random() * amplitude_max
        parameter_pair.append((position, amplitude))
    
    parameter_pair = sorted(parameter_pair)

    tk = []
    ak = []
    for parameter in parameter_pair:
        tk.append(parameter[0])
        ak.append(parameter[1])

    # Generate signal
    end_point = length * tau
    x = np.linspace(0, end_point, int(specificity * length * tau) + 1)
    y = np.zeros(x.size)

    '''
    for (position, amplitude) in parameter_pair:
        y += amplitude * 2 * cut_off_freq * np.sin((2 * cut_off_freq * x) - position) / (2 * cut_off_freq * x)
    '''
    for i in range(len(tk)):
        for j in range(tau):
            y[int((tk[i] * specificity)) + int(j * specificity * length)] = ak[i]

    # After low-pass signal
    yy = np.zeros(x.size)


    new_tk = []
    new_ak = []
    for freq in range(tau):
        for i in tk:
            new_tk.append(i + freq * length)
    
    for freq in range(tau):
        for i in ak:
            new_ak.append(i)
            

    for i in range(len(new_tk)):
        for j in range(len(x)):
            yy[j] += new_ak[i] * np.sinc((2 * np.pi * cutoff_freq  * (x[j] - new_tk[i])))

    return tk, ak, y, yy

def sampling_signal(signal, sample_rate, specificity):
    sample_signal = []
    
    position:int = 0
    while position < len(signal):
        sample_signal.append(signal[position])
        position += int(specificity / sample_rate)
    
    return sample_signal


if __name__ == '__main__':
    
    # Test function and plot
    k = 1
    length = 0.02
    amplitude_max = 5
    cutoff_freq = 60
    specificity = 10000
    sample_rate = 30
    tau = 1000

    tk, ak, y, yy = simulate_dirac(k, length, amplitude_max, cutoff_freq, specificity, tau)

    x = np.linspace(0, length * tau, int(specificity * length * tau) + 1)

    plt.figure()
    plt.stem(x, y)
    plt.show()

    plt.figure()
    plt.plot(x, yy)
    plt.show()

    sample_signal = sampling_signal(yy, sample_rate, specificity)

    plt.figure()
    plt.stem(np.linspace(0, length * tau, int(sample_rate * length * tau) + 1), sample_signal)
    plt.show()

    '''
    # Generate training data

    data_size = 10000

    length = 0.2
    amplitude_max = 5
    cutoff_freq = 10
    specificity = 10000
    sample_rate = 3

    df = pd.DataFrame(data={}, columns=['signal', 'key_parameter'])

    for i in range(data_size):

        k = random.randint(1, 10)

        tk, ak, y, yy = simulate_dirac(k, length, amplitude_max, cutoff_freq, specificity)

        sample_signal = sampling_signal(yy, sample_rate, specificity)

        df_tmp = pd.DataFrame({'signal': [sample_signal], 'key_parameter': [(tk, ak)]})

        df = pd.concat([df, df_tmp])

    df.to_csv('data.csv', index=False)
    '''