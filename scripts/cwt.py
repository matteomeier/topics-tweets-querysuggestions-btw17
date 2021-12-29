# functions for cwt
import ast
import pandas as pd
from tqdm.notebook import tqdm
import plotly.express as px
from kolzur_filter import kz_filter
from scipy import signal
import numpy as np
from datetime import datetime, timedelta

# define function plot_peak_detection
# to plot the peak detection
# for evaluation and reporting
def plot_peak_detection(hashtag, input_df):
    '''
    :params hashtag: hashtag to plot
    :params input_df: input dataframe
    :return: plot
    '''
    tmp1 = input_df[input_df['hashtag']==hashtag][['date', 'count']]
    tmp1.rename(columns={'date':'Datum', 'count':'Häufigkeit'}, inplace=True)
    tmp1['Zeitreihe'] = 'Originale Zeitreihe'
    
    filtered_wavelets = [0] * len(tmp1)
    filtered_wavelets[1:-1] = kz_filter(tmp1['Häufigkeit'].to_numpy(), 3, 1)    
    
    tmp2 = tmp1.copy()
    tmp2['Häufigkeit'] = filtered_wavelets
    tmp2['Zeitreihe'] = 'Bereinigte Zeitreihe'
    
    wavelets = tmp1.append(tmp2)
    
    results_prom = []
    
    for i in range(1, len(wavelets)):
        peakind_loop = signal.find_peaks_cwt(filtered_wavelets, np.arange(1, i+1))
        prominences = signal.peak_prominences(filtered_wavelets, peakind_loop)
        results_prom.append(np.median(prominences[0]))
        
    id_max_prom = results_prom.index(max(results_prom)) + 1
    peakind = signal.find_peaks_cwt(filtered_wavelets, np.arange(1, id_max_prom+1))
    peakind = peakind.tolist()
    final_prominences = signal.peak_prominences(filtered_wavelets, peakind)[0].tolist()
    
    fig = px.line(wavelets.sort_values(by=['Datum', 'Zeitreihe']), x='Datum', y='Häufigkeit',
                  color='Zeitreihe', template='simple_white',
                  color_discrete_sequence=px.colors.qualitative.Antique,
                  line_dash='Zeitreihe')
    
    for item in peakind:
        peak = wavelets['Datum'].tolist()[item]
        fig.add_vrect(x0=str(datetime.strptime(str(peak), '%Y-%m-%d').date() - timedelta(days=3)),
                      x1=str(datetime.strptime(str(peak), '%Y-%m-%d').date() + timedelta(days=3)),
                      line_width=0,
                      fillcolor='grey',
                      opacity=0.2)
    fig.update_layout(font=dict(family='Computer Modern', color='black', size=15))
    fig.show()

# define function peak_detection
# to detect the peaks via cwt
def peak_detection(hashtag, input_df):
    '''
    :params hashtag: hashtag to detect peaks
    :params input_df: input dataframe
    :return: indices of peaks per hashtag
    '''

    tmp = input_df[input_df['hashtag']==hashtag][['date', 'count']]
    
    filtered_wavelets = [0] * len(tmp)
    filtered_wavelets[1:-1] = kz_filter(tmp['count'].to_numpy(), 3, 1)    

    results_prom = []
    
    for i in range(1, len(tmp)):
        peakind_loop = signal.find_peaks_cwt(filtered_wavelets, np.arange(1, i+1))
        prominences = signal.peak_prominences(filtered_wavelets, peakind_loop)
        results_prom.append(np.median(prominences[0]))
        
    id_max_prom = results_prom.index(max(results_prom)) + 1
    peakind = signal.find_peaks_cwt(filtered_wavelets, np.arange(1, id_max_prom+1))
    
    return peakind