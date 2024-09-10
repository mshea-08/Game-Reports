#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:59:16 2024

@author: meredithshea
"""

##############################################################################
######################### Load Libraries ######################################
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# FCPython has been edited slightly from the original soccermatics file. It 
# needs to be stored locally.
from FCPython import createPitch 
from FCPython import createGoalMouthClose
from FCPython import createGoalMouth

###############################################################################
######################### Style ###############################################
###############################################################################

plt.rcParams['font.family'] = 'monospace'
oracle = '#346B6D'
rhino = '#2A445E'
buttercup = '#F3AA20'
disco = '#841E62'
loulou = '#58094F'
fuzz ='#40f786'

###############################################################################
######################### Game Specs ##########################################
###############################################################################
game1 = 'VCatEC' 
# Set first half direction of play as 'LtoR' or 'RtoL'.
dp1 = 'RtoL'

game2 = 'VCatSLC'
dp2 = 'RtoL'

game3 = 'VCatScranton'
dp3 = 'RtoL'

# Location of the goal mouth based on the first half direction of play.
gmx = 120
gmy = 40

###############################################################################
######################### Dataframe Creation ##################################
###############################################################################

# Download xlsx as dataframe.
df1 = pd.read_excel(game1 + '/GameData.xlsx')
df2 = pd.read_excel(game2 + '/GameData.xlsx')
df3 = pd.read_excel(game3 + '/GameData.xlsx')


# Flip y-coordinates (always have to do this).
df1['y1'] = 80 - df1['y1']
df1['y2'] = 80 - df1['y2']

df2['y1'] = 80 - df2['y1']
df2['y2'] = 80 - df2['y2']

df3['y1'] = 80 - df3['y1']
df3['y2'] = 80 - df3['y2']

# Ultimately, all play will run from left to right. This sets it, depending on 
# first half direction.
if dp1 == 'LtoR':
    # Flip second half data to match first half data
    df1.loc[df1['half'] == 2, 'y1'] = 80 - df1['y1']
    df1.loc[df1['half']==2, 'x1'] = 120 - df1['x1']
    df1.loc[df1['half'] == 2, 'y2'] = 80 - df1['y2']
    df1.loc[df1['half']==2, 'x2'] = 120 - df1['x2']
elif dp1 == 'RtoL':
    # Flip first half data
    df1.loc[df1['half'] == 1, 'y1'] = 80 - df1['y1']
    df1.loc[df1['half']== 1, 'x1'] = 120 - df1['x1']
    df1.loc[df1['half'] == 1, 'y2'] = 80 - df1['y2']
    df1.loc[df1['half']==1, 'x2'] = 120 - df1['x2']
    
if dp2 == 'LtoR':
    # Flip second half data to match first half data
    df2.loc[df2['half'] == 2, 'y1'] = 80 - df2['y1']
    df2.loc[df2['half']==2, 'x1'] = 120 - df2['x1']
    df2.loc[df2['half'] == 2, 'y2'] = 80 - df2['y2']
    df2.loc[df2['half']==2, 'x2'] = 120 - df2['x2']
elif dp2 == 'RtoL':
    # Flip first half data
    df2.loc[df2['half'] == 1, 'y1'] = 80 - df2['y1']
    df2.loc[df2['half']== 1, 'x1'] = 120 - df2['x1']
    df2.loc[df2['half'] == 1, 'y2'] = 80 - df2['y2']
    df2.loc[df2['half']==1, 'x2'] = 120 - df2['x2']
    
if dp3 == 'LtoR':
    # Flip second half data to match first half data
    df3.loc[df3['half'] == 2, 'y1'] = 80 - df3['y1']
    df3.loc[df3['half']==2, 'x1'] = 120 - df3['x1']
    df3.loc[df3['half'] == 2, 'y2'] = 80 - df3['y2']
    df3.loc[df3['half']==2, 'x2'] = 120 - df3['x2']
elif dp3 == 'RtoL':
    # Flip first half data
    df3.loc[df3['half'] == 1, 'y1'] = 80 - df3['y1']
    df3.loc[df3['half']== 1, 'x1'] = 120 - df3['x1']
    df3.loc[df3['half'] == 1, 'y2'] = 80 - df3['y2']
    df3.loc[df3['half']==1, 'x2'] = 120 - df3['x2']
    
df = pd.concat([df1, df2, df3], ignore_index=True)
    
######################## Add Sequence Information #############################

df['seq1'] = 0
df.loc[(df['detail'] == 'incomplete') | (df['detail'] == 'blocked') | (df['event'] == 'ball lost') | (df['event'] == 'shot') | (df['event'] == 'free kick shot'),'seq1'] = 1
df['seq'] = df['seq1'].cumsum() - df['seq1']

######################## Add Adjusted Coordinates #############################
# For goal mouth type plots
df['x1 adjusted'] = df['y1']
df['x2 adjusted'] = df['y2']
df['y1 adjusted'] = 120 - df['x1']
df['y2 adjusted'] = 120 - df['x2']

####################### Critical Zone #########################################
df['critical zone'] = 0

df.loc[(df['x1'] <= 102) & (df['x1'] >= 90) & (df['y1'] >= 22) & (df['y1'] <= 58), 'critical zone'] = 1

####################### Shot Zone #############################################

df['shot zone'] = 0
df.loc[(df['event'] == 'shot'), 'shot zone'] = 3
df.loc[(df['event'] == 'shot') & (df['x1'] >= 108) & (df['y1'] >= 33) & (df['y1'] <= 47), 'shot zone'] = 1
df.loc[(df['event'] == 'shot') & (df['x1'] >= 100) & (df['x1'] < 108) & (df['y1'] >= 33) & (df['y1'] <= 47), 'shot zone'] = 2
df.loc[(df['event'] == 'shot') & (df['x1'] >= 100) & (df['y1'] > 47) & (df['y1'] <= 46.5), 'shot zone'] = 2
df.loc[(df['event'] == 'shot') & (df['x1'] >= 100) & (df['y1'] >= 36) & (df['y1'] < 36), 'shot zone'] = 2

######################## Action Zone ##########################################

df['zone start'] = 0
df['zone end'] = 0
    
df.loc[((df['x1'] <= 30) & (df['y1'] > 53.33)),'zone start'] = 1
df.loc[((df['x1'] <= 30) & (df['y1'] <= 53.33) & (df['y1'] > 26.67)),'zone start'] = 2
df.loc[((df['x1'] <= 30) & (df['y1'] <= 26.67)),'zone start'] = 3

df.loc[((df['x1'] > 30) & (df['x1'] <= 60) & (df['y1'] > 53.33)),'zone start'] = 4
df.loc[((df['x1'] > 30) & (df['x1'] <= 60)  & (df['y1'] <= 53.33) & (df['y1'] > 26.67)),'zone start'] = 5
df.loc[((df['x1'] > 30) & (df['x1'] <= 60)  & (df['y1'] <= 26.67)),'zone start'] = 6

df.loc[((df['x1'] > 60) & (df['x1'] <= 90) & (df['y1'] > 53.33)),'zone start'] = 7
df.loc[((df['x1'] > 60) & (df['x1'] <= 90)  & (df['y1'] <= 53.33) & (df['y1'] > 26.67)),'zone start'] = 8
df.loc[((df['x1'] > 60) & (df['x1'] <= 90)  & (df['y1'] <= 26.67)),'zone start'] = 9
    
df.loc[((df['x1'] > 90) & (df['y1'] > 53.33)),'zone start'] = 10
df.loc[((df['x1'] > 90) & (df['y1'] <= 53.33) & (df['y1'] > 26.67)),'zone start'] = 11
df.loc[((df['x1'] > 90) & (df['y1'] <= 26.67)),'zone start'] = 12  

df.loc[((df['x2'] <= 30) & (df['y2'] > 53.33)),'zone end'] = 1
df.loc[((df['x2'] <= 30) & (df['y2'] <= 53.33) & (df['y2'] > 26.67)),'zone end'] = 2
df.loc[((df['x2'] <= 30) & (df['y2'] <= 26.67)),'zone end'] = 3

df.loc[((df['x2'] > 30) & (df['x2'] <= 60) & (df['y2'] > 53.33)),'zone end'] = 4
df.loc[((df['x2'] > 30) & (df['x2'] <= 60)  & (df['y2'] <= 53.33) & (df['y2'] > 26.67)),'zone end'] = 5
df.loc[((df['x2'] > 30) & (df['x2'] <= 60)  & (df['y2'] <= 26.67)),'zone end'] = 6

df.loc[((df['x2'] > 60) & (df['x2'] <= 90) & (df['y2'] > 53.33)),'zone end'] = 7
df.loc[((df['x2'] > 60) & (df['x2'] <= 90)  & (df['y2'] <= 53.33) & (df['y2'] > 26.67)),'zone end'] = 8
df.loc[((df['x2'] > 60) & (df['x2'] <= 90)  & (df['y2'] <= 26.67)),'zone end'] = 9
    
df.loc[((df['x2'] > 90) & (df['y2'] > 53.33)),'zone end'] = 10
df.loc[((df['x2'] > 90) & (df['y2'] <= 53.33) & (df['y2'] > 26.67)),'zone end'] = 11
df.loc[((df['x2'] > 90) & (df['y2'] <= 26.67)),'zone end'] = 12 

###############################################################################
######################### Shooting Statistics #################################
###############################################################################

######################## Cumulative Shots DF ##################################

# Create df of just shots. NO FREE KICKS OR PENALTIES INCLUDED
df_shots = df.loc[(df['event'] == 'shot')]
df_shots = df_shots.reset_index(drop=True)

df_shots['shot zone'] = 3
df_shots.loc[(df_shots['x1'] >= 108) & (df_shots['y1'] >= 33) & (df_shots['y1'] <= 47), 'shot zone'] = 1
df_shots.loc[(df_shots['x1'] >= 100) & (df_shots['x1'] < 108) & (df_shots['y1'] >= 33) & (df_shots['y1'] <= 47), 'shot zone'] = 2
df_shots.loc[(df_shots['x1'] >= 100) & (df_shots['y1'] > 47) & (df_shots['y1'] <= 46.5), 'shot zone'] = 2
df_shots.loc[(df_shots['x1'] >= 100) & (df_shots['y1'] >= 36) & (df_shots['y1'] < 36), 'shot zone'] = 2

######################### Shots Visual ########################################

(fig,ax) = createGoalMouthClose()

ax.plot([33,33],[0,12],alpha=0.5,color='grey')
ax.plot([47,47],[0,12],alpha=0.5,color='grey')
ax.plot([33,47],[12,12],alpha=0.5,color='grey')

ax.plot([26,26],[0,20],alpha=0.5,color='grey')
ax.plot([54,54],[0,20],alpha=0.5,color='grey')
ax.plot([26,54],[20,20],alpha=0.5,color='grey')

for i,shot in df_shots.iterrows():
    y1 = shot['y1 adjusted']
    x1 = shot['x1 adjusted']
    if shot['detail'] == 'goal':
        ax.scatter(x1,y1,s=10, color=buttercup,zorder=100)
    elif shot['detail'] == 'on target':
        ax.scatter(x1,y1,s=10,color=oracle,zorder=50)
    else:
        ax.scatter(x1,y1,s=10,color=oracle,alpha=0.4,zorder=10)

plt.title('Shots')
ax.set_aspect('equal')
fig.savefig('compiledstats/shots.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################### Crossing Statistics #################################
###############################################################################

df['cross result'] = 0

for i,action in df.iterrows():
    if action['event'] == 'cross' and action['detail'] == 'complete':
        if (df['event'][i+1] == 'shot') and (df['detail'][i+1] == 'on target'):
            df.loc[i,'cross result'] = 2
        elif (df['event'][i+1] == 'shot') and (df['detail'][i+1] == 'goal'):
            df.loc[i,'cross result'] = 3
        else: 
            df.loc[i,'cross result'] = 1
        
df_crosses = df.loc[df['event'] == 'cross']
df_crosses = df_crosses.reset_index(drop=True)

######################## Crosses Visual 1 #####################################
(fig,ax) = createGoalMouth()

for i,cross in df_crosses.iterrows():
    x1 = cross['x1 adjusted']
    x2 = cross['x2 adjusted']
    y1 = cross['y1 adjusted']
    y2 = cross['y2 adjusted']
    if cross['cross result'] == 0:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.2, lw=1.2),zorder=10)
    elif cross['cross result'] == 1:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.7, lw=1.2),zorder=50)
    elif cross['cross result'] == 2:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco, lw=1.2),zorder=80)
    elif cross['cross result'] == 3:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=buttercup),zorder=120)
          

ax.set_aspect('equal')
plt.title('Cumulative Crosses')
fig.savefig('compiledstats/crosses1.png', dpi=300, bbox_inches='tight')
plt.show()


###############################################################################
######################### Free Kicks ##########################################
###############################################################################

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df.iterrows():
    x1 = action['x1']
    y1 = action['y1']
    x2 = action['x2']
    y2 = action['y2']
    if action['event'] == 'free kick':
        if action['detail'] == 'complete':
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco, lw=1.2),zorder=80)
        else:
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco, lw=1.2,alpha=0.5),zorder=80)

ax.plot([30,30],[0,80],alpha=0.5,color='grey')
ax.plot([90,90],[0,80],alpha=0.5,color='grey')
ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
ax.annotate("", xy=(55,5), xytext=(25,5), arrowprops=dict(arrowstyle='->'))
ax.text(20,5,'play',va='center',ha='right')
ax.text(105,13.33,'12',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(105,40,'11',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(105,66.67,'10',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(75,13.33,'9',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(75,40,'8',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(75,66.67,'7',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(45,13.33,'6',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(45,40,'5',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(45,66.67,'4',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(15,13.33,'3',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(15,40,'2',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
ax.text(15,66.67,'1',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')

plt.title('Free Kicks')
plt.show()

###############################################################################
######################### Free Kicks ##########################################
###############################################################################


(fig,ax) = createGoalMouth()

for i,action in df.iterrows():
    if action['event'] == 'corner':
        goal = 0
        if action['x1 adjusted'] < 5:
            x1 = 0
        else:
            x1 = 80
        y1 = 0
        x2 = action['x2 adjusted']
        y2 = action['y2 adjusted']
        dist = np.sqrt((x1-x2)**2+(y1-y2)**2)
        if dist > 26:
            if action['detail'] == 'complete':
                for j in range(0,3):
                    if df['detail'][i+j] =='goal':
                        goal = 1
                if goal == 1:
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=buttercup, lw=1.2),zorder=80)
                else:
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco, lw=1.2),zorder=80)
            else:
                ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco, lw=1.2,alpha=0.5),zorder=80)

plt.title('Corner Kicks')
plt.show()

###############################################################################
###############################################################################

seqnum = df['seq'].max()

for k in range(0,seqnum):
    df_seq = df.loc[df['seq'] == (k+1)]
    df_seq = df_seq.reset_index(drop=True) 
    cz_num = len(df_seq[df_seq['critical zone'] == 1]) 
    sz_num = len(df_seq[(df_seq['shot zone'] == 1) | (df_seq['shot zone'] == 2)]) 
    if cz_num > 0 and sz_num > 0: 
        # Plot the chance
        (fig,ax) = createPitch(120,80,'yards','gray')
        ax.plot([30,30],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        ax.plot([90,90],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        count = 1 # counts the events
        for i,action in df_seq.iterrows():
            x1 = action['x1']
            y1 = action['y1']
            if (action['event'] == 'pass') or (action['event'] == 'cross'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    if df_seq['event'][i+1] == 'free kick' or df_seq['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = df_seq['x1'][i+1]
                        y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
            elif action['event'] == 'throw in':
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    x2 = df_seq['x1'][i+1]
                    y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
            elif (action['event'] == 'corner') or (action['event'] == 'free kick') or (action['event'] == 'goalie restart'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    x2 = df_seq['x1'][i+1]
                    y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
            elif action['event'] == 'dribble':
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    if df_seq['event'][i+1] == 'free kick' or df_seq['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = df_seq['x1'][i+1]
                        y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,ls='dashed',zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',ls='dashed',alpha=0.5, color=oracle))  
            elif (action['event'] == 'shot') or (action['event'] == 'free kick shot'):
                x2 = action['x2']
                y2 = action['y2']
                if (x2-x1 < 7) & (x1 > 60):
                    x2 = x1 + 7
                elif (x1-x2 < 7) & (x1 < 60):
                    x2 = x1 - 7
                if action['detail'] == 'goal':
                    ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                    ax.scatter(x1,y1,marker='o',s=60,color=buttercup,zorder=50+5*i)
                    count += 1
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=buttercup))
                elif action['detail'] == 'off target' or action['detail'] == 'blocked' or action['detail'] == 'on target':
                    ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                    ax.scatter(x1,y1,marker='o',s=60,edgecolor=disco,facecolor='white',zorder=50+5*i)
                    count += 1
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco))
            elif action['event'] == 'ball lost':
                ax.scatter(x1,y1,marker='o',s=60,edgecolor=fuzz,facecolor='white',zorder=50+5*i)
                ax.scatter(x1,y1,marker='x',s=40,color=fuzz,zorder=52+5*i)
        
    
        num = str(k+1)
        ax.set_aspect('equal')
        plt.title('Sequence ' + num, size=18, pad = 5)
        plt.show()
        
###############################################################################
###############################################################################

throw_count = [0,0,0,0,0,0,0,0,0,0,0,0]
throwsuccess_count = [0,0,0,0,0,0,0,0,0,0,0,0]

for i,action in df.iterrows():
    if action['event'] == 'throw in':
        k = action['zone start'] - 1
        throw_count[k] += 1
        if (i <= len(df) - 3) and (action['detail'] == 'complete'):
            t = 0
            j = 1
            p = 0
            x1 = action['x1']
            y1 = action['y1']
            while t == 0:
                x2 = df['x1'][i+j]
                y2 = df['y1'][i+j]
                dist = np.sqrt((x1-x2)**2+(y1-y2)**2)
                if (df['event'][i+j] == 'shot') or (df['event'][i+j] == 'free kick shot'):
                    p += 3
                    t = 1
                elif (df['detail'][i+j] == 'complete') and (df['event'][i+j] != 'ball lost'):
                    p += 1
                    j += 1
                elif dist > 40:
                    p += 3
                    t = 1
                elif df['event'][i+j] == 'ball lost':
                    t = 1
                elif df['detail'][i+j] == 'incomplete':
                    t = 1
                else: 
                    t = 1
            if p >= 3:
                throwsuccess_count[k] += 1
print(throw_count)
print(throwsuccess_count)
            
            
seqnum = df['seq'].max()

for k in range(0,seqnum):
    df_seq = df.loc[df['seq'] == (k+1)]
    df_seq = df_seq.reset_index(drop=True) 
    if (df_seq['event'][0] == 'throw in') and (df_seq['zone start'][0] == 6): 
        # Plot the chance
        (fig,ax) = createPitch(120,80,'yards','gray')
        ax.plot([30,30],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        ax.plot([90,90],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        count = 1 # counts the events
        for i,action in df_seq.iterrows():
            x1 = action['x1']
            y1 = action['y1']
            if (action['event'] == 'pass') or (action['event'] == 'cross'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    if df_seq['event'][i+1] == 'free kick' or df_seq['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = df_seq['x1'][i+1]
                        y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
            elif action['event'] == 'throw in':
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    x2 = df_seq['x1'][i+1]
                    y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,ls='dotted',zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle,ls='dotted'))
            elif (action['event'] == 'corner') or (action['event'] == 'free kick') or (action['event'] == 'goalie restart'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    x2 = df_seq['x1'][i+1]
                    y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
            elif action['event'] == 'dribble':
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    if df_seq['event'][i+1] == 'free kick' or df_seq['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = df_seq['x1'][i+1]
                        y2 = df_seq['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,ls='dashed',zorder=25)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',ls='dashed',alpha=0.5, color=oracle))  
            elif (action['event'] == 'shot') or (action['event'] == 'free kick shot'):
                x2 = action['x2']
                y2 = action['y2']
                if (x2-x1 < 7) & (x1 > 60):
                    x2 = x1 + 7
                elif (x1-x2 < 7) & (x1 < 60):
                    x2 = x1 - 7
                if action['detail'] == 'goal':
                    ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                    ax.scatter(x1,y1,marker='o',s=60,color=buttercup,zorder=50+5*i)
                    count += 1
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=buttercup))
                elif action['detail'] == 'off target' or action['detail'] == 'blocked' or action['detail'] == 'on target':
                    ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                    ax.scatter(x1,y1,marker='o',s=60,edgecolor=disco,facecolor='white',zorder=50+5*i)
                    count += 1
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco))
            elif action['event'] == 'ball lost':
                ax.scatter(x1,y1,marker='o',s=60,edgecolor=fuzz,facecolor='white',zorder=50+5*i)
                ax.scatter(x1,y1,marker='x',s=40,color=fuzz,zorder=52+5*i)
        
    
        num = str(k+1)
        ax.set_aspect('equal')
        plt.title('Sequence ' + num, size=18, pad = 5)
        plt.show()

