#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:28:57 2024

@author: meredithshea
"""

###############################################################################
######################### Load Libraries ######################################
###############################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# FCPython has been edited slightly from the original soccermatics file. It 
# needs to be stored locally.
from FCPython import createPitch 

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
game = 'VCatSLC' 

# Set first half direction of play as 'LtoR' or 'RtoL'.
directionofplay = 'RtoL'

# Location of the goal mouth based on the first half direction of play.
gmx = 120
gmy = 40

###############################################################################
######################### Dataframe Creation ##################################
###############################################################################
# Download xlsx as dataframe.
df = pd.read_excel(game + '/GameData.xlsx')


# Flip y-coordinates (always have to do this).
df['y1'] = 80 - df['y1']
df['y2'] = 80 - df['y2']

# Ultimately, all play will run from left to right. This sets it, depending on 
# first half direction.
if directionofplay == 'LtoR':
    # Flip second half data to match first half data
    df.loc[df['half'] == 2, 'y1'] = 80 - df['y1']
    df.loc[df['half']==2, 'x1'] = 120 - df['x1']
    df.loc[df['half'] == 2, 'y2'] = 80 - df['y2']
    df.loc[df['half']==2, 'x2'] = 120 - df['x2']
elif directionofplay == 'RtoL':
    # Flip first half data
    df.loc[df['half'] == 1, 'y1'] = 80 - df['y1']
    df.loc[df['half']== 1, 'x1'] = 120 - df['x1']
    df.loc[df['half'] == 1, 'y2'] = 80 - df['y2']
    df.loc[df['half']==1, 'x2'] = 120 - df['x2']
    
    
#########################  Add a Zone Column ##################################

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

######################## Passes & Dribbles DF #################################

df_pass = df.loc[(df['event'] == 'pass') | (df['event'] == 'cross')]
df_pass = df_pass.reset_index(drop=True)

df_dribble = df.loc[(df['event'] == 'dribble')]
df_dribble = df_dribble.reset_index(drop=True)

###############################################################################
######################### Visuals #############################################
###############################################################################

######################### Passes by Zone ######################################

# (fig,ax) = createPitch(120,80,'yards','gray')

# for i,action in df_pass.iterrows():
#     x1 = action['x1']
#     x2 = action['x2']
#     y1 = action['y1']
#     y2 = action['y2']
#     if action['zone start'] == 10:
#         if action['detail'] == 'complete':
#             ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.8,lw=2))
#         else:
#             ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3, lw=2))

# ax.plot([30,30],[0,80],alpha=0.5,color='grey')
# ax.plot([90,90],[0,80],alpha=0.5,color='grey')
# ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
# ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
# ax.text(105,13.33,'12',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(105,40,'11',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(105,66.67,'10',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(75,13.33,'9',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(75,40,'8',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(75,66.67,'7',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(45,13.33,'6',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(45,40,'5',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(45,66.67,'4',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(15,13.33,'3',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(15,40,'2',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')
# ax.text(15,66.67,'1',fontsize=24,alpha=0.5,va='center',ha='center',color='grey')

# plt.title('Zone 10 Passes',size=18,pad=10)
# fig.savefig(game + '/Zone10Passes.png', dpi=300, bbox_inches='tight')
# plt.show()


######################### Passes into final 3rd ###############################   

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_pass.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if (action['zone start'] <= 9) and (action['zone end'] >= 10):
        if action['detail'] == 'complete':
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.8,lw=2))
        else:
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3, lw=2))

ax.plot([30,30],[0,80],alpha=0.5,color='grey')
ax.plot([90,90],[0,80],alpha=0.5,color='grey')
ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
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

plt.title('Passes into Final 3rd',size=18,pad=10)
fig.savefig(game + '/Final3rdPasses.png', dpi=300, bbox_inches='tight')
plt.show()

########################### All Dribbles ######################################

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_dribble.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if action['detail'] == 'complete':
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.8,lw=2))
    else:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3, lw=2))

ax.plot([30,30],[0,80],alpha=0.5,color='grey')
ax.plot([90,90],[0,80],alpha=0.5,color='grey')
ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
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

plt.show()


######################### All Passes ##########################################

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_pass.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if action['detail'] == 'complete':
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.8,lw=2))
    else:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3, lw=2))

ax.plot([30,30],[0,80],alpha=0.5,color='grey')
ax.plot([90,90],[0,80],alpha=0.5,color='grey')
ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
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

plt.show()

######################### All Passes into Final Third #########################

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_pass.iterrows():
    if action['zone start'] <= 9 and action['zone end'] >= 10:
        x1 = action['x1']
        x2 = action['x2']
        y1 = action['y1']
        y2 = action['y2']
        if action['detail'] == 'complete':
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.8,lw=2))
        else:
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3, lw=2))

ax.plot([30,30],[0,80],alpha=0.5,color='grey')
ax.plot([90,90],[0,80],alpha=0.5,color='grey')
ax.plot([0,120],[26.67,26.67],alpha=0.5,color='grey')
ax.plot([0,120],[53.33,53.33],alpha=0.5,color='grey')
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

plt.show()