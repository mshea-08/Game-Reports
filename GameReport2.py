#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 17:59:51 2024

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
game = 'VCatScranton' 

# Set first half direction of play as 'LtoR' or 'RtoL'.
directionofplay = 'RtoL'

# Location of the goal mouth based on the first half direction of play.
gmx = 120
gmy = 40

###############################################################################
######################### Dataframe Creation ##################################
###############################################################################

######################### Main Dataframe ######################################
# Download xlsx as dataframe.
# What should I name the file generally?
df = pd.read_excel(game + '/GameData.xlsx')

# ONLY FOR SCRIMMAGES
df.loc[df['half'] == 3, 'half'] = 1

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
    

######################## Crosses Dataframe ####################################
# Create a df of just crosses.
df_crosses = df.loc[df['event'] == 'cross']
df_crosses = df_crosses.reset_index(drop=True)

# Changes coordinates so that all first half crosses are on the left side goal and all 
# second half crosses  are on the right side goal.
df_crosses.loc[df_crosses['half'] == 1,'x1'] = 120 - df_crosses['x1']
df_crosses.loc[df_crosses['half'] == 1,'y1'] = 80 - df_crosses['y1']
df_crosses.loc[df_crosses['half'] == 1,'x2'] = 120 - df_crosses['x2']
df_crosses.loc[df_crosses['half'] == 1,'y2'] = 80 - df_crosses['y2']


######################## Shots Dataframe ######################################
# Create df of just shots. 
df_shots = df.loc[(df['event'] == 'shot') | (df['event'] == 'free kick shot')]
df_shots = df_shots.reset_index(drop=True)

# Adjust all shots so they occur at the left side goal. Note I am only changin the starting
# coordinate of the shot (x1,y1). I do not use the second coordinate for the two shot maps.
df_shots['x1adjusted'] = 120 - df_shots['x1']
df_shots['y1adjusted'] = 80 - df_shots['y1']



# Make first half shots happen on left goal.
df_shots.loc[df_shots['half'] == 1,'x1'] = 120 - df_shots['x1']
df_shots.loc[df_shots['half'] == 1,'y1'] = 80 - df_shots['y1']


# Add columns necessary for xG model.
df_shots['a'] = np.sqrt((df_shots['x1adjusted'])**2+(df_shots['y1adjusted']-44)**2)
df_shots['b'] = np.sqrt((df_shots['x1adjusted'])**2+(df_shots['y1adjusted']-36)**2)
df_shots['cos theta'] = (df_shots['a']**2+df_shots['b']**2-64)/(2*df_shots['a']*df_shots['b'])
df_shots['theta'] = np.arccos(df_shots['cos theta'])
df_shots['distance'] = np.sqrt((df_shots['x1adjusted'])**2+(df_shots['y1adjusted']-40)**2)

######################### Passes and crosses Dataframe ########################
# Create a df of passes and crosses.
df_pass = df.loc[(df['event'] == 'pass') | (df['event'] == 'cross')]
df_pass = df_pass.reset_index(drop=True)

# Create df of only successful passes and crosses.
df_passcomplete = df_pass.loc[df_pass['detail'] == 'complete']
df_passcomplete = df_passcomplete.reset_index(drop=True)

######################## Dribbles Dataframe ###################################
# Create a df of dribbles.
df_dribble = df.loc[(df['event'] == 'dribble')]
df_dribble = df_dribble.reset_index(drop=True)

# Create df of only successful dribbles.
df_dribblecomplete = df_dribble.loc[df_dribble['detail'] == 'complete']
df_dribblecomplete = df_dribblecomplete.reset_index(drop=True)

######################## Sequence Dataframe ###################################
# First add a column of all zeros.
df['seq1'] = 0

# Make seq1 col equal to 1 when there is a dispossession. 
df.loc[(df['detail'] == 'incomplete') | (df['detail'] == 'blocked') | (df['event'] == 'ball lost') | (df['event'] == 'shot') | (df['event'] == 'free kick shot'),'seq1'] = 1

# Create final seq column.
df['seq'] = df['seq1'].cumsum() - df['seq1']
    

###############################################################################
######################### Shots Visuals #######################################
###############################################################################

######################### xG Model ############################################
# This model is compiled in xGmodel.py. It currently uses shot data from the previous
# three seasons. This model does NOT automatically update. I will incorporate this 
# feature later. 

def calculate_xG(sh):    
   xG = 1-1/(1+np.exp(-0.3454-0.0924*sh['distance']+0.2863*sh['theta'])) 
   return xG  

######################### xG Shot Map #########################################
# Shot map 1 plots shots proportionally to their xG. It also computes total xG for 
# each half. 

# Create plot.
(fig,ax) = createPitch(120,80,'yards','gray')
tot_xG = 0
half1_xG = 0
half2_xG = 0
for i,shot in df_shots.iterrows():
    # For now I am not including blocked shots. I am unsure what to do with them.
    if shot['detail'] != 'blocked' and shot['event'] != 'free kick shot':
        y1 = shot['y1']
        x1 = shot['x1']
        # The variable tot_xG is calculating the cumulative xG.
        tot_xG += calculate_xG(shot)
        if shot['half'] == 1:
            half1_xG += calculate_xG(shot)
        else:
            half2_xG += calculate_xG(shot)
        if shot['detail'] == 'off target':
            # Off taget shots are hollow triangle
            ax.scatter(x1,y1,s=400*calculate_xG(shot), marker='v',edgecolor=disco,facecolor='none',zorder=50)  
        elif shot['detail'] == 'on target':
            # On target shots are filled triangle
                ax.scatter(x1,y1,s=400*calculate_xG(shot), marker='v',color=disco,zorder=75)
        elif shot['detail'] == 'goal':
            # Goals are yellow star
            ax.scatter(x1,y1,s=400*calculate_xG(shot), marker='*',color=buttercup,zorder=100)

ax.scatter(5,5,s=50, marker='v',color='grey',facecolor='none',zorder=50)
ax.scatter(5,10,s=50, marker='v',color='grey',zorder=50)
ax.scatter(35,10,s=50, marker='*',color='grey',zorder=50)
ax.text(8,4,'off target',color='grey')
ax.text(8,9,'on target',color='grey')
ax.text(38,9,'goal',color='grey')

plt.title('xG Shot Map', size=18, pad=4)
ax.text(30, 75, '1st Half xG: ' + str(round(half1_xG,2)), horizontalalignment = 'center', fontsize = 10)
ax.text(90, 75, '2st Half xG: ' + str(round(half2_xG,2)), horizontalalignment = 'center', fontsize = 10)
    
fig.savefig(game + '/shots1.png', dpi=300, bbox_inches='tight')
plt.show()

######################### Surface Shot Map ####################################
# Shot map 2 plots shots accoring to the surface used (foot, head, volley).

# Create plot.
(fig,ax) = createPitch(120,80,'yards','gray')

foot_count1 = 0
footog_count1 = 0
foot_count2 = 0
footog_count2 = 0
head_count1 = 0
headog_count1 = 0
head_count2 = 0
headog_count2 = 0
volley_count1 = 0
volleyog_count1 = 0
volley_count2 = 0
volleyog_count2 = 0

goals1 = 0
goals2 = 0

for i,shot in df_shots.iterrows():
    y1 = shot['y1']
    x1 = shot['x1']
    if shot['surface'] == 'foot':
        if shot['half'] == 1:
            foot_count1 += 1
        else:
            foot_count2 += 1
        if shot['detail'] == 'off target' or shot['detail'] == 'blocked':
            ax.scatter(x1,y1,s=80, marker='v', edgecolor=disco,facecolor='none',zorder=50) 
        elif shot['detail'] == 'on target':
            if shot['half'] == 1:
                footog_count1 += 1
            else:
                footog_count2 += 1
            ax.scatter(x1,y1,s=80, marker='v',color=disco,zorder=75)
        elif shot['detail'] == 'goal':
            if shot['half'] == 1:
                goals1 += 1
                footog_count1 += 1
            else:
                goals2 += 1
                footog_count2 += 1
            ax.scatter(x1,y1,s=150, marker='v',facecolor=buttercup,edgecolor='white',zorder=100)
    elif shot['surface'] == 'head':
        if shot['half'] == 1:
            head_count1 += 1
        else:
            head_count2 +=1
        if shot['detail'] == 'off target' or shot['detail'] == 'blocked':
            ax.scatter(x1,y1,s=80, marker='o',edgecolor=disco,facecolor='none',zorder=50)
        elif shot['detail'] == 'on target':
            if shot['half'] == 1:
                headog_count1 += 1
            else:
                headog_count2 += 1
            ax.scatter(x1,y1,s=80, marker='o',color=disco,zorder=75)
        elif shot['detail'] == 'goal':
            if shot['half'] == 1:
                goals1 += 1
                headog_count1 += 1
            else:
                goals2 += 1
                headog_count2 += 1
            ax.scatter(x1,y1,s=150, marker='o',facecolor=buttercup,edgecolor='white',zorder=100)
    elif shot['surface'] == 'volley':
        if shot['half'] == 1:
            volley_count1 += 1
        else:
            volley_count2 += 1
        if shot['detail'] == 'off target' or shot['detail'] == 'blocked':
            ax.scatter(x1,y1,s=80, marker='s',edgecolor=disco,facecolor='none',zorder=50) 
        elif shot['detail'] == 'on target':
            if shot['half'] == 1:
                volleyog_count1 += 1
            else:
                volleyog_count2 += 1
            ax.scatter(x1,y1,s=80, marker='s',color=disco,zorder=75)
        elif shot['detail'] == 'goal':
            if shot['half'] == 1:
                goals1 += 1
                volleyog_count1 += 1
            else:
                goals2 += 1
                volleyog_count2 += 1
            ax.scatter(x1,y1,s=150, marker='s',facecolor=buttercup,edgecolor='white',zorder=100)

ax.scatter(5,5,s=50, marker='v',edgecolor='grey',facecolor='none',zorder=50)
ax.scatter(5,10,s=50, marker='o',edgecolor='grey',facecolor='none',zorder=50)
ax.scatter(30,10,s=50, marker='s',edgecolor='grey',facecolor='none',zorder=50)
ax.text(8,4,'foot',color='grey')
ax.text(8,9,'head', color='grey')
ax.text(33,9,'volley',color='grey')

foot_count = foot_count1 + foot_count2
head_count = head_count1 + head_count2
volley_count= volley_count1 + volley_count2

footog_count = footog_count1 + footog_count2
headog_count = headog_count1 + headog_count2
volleyog_count = volleyog_count1 + volleyog_count2

half1_shots = foot_count1 + head_count1 + volley_count1
half2_shots = foot_count2 + head_count2 + volley_count2

half1_shotsog = headog_count1 + footog_count1 + volleyog_count1
half2_shotsog = headog_count2 + footog_count2 + volleyog_count2  

total_shots = foot_count + head_count + volley_count
total_shotsog = headog_count + footog_count + volleyog_count 

plt.title('Shots by Surface', size=18, pad=50)
ax.text(60,97,'total (on goal): ' + str(total_shots) + ' (' + str(total_shotsog) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,92,'foot: ' + str(foot_count) + ' (' + str(footog_count) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,87,'head: ' + str(head_count) + ' (' + str(headog_count) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,82,'volley: ' + str(volley_count) + ' (' + str(volleyog_count) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(30, 75, '1st Half', horizontalalignment = 'center', fontsize = 10)
ax.text(90, 75, '2st Half', horizontalalignment = 'center', fontsize = 10)
  

fig.savefig(game + '/shots2.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################### Crosses Visuals #####################################
###############################################################################

# Plot of all crosses, organized by half. 
(fig,ax) = createPitch(120,80,'yards','gray')

crosses1= 0
crosses2 = 0
crosses_complete1 = 0
crosses_complete2 = 0

for i,action in df_crosses.iterrows():
    if action['half'] == 1:
        crosses1 += 1
    else:
        crosses2 += 1
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if action['detail'] == 'complete':
        if action['half'] == 1:
            crosses_complete1 += 1
        else:
            crosses_complete2 += 1
        ax.scatter(x1,y1,marker='o',s=20,color=oracle)
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle))
    else:
        ax.scatter(x1,y1,marker='o',s=20,color=oracle,alpha=0.3)
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle,alpha=0.3))
      

plt.title('Crosses',size=18,pad=10)
ax.text(30, 75, '1st Half', horizontalalignment = 'center', fontsize = 10)
ax.text(90, 75, '2st Half', horizontalalignment = 'center', fontsize = 10)
    
fig.savefig(game + '/crosses.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################### Pass and Dribble Visuals ############################
###############################################################################

######################### Progressive Passes ##################################
# Variables for counting .
prgpass_count1 = 0
prgpasscomplete_count1 = 0
prgpass_count2 = 0
prgpasscomplete_count2 = 0

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_pass.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    dist_pass = np.sqrt((x1-x2)**2+(y1-y2)**2)
    dist_goal1 = np.sqrt((x1-gmx)**2+(y1-gmy)**2)
    dist_goal2 = np.sqrt((x2-gmx)**2+(y2-gmy)**2)
    if dist_pass >= 10 and 0.67*dist_goal1 >= dist_goal2:
        if action['half'] == 1:
            prgpass_count1 += 1
        else:
           prgpass_count2 += 1 
        if action['detail'] == 'complete':
            if action['half'] == 1:
                prgpasscomplete_count1 += 1
            else:
               prgpasscomplete_count2 += 1 
            ax.scatter(x1,y1,marker='o',s=20,color=oracle)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle))
        else:
            ax.scatter(x1,y1,marker='o',s=20,color=oracle,alpha=0.3)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3))

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

plt.title('Progressive Passes',size=18,pad=10)
fig.savefig(game + '/prog_passes.png', dpi=300, bbox_inches='tight')
plt.show()


######################### Progressive Dribbles ################################
(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_dribble.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    dist_dribble = np.sqrt((x1-x2)**2+(y1-y2)**2)
    dist_goal1 = np.sqrt((x1-gmx)**2+(y1-gmy)**2)
    dist_goal2 = np.sqrt((x2-gmx)**2+(y2-gmy)**2)
    if dist_dribble >= 10 and 0.67*dist_goal1 >= dist_goal2:
        if action['detail'] == 'complete':
            ax.scatter(x1,y1,marker='o',s=20,color=oracle)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle))
        else:
            ax.scatter(x1,y1,marker='o',s=20,color=oracle,alpha=0.3)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle, alpha=0.3))

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

plt.title('Progressive Dribbles',size=18,pad=10)
fig.savefig(game + '/prog_dribbles.png', dpi=300, bbox_inches='tight')
plt.show()

######################### Entrances into the Box ##############################
# This variable will count entrances into the box in each half.
boxentrance_count1 = 0
boxentrance_count2 = 0


# Plot
(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df_pass.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if ((x1 < 102) or (y1 < 22) or (y1 > 58)) and ((x2 > 102) and (y2 < 58) and (y2 > 22)):
        if action['detail'] == 'complete':
            if action['half'] == 1:
                boxentrance_count1 += 1
            else:
                boxentrance_count2 += 1
            ax.scatter(x1,y1,marker='o',s=20,color=oracle)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle))
        else: 
            ax.scatter(x1,y1,marker='o',s=20,color=oracle,alpha=0.3)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle,alpha=0.5))
for i,action in df_dribble.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if ((x1 < 102) or (y1 < 22) or (y1 > 58)) and ((x2 > 102) and (y2 < 58) and (y2 > 22)):
        if action['detail'] == 'complete':
            if action['half'] == 1:
                boxentrance_count1 += 1
            else:
                boxentrance_count2 += 1
            ax.scatter(x1,y1,marker='o',s=20,color=disco)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco,ls='dashed'))
        else: 
            ax.scatter(x1,y1,marker='o',s=20,color=disco,alpha=0.3)
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=disco,alpha=0.5,ls='dashed'))
            
plt.title('Entrances into the Box',size=18, pad=5)
ax.plot([30,37],[5,5],color='grey',zorder=10)
ax.plot([65,72],[5,5],color='grey',ls='dashed',zorder=10)
ax.text(40,5,'pass',fontsize=8,ha='left',va='center',color='grey')
ax.text(75,5,'dribble',fontsize=8,ha='left',va='center',color='grey')
ax.text(60,82,'from the run of play',fontsize=8,ha='center')
fig.savefig(game + '/boxentrances.png', dpi=300, bbox_inches='tight')
plt.show()    

###############################################################################
######################### Histogram Plots #####################################
###############################################################################

######################### Pass Completion per Zone ############################

#Create 2 dimensional histograms.
hist_pass = np.histogram2d(df_pass['y1'], df_pass['x1'],bins=(3,4),range=[[0, 80],[0, 120]])
hist_passcomplete = np.histogram2d(df_passcomplete['y1'], df_passcomplete['x1'],bins=(3,4),range=[[0, 80],[0, 120]])   

complete_percent = hist_passcomplete[0]/hist_pass[0]

#Figure 1 uses, roughly, the zones from Keith.
(fig,ax) = createPitch(120,80,'yards','black')

pos = ax.imshow(complete_percent, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Pass Completion Percentage')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
ax.annotate("", xy=(55,5), xytext=(25,5), arrowprops=dict(arrowstyle='->'))
ax.text(20,5,'play',va='center',ha='right')
ax.text(105,13.33,'12',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,40,'11',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,66.67,'10',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,13.33,'9',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,40,'8',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,66.67,'7',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,13.33,'6',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,40,'5',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,66.67,'4',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,13.33,'3',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,40,'2',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,66.67,'1',fontsize=24,alpha=0.5,va='center',ha='center')
 
fig.savefig(game + '/passcompletion.png', dpi=300, bbox_inches='tight')
plt.show()  

# Table to accompany histogram.
(fig,ax) = plt.subplots()
ax.plot([0,10],[6,6],color='grey')
ax.plot([0,10],[5,5],color='grey')
ax.plot([0,10],[4,4],color='grey')
ax.plot([0,10],[3,3],color='grey')
ax.plot([0,10],[2,2],color='grey')
ax.plot([0,10],[1,1],color='grey')
ax.plot([0,10],[0,0],color='grey')
ax.plot([0,10],[-1,-1],color='grey')
ax.plot([5,5],[6,-1],color='grey')
ax.plot([0,0],[6,-1],color='grey')
ax.plot([10,10],[6,-1],color='grey')
ax.text(2.5,5.5,'Zone',ha='center',va='center')
ax.text(7.5,5.5,'Pass Completion',ha='center',va='center')

ax.text(2.5,4.5,'1',ha='center',va='center')
ax.text(2.5,3.5,'2',ha='center',va='center')
ax.text(2.5,2.5,'3',ha='center',va='center')
ax.text(2.5,1.5,'4',ha='center',va='center')
ax.text(2.5,0.5,'5',ha='center',va='center')
ax.text(2.5,-0.5,'6',ha='center',va='center')

ax.text(7.5,4.5,str(round(complete_percent[0,0]*100,0)),ha='center',va='center')
ax.text(7.5,3.5,str(round(complete_percent[1,0]*100,0)),ha='center',va='center')
ax.text(7.5,2.5,str(round(complete_percent[2,0]*100,0)),ha='center',va='center')
ax.text(7.5,1.5,str(round(complete_percent[0,1]*100,0)),ha='center',va='center')
ax.text(7.5,0.5,str(round(complete_percent[1,1]*100,0)),ha='center',va='center')
ax.text(7.5,-0.5,str(round(complete_percent[2,1]*100,0)),ha='center',va='center')
    
plt.axis('off')
ax.set_aspect('equal')
fig.savefig(game + '/passcomplete_table1.png', dpi=300, bbox_inches='tight')
plt.show()

(fig,ax) = plt.subplots()
ax.plot([0,10],[6,6],color='grey')
ax.plot([0,10],[5,5],color='grey')
ax.plot([0,10],[4,4],color='grey')
ax.plot([0,10],[3,3],color='grey')
ax.plot([0,10],[2,2],color='grey')
ax.plot([0,10],[1,1],color='grey')
ax.plot([0,10],[0,0],color='grey')
ax.plot([0,10],[-1,-1],color='grey')
ax.plot([5,5],[6,-1],color='grey')
ax.plot([0,0],[6,-1],color='grey')
ax.plot([10,10],[6,-1],color='grey')
ax.text(2.5,5.5,'Zone',ha='center',va='center')
ax.text(7.5,5.5,'Pass Completion',ha='center',va='center')

ax.text(2.5,4.5,'7',ha='center',va='center')
ax.text(2.5,3.5,'8',ha='center',va='center')
ax.text(2.5,2.5,'9',ha='center',va='center')
ax.text(2.5,1.5,'10',ha='center',va='center')
ax.text(2.5,0.5,'11',ha='center',va='center')
ax.text(2.5,-0.5,'12',ha='center',va='center')

ax.text(7.5,4.5,str(round(complete_percent[0,2]*100,0)),ha='center',va='center')
ax.text(7.5,3.5,str(round(complete_percent[1,2]*100,0)),ha='center',va='center')
ax.text(7.5,2.5,str(round(complete_percent[2,2]*100,0)),ha='center',va='center')
ax.text(7.5,1.5,str(round(complete_percent[0,3]*100,0)),ha='center',va='center')
ax.text(7.5,0.5,str(round(complete_percent[1,3]*100,0)),ha='center',va='center')
ax.text(7.5,-0.5,str(round(complete_percent[2,3]*100,0)),ha='center',va='center')
    
plt.axis('off')
ax.set_aspect('equal')
fig.savefig(game + '/passcomplete_table2.png', dpi=300, bbox_inches='tight')
plt.show()


######################### Total Offensive Touches #############################
# Let's make some new dataframes. For touches I'm focusing on passes, dribbles, and shots.
# Maybe I can add free kicks... I'm torn. 
df_touches = df.loc[(df['event'] == 'pass') | (df['event'] == 'dribble') | (df['event'] == 'shot')]
df_touches_successful = df_touches.loc[(df_touches['event'] == 'shot') | (df_touches['detail'] == 'complete')]


# Create histograms of touches and successful touches
hist_touch = np.histogram2d(df_touches['y1'], df_touches['x1'],bins=(3,4),range=[[0, 80],[0, 120]])  
hist_touch_successful = np.histogram2d(df_touches_successful['y1'], df_touches_successful['x1'],bins=(3,4),range=[[0, 80],[0, 120]])  
total_touches = hist_touch[0]
total_touches_successful = hist_touch_successful[0]

# Plot all touches
(fig,ax) = createPitch(120,80,'yards','black')

pos = ax.imshow(total_touches, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Total Offensive Touches per Zone')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
ax.annotate("", xy=(55,5), xytext=(25,5), arrowprops=dict(arrowstyle='->'))
ax.text(20,5,'play',va='center',ha='right')
ax.text(105,13.33,'12',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,40,'11',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,66.67,'10',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,13.33,'9',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,40,'8',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,66.67,'7',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,13.33,'6',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,40,'5',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,66.67,'4',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,13.33,'3',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,40,'2',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,66.67,'1',fontsize=24,alpha=0.5,va='center',ha='center')
 
fig.savefig(game + '/touchesperzone.png', dpi=300, bbox_inches='tight')
plt.show() 

# Plot successful touches
(fig,ax) = createPitch(120,80,'yards','black')

pos = ax.imshow(total_touches_successful, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Successful Offensive Touches per Zone')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
ax.annotate("", xy=(55,5), xytext=(25,5), arrowprops=dict(arrowstyle='->'))
ax.text(20,5,'play',va='center',ha='right')
ax.text(105,13.33,'12',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,40,'11',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(105,66.67,'10',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,13.33,'9',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,40,'8',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(75,66.67,'7',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,13.33,'6',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,40,'5',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(45,66.67,'4',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,13.33,'3',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,40,'2',fontsize=24,alpha=0.5,va='center',ha='center')
ax.text(15,66.67,'1',fontsize=24,alpha=0.5,va='center',ha='center')
 
fig.savefig(game + '/suctouchesperzone.png', dpi=300, bbox_inches='tight')
plt.show() 

# Table to accompany histogram.
(fig,ax) = plt.subplots()
ax.plot([0,10],[6,6],color='grey')
ax.plot([0,10],[5,5],color='grey')
ax.plot([0,10],[4,4],color='grey')
ax.plot([0,10],[3,3],color='grey')
ax.plot([0,10],[2,2],color='grey')
ax.plot([0,10],[1,1],color='grey')
ax.plot([0,10],[0,0],color='grey')
ax.plot([0,10],[-1,-1],color='grey')
ax.plot([5,5],[6,-1],color='grey')
ax.plot([0,0],[6,-1],color='grey')
ax.plot([10,10],[6,-1],color='grey')
ax.text(2.5,5.5,'Zone',ha='center',va='center')
ax.text(7.5,5.5,'Touches (Successful)',ha='center',va='center')

ax.text(2.5,4.5,'1',ha='center',va='center')
ax.text(2.5,3.5,'2',ha='center',va='center')
ax.text(2.5,2.5,'3',ha='center',va='center')
ax.text(2.5,1.5,'4',ha='center',va='center')
ax.text(2.5,0.5,'5',ha='center',va='center')
ax.text(2.5,-0.5,'6',ha='center',va='center')

ax.text(7.5,4.5,str(total_touches[0,0]) + ' (' + str(total_touches_successful[0,0]) + ')',ha='center',va='center')
ax.text(7.5,3.5,str(total_touches[1,0]) + ' (' + str(total_touches_successful[1,0]) + ')',ha='center',va='center')
ax.text(7.5,2.5,str(total_touches[2,0]) + ' (' + str(total_touches_successful[2,0]) + ')',ha='center',va='center')
ax.text(7.5,1.5,str(total_touches[0,1]) + ' (' + str(total_touches_successful[0,1]) + ')',ha='center',va='center')
ax.text(7.5,0.5,str(total_touches[1,1]) + ' (' + str(total_touches_successful[1,1]) + ')',ha='center',va='center')
ax.text(7.5,-0.5,str(total_touches[2,1]) + ' (' + str(total_touches_successful[2,1]) + ')',ha='center',va='center')
    
plt.axis('off')
ax.set_aspect('equal')
fig.savefig(game + '/touches_table1.png', dpi=300, bbox_inches='tight')
plt.show()

(fig,ax) = plt.subplots()
ax.plot([0,10],[6,6],color='grey')
ax.plot([0,10],[5,5],color='grey')
ax.plot([0,10],[4,4],color='grey')
ax.plot([0,10],[3,3],color='grey')
ax.plot([0,10],[2,2],color='grey')
ax.plot([0,10],[1,1],color='grey')
ax.plot([0,10],[0,0],color='grey')
ax.plot([0,10],[-1,-1],color='grey')
ax.plot([5,5],[6,-1],color='grey')
ax.plot([0,0],[6,-1],color='grey')
ax.plot([10,10],[6,-1],color='grey')
ax.text(2.5,5.5,'Zone',ha='center',va='center')
ax.text(7.5,5.5,'Touches (Successful)',ha='center',va='center')

ax.text(2.5,4.5,'7',ha='center',va='center')
ax.text(2.5,3.5,'8',ha='center',va='center')
ax.text(2.5,2.5,'9',ha='center',va='center')
ax.text(2.5,1.5,'10',ha='center',va='center')
ax.text(2.5,0.5,'11',ha='center',va='center')
ax.text(2.5,-0.5,'12',ha='center',va='center')

ax.text(7.5,4.5,str(total_touches[0,2]) + ' (' + str(total_touches_successful[0,2]) + ')',ha='center',va='center')
ax.text(7.5,3.5,str(total_touches[1,2]) + ' (' + str(total_touches_successful[1,2]) + ')',ha='center',va='center')
ax.text(7.5,2.5,str(total_touches[2,2]) + ' (' + str(total_touches_successful[2,2]) + ')',ha='center',va='center')
ax.text(7.5,1.5,str(total_touches[0,3]) + ' (' + str(total_touches_successful[0,3]) + ')',ha='center',va='center')
ax.text(7.5,0.5,str(total_touches[1,3]) + ' (' + str(total_touches_successful[1,3]) + ')',ha='center',va='center')
ax.text(7.5,-0.5,str(total_touches[2,3]) + ' (' + str(total_touches_successful[2,3]) + ')',ha='center',va='center')
    
plt.axis('off')
ax.set_aspect('equal')
fig.savefig(game + '/touches_table2.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################### Sequence Visuals ####################################
###############################################################################
# Number of sequences    
seqnum = df['seq'].max()
# These count the number of sequences that match certain criteria. For summary 
# tables later. 
completepasses_perseq = np.zeros(seqnum+1)
highpress_count = 0
highpressopp_count = 0
prg_count = 0
prgopp_count = 0
trans_count = 0
transopp_count = 0
opp_count = 0
longstring_count = 0
longstringopp_count = 0
# For denoting entrances into the final third
ballwinx = []
ballwiny = []
ballstartsx = []
ballstartsy = []



for k in range(0,seqnum):
    df_seq = df.loc[df['seq'] == (k+1)]
    df_seq = df_seq.reset_index(drop=True) 
    n = len(df_seq)
    # These variable will be useful in singling out certain types of seq
    highpress = 0
    prg = 0
    trans = 0
    opp = 0
    longstring = 0
    final3 = 0
    # Here I am computing a bunch of variables for the tables
    completepasses_perseq[k] = len(df_seq[((df_seq['event'] == 'pass') | (df_seq['event'] == 'cross') | (df_seq['event'] == 'free kick')) & (df_seq['detail'] == 'complete')])
    if completepasses_perseq[k] >= 5:
        longstring_count += 1
        longstring = 1
    xstart = df_seq['x1'][0]
    ystart = df_seq['y1'][0]
    xend = df_seq['x1'][n-1]
    yend = df_seq['y1'][n-1]
    seq_dist = np.sqrt((xstart-xend)**2+(ystart-yend)**2)
    start_dist =np.sqrt((xstart-gmx)**2+(ystart-gmy)**2)
    end_dist = np.sqrt((gmx-xend)**2+(gmy-yend)**2)
    if end_dist <= 0.5*start_dist:
        prg = 1
        prg_count += 1
    if xstart >= 90:
        highpress = 1
        highpress_count += 1
    elif (xstart < 90) & (prg == 1) & (len(df_seq[df_seq['event'] == 'pass']) < 5):
        trans = 1
        trans_count += 1
    if (len(df_seq[(df_seq['event'] == 'shot') | (df_seq['event'] == 'free kick shot')]) > 0) | (len(df_seq[(df_seq['x1'] >= 102) & (df_seq['y1'] >= 22) & (df_seq['y1'] <=58)]) > 0):
        opp_count += 1
        opp = 1
        if highpress == 1:
            highpressopp_count += 1
        if prg == 1:
            prgopp_count += 1
            if trans == 1:
                transopp_count +=1
        if longstring == 1:
            longstringopp_count += 1
    if df_seq['x1'].max() >= 90:
            final3 = 1
            ballstartsx = np.append(ballstartsx,df_seq['x1'][0])
            ballstartsy = np.append(ballstartsy,df_seq['y1'][0])
            if (df_seq['event'][0] != 'throw in') & (df_seq['event'][0] != 'corner')& (df_seq['event'][0] != 'free kick') & (df_seq['event'][0] != 'free kick shot') & (df_seq['event'][0] != 'goalie restart'):
                ballwinx = np.append(ballwinx,df_seq['x1'][0])
                ballwiny = np.append(ballwiny,df_seq['y1'][0])
    # Only plots chances that enter final third.         
    if opp == 1 or final3 == 1: 
        # Draw the pitch
        (fig,ax) = createPitch(120,80,'yards','gray')
        ax.plot([30,30],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        ax.plot([90,90],[0,80],ls='dashed',color='grey',zorder=0,alpha=0.5)
        # Create key
        # throw in
        ax.plot([-40,-30],[75,75],color=oracle,ls='dotted', zorder=25)
        ax.text(-25,75,'throw in',color='grey',fontsize=6,va='center')
        # pass
        ax.plot([-40,-30],[70,70],color=oracle,zorder=25)
        ax.text(-25,70,'pass',color='grey',fontsize=6,va='center')
        # dribble
        ax.plot([-40,-30],[65,65],color=oracle,ls='dashed',zorder=25)
        ax.text(-25,65,'dribble',color='grey',fontsize=6,va='center')
        # complete
        ax.plot([-40,-30],[55,55],color=oracle,zorder=25)
        ax.text(-25,55,'complete',color='grey',fontsize=6,va='center')
        # incomplete
        ax.annotate("", xy=(-28,50), xytext=(-38,50), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle))
        ax.text(-25,50,'incomplete',color='grey',fontsize=6,va='center')
        # progressive
        ax.plot([-40,-30],[40,40],color=oracle,zorder=25)
        ax.plot([-40,-30],[40,40],color=oracle,zorder=10,linewidth=5,alpha=0.25)
        ax.text(-25,40,'progressive',color='grey',fontsize=6,va='center')
        # goal
        ax.scatter(-38,30,marker='o',s=40,color=buttercup,zorder=50)
        ax.annotate("", xy=(-28,30), xytext=(-38,30), arrowprops=dict(arrowstyle='->', color=buttercup))
        ax.text(-25,30,'goal',color='grey',fontsize=6,va='center')
        # shot
        ax.scatter(-38,25,marker='o',s=40,edgecolor=disco,facecolor='white',zorder=50)
        ax.annotate("", xy=(-28,25), xytext=(-38,25), arrowprops=dict(arrowstyle='->', color=disco))
        ax.text(-25,25,'shot',color='grey',fontsize=6,va='center')
        # ball lost
        ax.scatter(-35,15,marker='o',s=60,edgecolor=fuzz,facecolor='white',zorder=25)
        ax.scatter(-35,15,marker='x',s=40,color=fuzz,zorder=30)
        ax.text(-25,15,'ball lost',color='grey',fontsize=6,va='center')
        # Plot the chance
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
                    dist_pass = np.sqrt((x1-x2)**2+(y1-y2)**2)
                    dist_goal1 = np.sqrt((x1-gmx)**2+(y1-gmy)**2)
                    dist_goal2 = np.sqrt((x2-gmx)**2+(y2-gmy)**2)
                    if (dist_pass >= 10) and (0.67*dist_goal1 >= dist_goal2): # I'm not currently sold on this definition of progressive pass. 
                        ax.plot([x1,x2],[y1,y2],color=oracle,zorder=10,linewidth=5,alpha=0.25)
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
                    dist_pass = np.sqrt((x1-x2)**2+(y1-y2)**2)
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
                    dist_pass = np.sqrt((x1-x2)**2+(y1-y2)**2)
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
                    dist_dribble = np.sqrt((x1-x2)**2+(y1-y2)**2)
                    dist_goal1 = np.sqrt((x1-gmx)**2+(y1-gmy)**2)
                    dist_goal2 = np.sqrt((x2-gmx)**2+(y2-gmy)**2)
                    if (dist_pass >= 10) and (0.67*dist_goal1 >= dist_goal2):
                        ax.plot([x1,x2],[y1,y2],color=oracle,zorder=10,linewidth=5,alpha=0.25)
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
        fig.savefig(game + '/seqs/seq' + num + '.png', dpi=300, bbox_inches='tight')
        plt.show()


###############################################################################
######################### Ball Starts and Ball Wins ###########################
###############################################################################

(fig,ax) = createPitch(120,80,'yards','gray')

for i,action in df.iterrows():
    if i >= 1:
        if df['seq1'][i-1] == 1:
            x1 = action['x1']
            y1 = action['y1']
            ax.scatter(x1,y1,marker='o',edgecolor=oracle,facecolor = 'white',s=30,zorder=20)
            if (action['event'] != 'throw in') & (action['event'] != 'free kick') & (action['event'] != 'corner') & (action['event'] != 'free kick shot') & (action['event'] != 'goalie restart'):
                ax.scatter(x1,y1,marker='o',edgecolor=oracle,facecolor = oracle,s=30,zorder=30)
ax.scatter(ballstartsx,ballstartsy,marker='o',s=40,edgecolor=fuzz,facecolor='white',zorder=40)
ax.scatter(ballwinx,ballwiny,marker='o',s=40,edgecolor=fuzz,facecolor=fuzz,zorder=50)
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
# Make graph key
ax.scatter(5,-5,marker='o',s=40,color='grey')
ax.text(10,-5,'run of play',va='center')
ax.scatter(45,-5,marker='o',s=40,edgecolor='grey',facecolor='white')
ax.text(50,-5,'restart',va='center')
ax.scatter(75,-5,marker='o',s=40,color=fuzz)
ax.text(80,-5,'promising attack',va='center')


plt.title('Sequence Starting Locations',size=18, pad=5)
fig.savefig(game + '/seqtarts.png', dpi=300, bbox_inches='tight')

plt.show()

###############################################################################
######################### Data Tables #########################################
###############################################################################

######################## Basic Summary ########################################

df['boxtouch1'] = 0
df['boxtouch2'] = 0
if directionofplay == 'RtoL':
    df.loc[(df['x1'] <= 18) & (df['y1'] >= 22) & (df['y1'] <= 58) & (df['half'] == 1), 'boxtouch1'] = 1
    df.loc[(df['x1'] <= 18) & (df['y1'] >= 22) & (df['y1'] <= 58) & (df['half'] == 2), 'boxtouch2'] = 1
else:
    df.loc[(df['x1'] >= 102) & (df['y1'] >= 22) & (df['y1'] <= 58) & (df['half'] == 1), 'boxtouch1'] = 1
    df.loc[(df['x1'] >= 102) & (df['y1'] >= 22) & (df['y1'] <= 58) & (df['half'] == 2), 'boxtouch2'] = 1

(fig,ax) = plt.subplots()
ax.plot([-4,10],[4,4],color='grey')
ax.plot([-4,10],[3,3],color='grey')
ax.plot([-4,10],[2,2],color='grey')
ax.plot([-4,10],[1,1],color='grey')
ax.plot([-4,10],[0,0],color='grey')
ax.plot([-4,10],[-1,-1],color='grey')
ax.plot([-4,10],[-2,-2],color='grey')
ax.plot([-4,10],[-3,-3],color='grey')
ax.plot([-4,10],[-4,-4],color='grey')
ax.plot([5,5],[-4,5],color='grey')
ax.plot([7.5,7.5],[-4,5],color='grey')
ax.text(6.25,4.5,'1st Half', ha='center',va='center')
ax.text(8.75,4.5,'2st Half',ha='center',va='center')
ax.text(4.5,3.5,'Goals',ha='right',va='center')
ax.text(4.5,2.5,'Shots (on goal)',ha='right',va='center')
ax.text(4.5,1.5,'xG per shot',ha='right',va='center')
ax.text(4.5,0.5,'Crosses (complete)',ha='right',va='center')
ax.text(4.5,-0.5,'Progressive passes (complete)',ha='right',va='center')
ax.text(4.5,-1.5,'Entrances into the box',ha='right',va='center')
ax.text(4.5,-2.5,'Touches in the box',ha='right',va='center')
ax.text(4.5,-3.5,'Corners',ha='right',va='center')

ax.text(6.25,3.5,str(goals1),ha='center',va='center')
ax.text(8.75,3.5,str(goals2),ha='center',va='center')

ax.text(6.25,2.5,str(half1_shots) + ' (' + str(half1_shotsog) + ')',ha='center',va='center')
ax.text(8.75,2.5,str(half2_shots) + ' (' + str(half2_shotsog) + ')',ha='center',va='center')

ax.text(6.25,1.5,str(round(half1_xG/half1_shots,2)),ha='center',va='center')
ax.text(8.75,1.5,str(round(half2_xG/half2_shots,2)),ha='center',va='center')

ax.text(6.25,0.5,str(crosses1) + ' (' + str(crosses_complete1) + ')',ha='center',va='center')
ax.text(8.75,0.5,str(crosses2) + ' (' + str(crosses_complete2) + ')',ha='center',va='center')

ax.text(6.25,-0.5,str(prgpass_count1) + ' (' + str(prgpasscomplete_count1) + ')',ha='center',va='center')
ax.text(8.75,-0.5,str(prgpass_count2) + ' (' + str(prgpasscomplete_count2) + ')',ha='center',va='center')

ax.text(6.25,-1.5,str(boxentrance_count1),ha='center',va='center')
ax.text(8.75,-1.5,str(boxentrance_count2),ha='center',va='center')

ax.text(6.25,-2.5,str(df['boxtouch1'].sum()),ha='center',va='center')
ax.text(8.75,-2.5,str(df['boxtouch2'].sum()),ha='center',va='center')

ax.text(6.25,-3.5,str(len(df[(df['event'] == 'corner') & (df['half'] == 1)])),ha='center',va='center')
ax.text(8.75,-3.5,str(len(df[(df['event'] == 'corner') & (df['half'] == 2)])),ha='center',va='center')

plt.axis('off')
ax.set_aspect('equal')
plt.title('Summary',size=18,pad=4)
fig.savefig(game + '/summarytable.png', dpi=300, bbox_inches='tight')
plt.show()

######################## Sequence Summary #####################################

(fig,ax) = plt.subplots()
ax.plot([0,17],[5,5],color='grey')
ax.plot([0,17],[4,4],color='grey')
ax.plot([0,17],[3,3],color='grey')
ax.plot([0,17],[2,2],color='grey')
ax.plot([0,17],[1,1],color='grey')
ax.plot([10,10],[1,6],color='grey')
ax.plot([7,7],[1,6],color='grey')
ax.text(8.5,5.5,'Total',ha='center',va='center')
ax.text(13.5,5.5,'Create Opportunity',ha='center',va='center')
ax.text(6.5,4.5,'Long passing string',ha='right',va='center')
ax.text(6.5,3.5,'High press',ha='right',va='center')
ax.text(6.5,2.5,'Progressive',ha='right',va='center')
ax.text(6.5,1.5,'Transitional',ha='right',va='center')

ax.text(8.5,4.5,str(longstring_count),ha='center',va='center')
ax.text(13.5,4.5,str(longstringopp_count),ha='center',va='center')

ax.text(8.5,3.5,str(highpress_count),ha='center',va='center')
ax.text(13.5,3.5,str(highpressopp_count),ha='center',va='center')

ax.text(8.5,2.5,str(prg_count),ha='center',va='center')
ax.text(13.5,2.5,str(prgopp_count),ha='center',va='center')

ax.text(8.5,1.5,str(trans_count),ha='center',va='center')
ax.text(13.5,1.5,str(transopp_count),ha='center',va='center')



plt.axis('off')
ax.set_aspect('equal')
plt.title('Sequence Summary',size=18,pad=4)
fig.savefig(game + '/seqtable.png', dpi=300, bbox_inches='tight')
plt.show()




