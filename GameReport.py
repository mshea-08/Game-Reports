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
from FCPython import createZonalPitch
import seaborn as sns

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
redish = '#a6230c'

###############################################################################
######################### Game Specs ##########################################
###############################################################################
# Enter game folder. 
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
df = pd.read_excel(game + '/GameData.xlsx')

# This is for scrimmages that play 3 periods instead of two halfs. First half data
# becomes 1st and 3rd periods.
df.loc[df['half'] == 3, 'half'] = 1

# To denote an offsides pass while tagging, I'll tag the pass detail as incomplete 
# and the surface as foot (surface is empty for typical passes). This code switches them
# so the detail becomes offsides.
df.loc[(df['event'] == 'pass') & (df['surface'] == 'foot'), 'detail'] = 'offsides'

# In older data files 'ball lost' will have detail 'complete'. This will change ball lost
# to have the detail 'incomplete'.
df.loc[(df['event'] == 'ball lost'), 'detail'] = 'incomplete'

# Flip y-coordinates (always have to do this).
df['y1'] = 80 - df['y1']
df['y2'] = 80 - df['y2']

# Ultimately, all play visuals will run from left to right unless it's the shot or crosses
# maps. 
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

# Make seq1 col, which equals 1 when there is a dispossession.
df['seq1'] = 0
df.loc[(df['detail'] == 'incomplete') | (df['detail'] == 'blocked') | (df['event'] == 'ball lost') | (df['event'] == 'shot') | (df['event'] == 'free kick shot') | (df['detail'] == 'offsides'),'seq1'] = 1

# Creates a sequence column.
df['seq'] = df['seq1'].cumsum() - df['seq1']

# Add starting and ending zone to events. 
xbins = [-5,30,60,90,125]
ybins = [-5,26.67,53.33,85] # bins are extended to include tags that fall slightly outside

for i in range(len(xbins)-1):
    for j in range(len(ybins)-1):
        df.loc[((df['x1'] > xbins[i]) & (df['x1'] <= xbins[i+1]) & 
                (df['y1'] > ybins[j]) & (df['y1']<= ybins[j+1])),'zone start'] = (j+1) + 3*i
        df.loc[((df['x2'] > xbins[i]) & (df['x2'] <= xbins[i+1]) & 
                (df['y2'] > ybins[j]) & (df['y2']<= ybins[j+1])),'zone end'] = (j+1) + 3*i

# Clean up corner kick and throw in location...
df.loc[(df['event'] == 'corner'), 'x1'] = 120
df.loc[(df['event'] == 'corner') & (df['y1'] > 40), 'y1'] = 80
df.loc[(df['event'] == 'corner') & (df['y1'] < 40), 'y1'] = 0

df.loc[(df['event'] == 'throw in') & (df['y1'] > 40), 'y1'] = 80
df.loc[(df['event'] == 'throw in') & (df['y1'] < 40), 'y1'] = 0
    
######################## Moving Pass Length to main df ########################

# Add pass length 
df.loc[(df['event'] == 'pass') | (df['event'] == 'cross'), 'pass_distance'] = np.sqrt((df['x2'] - df['x1'])**2 + (df['y2'] - df['y1'])**2)

#manually defining bins for pass lengths. using FB Ref bins
bins = [0, 15, 30, np.inf]  #short, medium, long
labels = ['short', 'medium', 'long']

#creates a new column that categorizes each pass in 'pass_distance' into the bins, and labeling them "short," "medium," or "long".
df['pass_type'] = pd.cut(df['pass_distance'], bins=bins, labels=labels, right=False) #pd.cut --> "categorize". right = False excludes right endpoint. e.g., [5, 15) will include x >= 5 and x < 15.


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

# Make df of shots on target, off target, and goals.
df_shots_ontarget = df_shots.loc[df_shots['detail'] == 'on target']
df_shots_goal = df_shots.loc[df_shots['detail'] == 'goal']
df_shots_offtarget = df_shots.loc[df_shots['detail'] == 'off target']
df_shots_blocked = df_shots.loc[df_shots['detail'] == 'blocked']

######################### Passes and crosses Dataframe ########################
# Create a df of passes and crosses.
df_pass = df.loc[(df['event'] == 'pass') | (df['event'] == 'cross')]
df_pass = df_pass.reset_index(drop=True)

# Create df of only successful passes and crosses.
df_passcomplete = df_pass.loc[df_pass['detail'] == 'complete']
df_passcomplete = df_passcomplete.reset_index(drop=True)

df_short_pass = df_pass.loc[df_pass['pass_type'] == 'short']
df_medium_pass = df_pass.loc[df_pass['pass_type'] == 'medium']
df_long_pass = df_pass.loc[df_pass['pass_type'] == 'long']


######################## Dribbles Dataframe ###################################
# Create a df of dribbles.
df_dribble = df.loc[(df['event'] == 'dribble')]
df_dribble = df_dribble.reset_index(drop=True)

# Create df of only successful dribbles.
df_dribblecomplete = df_dribble.loc[df_dribble['detail'] == 'complete']
df_dribblecomplete = df_dribblecomplete.reset_index(drop=True)


###############################################################################
######################### Shots Visuals #######################################
###############################################################################

######################### xG Model ############################################
# This model is compiled in xGmodel.py. It currently uses shot data from the previous
# three seasons. This model does NOT automatically update. I will incorporate this 
# feature later. 

def calculate_xG(dist,theta):    
   xG = 1-1/(1+np.exp(-0.3454-0.0924*dist+0.2863*theta)) 
   return xG  

# Add xG column to dataframe
df_shots['xG'] = calculate_xG(df_shots['distance'],df_shots['theta'])

######################### xG Shot Map #########################################
# Shot map 1 plots shots proportionally to their xG. It also computes total xG for 
# each half. 

half1_xG = df_shots.loc[(df_shots['half'] == 1) & (df_shots['detail'] != 'blocked'), 'xG'].sum()
half2_xG = df_shots.loc[(df_shots['half'] == 2) & (df_shots['detail'] != 'blocked'), 'xG'].sum()
tot_xG = half1_xG + half2_xG

# Create plot.
(fig,ax) = createPitch(120,80,'yards','gray')
ax.scatter(df_shots_offtarget['x1'],df_shots_offtarget['y1'],s=400*calculate_xG(df_shots_offtarget['distance'], df_shots_offtarget['theta']), marker='v',edgecolor=disco,facecolor='none',zorder=50) 
ax.scatter(df_shots_ontarget['x1'],df_shots_ontarget['y1'],s=400*calculate_xG(df_shots_ontarget['distance'], df_shots_ontarget['theta']), marker='v',edgecolor=disco,facecolor=disco,zorder=50)  
ax.scatter(df_shots_goal['x1'],df_shots_goal['y1'],s=400*calculate_xG(df_shots_goal['distance'], df_shots_goal['theta']), marker='*',color=buttercup,zorder=50)  

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

# Some shooting variables for later
goals1 = len(df_shots.loc[(df_shots['half'] == 1) & (df_shots['detail'] == 'goal')])
goals2 = len(df_shots.loc[(df_shots['half'] == 2) & (df_shots['detail'] == 'goal')])

half1_shots = len(df_shots.loc[df_shots['half'] == 1])
half1shots_noblocks = len(df_shots.loc[(df_shots['half'] == 1) & (df_shots['detail'] != 'blocked')])
half1_shotsog = len(df_shots.loc[df_shots['half'] == 1]) - len(df_shots_offtarget.loc[(df_shots['half'] == 1)]) - len(df_shots_blocked.loc[(df_shots['half'] == 1)])

half2_shots = len(df_shots.loc[df_shots['half'] == 2])
half2shots_noblocks = len(df_shots.loc[(df_shots['half'] == 2) & (df_shots['detail'] != 'blocked')])
half2_shotsog = len(df_shots.loc[df_shots['half'] == 2]) - len(df_shots_offtarget.loc[(df_shots['half'] == 2)]) - len(df_shots_blocked.loc[(df_shots['half'] == 2)])

######################### Surface Shot Map ####################################
# Shot map 2 plots shots accoring to the surface used (foot, head, volley).

# Count occurances 
foot = len(df_shots.loc[(df_shots['surface'] == 'foot')])
foot_ongoal = len(df_shots.loc[(df_shots['surface'] == 'foot') & ((df_shots['detail'] == 'on target') | (df_shots['detail'] == 'goal'))])

head = len(df_shots.loc[(df_shots['surface'] == 'head')])
head_ongoal = len(df_shots.loc[(df_shots['surface'] == 'head') & ((df_shots['detail'] == 'on target') | (df_shots['detail'] == 'goal'))])

volley = len(df_shots.loc[(df_shots['surface'] == 'volley')])
volley_ongoal = len(df_shots.loc[(df_shots['surface'] == 'volley') & ((df_shots['detail'] == 'on target') | (df_shots['detail'] == 'goal'))])

shot_total = len(df_shots)
ongoal_total = len(df_shots_ontarget) + len(df_shots_goal)

# Create plot.
(fig,ax) = createPitch(120,80,'yards','gray')

ax.scatter(df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'off target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'off target'), 'y1'],marker='v',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'blocked'), 'x1'],df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'blocked'), 'y1'],marker='v',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'on target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'on target'), 'y1'],marker='v',s=50,edgecolor=disco,facecolor=disco)
ax.scatter(df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'goal'), 'x1'],df_shots.loc[(df_shots['surface'] == 'foot') & (df_shots['detail'] == 'goal'), 'y1'],marker='v',s=50,edgecolor=buttercup,facecolor=buttercup)

ax.scatter(df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'off target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'off target'), 'y1'],marker='o',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'blocked'), 'x1'],df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'blocked'), 'y1'],marker='o',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'on target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'on target'), 'y1'],marker='o',s=50,edgecolor=disco,facecolor=disco)
ax.scatter(df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'goal'), 'x1'],df_shots.loc[(df_shots['surface'] == 'head') & (df_shots['detail'] == 'goal'), 'y1'],marker='o',s=50,edgecolor=buttercup,facecolor=buttercup)

ax.scatter(df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'off target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'off target'), 'y1'],marker='s',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'blocked'), 'x1'],df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'blocked'), 'y1'],marker='s',s=50,edgecolor=disco,facecolor='none')
ax.scatter(df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'on target'), 'x1'],df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'on target'), 'y1'],marker='s',s=50,edgecolor=disco,facecolor=disco)
ax.scatter(df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'goal'), 'x1'],df_shots.loc[(df_shots['surface'] == 'volley') & (df_shots['detail'] == 'goal'), 'y1'],marker='s',s=50,edgecolor=buttercup,facecolor=buttercup)

ax.scatter(5,5,s=50, marker='v',edgecolor='grey',facecolor='none',zorder=50)
ax.scatter(5,10,s=50, marker='o',edgecolor='grey',facecolor='none',zorder=50)
ax.scatter(30,10,s=50, marker='s',edgecolor='grey',facecolor='none',zorder=50)
ax.text(8,4,'foot',color='grey')
ax.text(8,9,'head', color='grey')
ax.text(33,9,'volley',color='grey')

plt.title('Shots by Surface', size=18, pad=50)
ax.text(60,97,'total (on goal): ' + str(shot_total) + ' (' + str(ongoal_total) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,92,'foot: ' + str(foot) + ' (' + str(foot_ongoal) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,87,'head: ' + str(head) + ' (' + str(head_ongoal) + ')', horizontalalignment = 'center', fontsize = 10)
ax.text(60,82,'volley: ' + str(volley) + ' (' + str(volley_ongoal) + ')', horizontalalignment = 'center', fontsize = 10)
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

(fig,ax) = createZonalPitch(120,80,'yards','gray')

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

plt.title('Progressive Passes',size=18,pad=10)
fig.savefig(game + '/prog_passes.png', dpi=300, bbox_inches='tight')
plt.show()


######################### Progressive Dribbles ################################
(fig,ax) = createZonalPitch(120,80,'yards','gray')

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
            ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', 
                                                                        color=oracle, alpha=0.3))

plt.title('Progressive Dribbles',size=18,pad=10)
fig.savefig(game + '/prog_dribbles.png', dpi=300, bbox_inches='tight')
plt.show()

######################### All Dribbles ########################################
(fig,ax) = createZonalPitch(120,80,'yards','gray')

for i,action in df_dribble.iterrows():
    x1 = action['x1']
    x2 = action['x2']
    y1 = action['y1']
    y2 = action['y2']
    if action['detail'] == 'complete':
        ax.scatter(x1,y1,marker='o',s=20,color=oracle)
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', color=oracle))
    else:
        ax.scatter(x1,y1,marker='o',s=20,color=oracle,alpha=0.3)
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->', 
                                                                    color=oracle, alpha=0.3))

plt.title('Dribbles',size=18,pad=10)
fig.savefig(game + '/dribbles.png', dpi=300, bbox_inches='tight')
plt.show()

######################### Long Passes #########################################

# Add pass ratio column to long passes df.
xdiff = df_long_pass['x2'] - df_long_pass['x1']
ydiff = df_long_pass['y2'] - df_long_pass['y1']

df_long_pass['pass_ratio'] = xdiff/abs(ydiff)

# Create plot
(fig, ax) = createZonalPitch(120, 80, 'yards', 'gray')

for i, action in df_long_pass.iterrows():
    x1 = action['x1']
    y1 = action['y1']
    x2 = action['x2']
    y2 = action['y2']

    #plots the passes
    if action['detail'] == 'complete':
        # color based on pass type
        if action['pass_ratio'] >= 1 :
            color = redish #can use the alredy implemented colors just wanted to see clearly
        elif action['pass_ratio'] < 1 and action['pass_ratio'] >= -0.5 :
            color = oracle  
        else:
            color = loulou  #any "other" passes (basically just long backwards passes)
        ax.scatter(x1, y1, marker='o', s=20, color=color)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', color=color))
    else:
        # all incomplete passes are grey
        ax.scatter(x1, y1, marker='o', s=20, color='grey', alpha=0.3)
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='->', 
                                                                      color='grey', alpha=0.3))

plt.title('Long Passes', size=18, pad=10)
fig.savefig(game + '/long_passes.png',dpi=300, bbox_inches='tight')
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
(fig,ax) = createZonalPitch(120,80,'yards','black')

pos = ax.imshow(complete_percent, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Pass Completion Percentage')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
 
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
(fig,ax) = createZonalPitch(120,80,'yards','black')

pos = ax.imshow(total_touches, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Total Offensive Touches per Zone')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
 
fig.savefig(game + '/touchesperzone.png', dpi=300, bbox_inches='tight')
plt.show() 

# Plot successful touches
(fig,ax) = createZonalPitch(120,80,'yards','black')

pos = ax.imshow(total_touches_successful, extent=[0,120,0,80], aspect='auto',cmap=plt.cm.BuPu)
fig.colorbar(pos, ax=ax)
ax.set_title('Successful Offensive Touches per Zone')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
 
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

# Start by creating sequence summary dataframe
seq_df = dict()

seq_num =df['seq'].max()

for k in range(0,seq_num):
    seq_df[str(k)] = df.loc[df['seq'] == (k)]
    seq_df[str(k)] = seq_df[str(k)].reset_index(drop=True) 
    
seq_summary_df = pd.DataFrame()

for i in range(seq_num):
    last = len(seq_df[str(i)])-1
    
    xstart = seq_df[str(i)]['x1'][0]
    ystart = seq_df[str(i)]['y1'][0]
    start_action = seq_df[str(i)]['event'][0]
    zone_start = seq_df[str(i)]['zone start'][0]
    
    xend = seq_df[str(i)]['x1'][last]
    yend = seq_df[str(i)]['y1'][last]
    end_action = seq_df[str(i)]['event'][last]
    end_action_x2 = seq_df[str(i)]['x2'][last]
    end_action_y2 = seq_df[str(i)]['y2'][last]
    zone_end = seq_df[str(i)]['zone start'][last]
    
    if (end_action == 'shot') or (end_action == 'free kick shot'):
        actions = last + 1
    else:
        actions = last
    
    passes = len(seq_df[str(i)].loc[(seq_df[str(i)]['event'] == 'pass') | (seq_df[str(i)]['event'] == 'free kick') |
                                (seq_df[str(i)]['event'] == 'corner kick') | (seq_df[str(i)]['event'] == 'goalie restart')])
    if (end_action == 'pass') or (end_action == 'free kick') or (end_action == 'corner kick') or (end_action == 'goalie restart'):
        passes += -1
    
    if seq_df[str(i)]['x1'].max() > 90:
        final_third = 'yes'
    else:
        final_third = 'no'
    
    if len(seq_df[str(i)].loc[(seq_df[str(i)]['x1'] >= 102) & (seq_df[str(i)]['y1'] > 22) & (seq_df[str(i)]['y1'] < 58)]) > 0:
        box = 'yes'
    else:
        box = 'no'
    
    if (end_action == 'shot') or (end_action == 'free kick shot'):
        shot_result = seq_df[str(i)]['detail'][last]
    else:
        shot_result = np.nan
        
    if (end_action == 'shot') or (end_action == 'free kick shot') or box == 'yes':
        opp = 'yes'
    else:
        opp = 'no'
    
    row = pd.DataFrame({
        'x start' : [xstart],
        'y start' : [ystart],
        'zone start' : [zone_start],
        'x end' : [xend], 
        'y end' : [yend],
        'zone end': [zone_end],
        'starting action' : [start_action],
        'ending action' : [end_action],
        'ending action x2' : [end_action_x2],
        'ending action y2' : [end_action_y2],
        'successful actions' : [actions],
        'completed passes' : [passes],
        'final third entrance' : [final_third],
        'box entrance' : [box],
        'shot result' : [shot_result],
        'opp' : [opp]
        })
    
    seq_summary_df = pd.concat([seq_summary_df, row], 
                   ignore_index=True)


###############################################################################
#################### Opportunity Sequence Visuals #############################
###############################################################################


for k in range(0,seq_num):
    if (seq_summary_df['box entrance'][k] == 'yes') or (seq_summary_df['ending action'][k] == 'shot') or (seq_summary_df['ending action'][k] == 'free kick shot'): 
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
        for i,action in seq_df[str(k)].iterrows():
            x1 = action['x1']
            y1 = action['y1']
            if (action['event'] == 'pass') or (action['event'] == 'cross'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    if seq_df[str(k)]['event'][i+1] == 'free kick' or seq_df[str(k)]['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = seq_df[str(k)]['x1'][i+1]
                        y2 = seq_df[str(k)]['y1'][i+1]
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
                    x2 = seq_df[str(k)]['x1'][i+1]
                    y2 = seq_df[str(k)]['y1'][i+1]
                    ax.plot([x1,x2],[y1,y2],color=oracle,zorder=25, ls='dotted')
                    dist_pass = np.sqrt((x1-x2)**2+(y1-y2)**2)
                else:
                    x2 = action['x2']
                    y2 = action['y2']
                    ax.annotate("", xy=(x2,y2), xytext=(x1,y1), arrowprops=dict(arrowstyle='->',alpha=0.5, color=oracle, ls='dotted'))
            elif (action['event'] == 'corner') or (action['event'] == 'free kick') or (action['event'] == 'goalie restart'):
                ax.text(x1,y1,str(count),fontsize=4,horizontalalignment = 'center',verticalalignment = 'center',zorder=52+5*i)
                ax.scatter(x1,y1,marker='o',s=60,edgecolor='black',facecolor='white',zorder=50+5*i)
                count += 1
                if action['detail'] == 'complete':
                    x2 = seq_df[str(k)]['x1'][i+1]
                    y2 = seq_df[str(k)]['y1'][i+1]
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
                    if seq_df[str(k)]['event'][i+1] == 'free kick' or seq_df[str(k)]['event'][i+1] == 'free kick shot':
                        x2 = action['x2']
                        y2 = action['y2']
                    else:
                        x2 = seq_df[str(k)]['x1'][i+1]
                        y2 = seq_df[str(k)]['y1'][i+1]
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
######################### Sequence Starts #####################################
###############################################################################

# Dataframs of live ball start (okay, so it includes free kicks too) and throw in starts.
throw_in_start_df = seq_summary_df.loc[seq_summary_df['starting action'] == 'throw in']
live_ball_start_df = seq_summary_df.loc[(seq_summary_df['starting action'] != 'throw in') & (seq_summary_df['starting action'] != 'corner') & (seq_summary_df['starting action'] != 'goalie restart')]

# Plot starts as scatter data.
(fig,ax) = createZonalPitch(120,80,'yards','gray')

ax.scatter(throw_in_start_df['x start'],throw_in_start_df['y start'],color='grey')
ax.scatter(live_ball_start_df['x start'],live_ball_start_df['y start'],color=disco)

plt.title('Possession Starting Locations',size=18, pad=5)
fig.savefig(game + '/poss_starts.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram of x starting locations of just live ball starts.
sns.histplot(data=live_ball_start_df,x='x start',hue='opp',edgecolor='white',
             bins=24,multiple='stack') 
plt.xlabel('field x position')
plt.ylabel('frequency')
plt.title('Possession Starting Locations, x values',size=18, pad=5)
fig.savefig(game + '/poss_starts_xhist.png', dpi=300, bbox_inches='tight')
plt.show()

# add column to df that is shot or box entrance

# Histogram of y starting locations of just live ball starts.
sns.histplot(data=live_ball_start_df,x='y start',hue='opp',edgecolor='white',
             bins=16,multiple='stack') 
plt.xlabel('field y position')
plt.ylabel('frequency')
plt.title('Possession Starting Locations, y values',size=18, pad=5)
fig.savefig(game + '/poss_starts_yhist.png', dpi=300, bbox_inches='tight')
plt.show()



###############################################################################
######################### Sequence Ends #######################################
###############################################################################

ending_location_noshots_df = seq_summary_df.loc[(seq_summary_df['ending action'] != 'shot') & 
                                                (seq_summary_df['ending action'] != 'free kick shot') &
                                                (seq_summary_df['ending action'] != 'throw in') &
                                                (seq_summary_df['ending action'] != 'corner')]
ending_location_onlyshots_df = seq_summary_df.loc[(seq_summary_df['ending action'] == 'shot') | 
                                                  (seq_summary_df['ending action'] == 'free kick shot')]

(fig,ax) = createZonalPitch(120,80,'yards','gray')

ax.scatter(ending_location_noshots_df['x end'],ending_location_noshots_df['y end'],color=redish)
ax.scatter(ending_location_onlyshots_df['x end'],ending_location_onlyshots_df['y end'],marker='*',color=buttercup)

plt.title('Possession Ending Locations',size=18, pad=5)
fig.savefig(game + '/poss_ends.png', dpi=300, bbox_inches='tight')

plt.show()

# Histogram of x ending locations of just live ball starts.
sns.histplot(data=ending_location_noshots_df, x='x end',
             edgecolor='white',bins=24) 
plt.xlabel('field x position')
plt.ylabel('frequency')
plt.title('Non-shot Possession Ending Locations, x values',size=18, pad=5)
fig.savefig(game + '/poss_ends_xhist.png', dpi=300, bbox_inches='tight')
plt.show()

# Histogram of y starting locations of just live ball starts.
sns.histplot(data=ending_location_noshots_df, x='y end',hue='final third entrance',
             multiple='stack',edgecolor='white',bins=16) 
plt.xlabel('field y position')
plt.ylabel('frequency')
plt.title('Non-shot Possession Ending Locations, y values',size=18, pad=5)
fig.savefig(game + '/poss_ends_yhist.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################### Sequence trajectories ###############################
###############################################################################

# start in defensive quarter
(fig,ax) = createZonalPitch(120,80,'yards','gray')

for i,seq in seq_summary_df.iterrows():
    if (seq['zone start'] <= 3):
        x1 = seq['x start']
        y1 = seq['y start']
        x2 = seq['x end']
        y2 = seq['y end']
        ax.plot([x1,x2],[y1,y2],ls='dashed',color='grey')
        ax.scatter(x1,y1,s=20,color='black')
        if (seq['ending action'] == 'shot') or (seq['ending action'] == 'free kick shot'):
            ax.scatter(x2,y2,s=20,marker='*',color=buttercup)
        else:
            ax.scatter(x2,y2,s=20,color=redish)

plt.title('Sequence Start and Ends, 1/4')
fig.savefig(game + '/seq_tracks_1.png', dpi=300, bbox_inches='tight')
plt.show()  

# start in mid-def quarter
(fig,ax) = createZonalPitch(120,80,'yards','gray')

for i,seq in seq_summary_df.iterrows():
    if (seq['zone start'] <= 6) and (seq['zone start'] > 3):
        x1 = seq['x start']
        y1 = seq['y start']
        x2 = seq['x end']
        y2 = seq['y end']
        ax.plot([x1,x2],[y1,y2],ls='dashed',color='grey')
        ax.scatter(x1,y1,s=20,color='black')
        if (seq['ending action'] == 'shot') or (seq['ending action'] == 'free kick shot'):
            ax.scatter(x2,y2,s=20,marker='*',color=buttercup)
        else:
            ax.scatter(x2,y2,s=20,color=redish)

plt.title('Sequence Start and Ends, 2/4')
fig.savefig(game + '/seq_tracks_2.png', dpi=300, bbox_inches='tight')
plt.show()  

# start in mid-attacking quarter
(fig,ax) = createZonalPitch(120,80,'yards','gray')

for i,seq in seq_summary_df.iterrows():
    if (seq['zone start'] <= 9) and (seq['zone start'] > 6):
        x1 = seq['x start']
        y1 = seq['y start']
        x2 = seq['x end']
        y2 = seq['y end']
        ax.plot([x1,x2],[y1,y2],ls='dashed',color='grey')
        ax.scatter(x1,y1,s=20,color='black')
        if (seq['ending action'] == 'shot') or (seq['ending action'] == 'free kick shot'):
            ax.scatter(x2,y2,s=20,marker='*',color=buttercup)
        else:
            ax.scatter(x2,y2,s=20,color=redish)

plt.title('Sequence Start and Ends, 3/4')
fig.savefig(game + '/seq_tracks_3.png', dpi=300, bbox_inches='tight')
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

ax.text(6.25,1.5,str(round(half1_xG/half1shots_noblocks,2)),ha='center',va='center')
ax.text(8.75,1.5,str(round(half2_xG/half2shots_noblocks,2)),ha='center',va='center')

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

###############################################################################
######################## Sequence Summary #####################################
###############################################################################

first_quarter_df = seq_summary_df.loc[(seq_summary_df['zone start'] <= 3)]
second_quarter_df = seq_summary_df.loc[(seq_summary_df['zone start'] > 3) & 
                                       (seq_summary_df['zone start'] <= 6)]
third_quarter_df = seq_summary_df.loc[(seq_summary_df['zone start'] <= 9) & 
                                      (seq_summary_df['zone start'] > 6)]
fourth_quarter_df = seq_summary_df.loc[(seq_summary_df['zone start'] >= 10)]

(fig,ax) = plt.subplots()

ax.plot([-2,19],[6,6],color='grey')
ax.plot([-2,19],[5,5],color='grey')
ax.plot([-2,19],[3,3],color='grey')
ax.plot([-2,19],[1,1],color='grey')
ax.plot([-2,19],[-1,-1],color='grey')
ax.plot([-2,19],[-3,-3],color='grey')
ax.plot([-2,19],[-5,-5],color='grey')

ax.plot([19,19],[-5,6],color='grey')
ax.plot([16,16],[-5,6],color='grey')
ax.plot([13,13],[-5,6],color='grey')
ax.plot([10,10],[-5,6],color='grey')
ax.plot([7,7],[-5,6],color='grey')
ax.plot([-2,-2],[-5,6],color='grey')

ax.text(8.5,5.5,'1-3',ha='center',va='center')
ax.text(11.5,5.5,'4-6',ha='center',va='center')
ax.text(14.5,5.5,'7-9',ha='center',va='center')
ax.text(17.5,5.5,'10-12',ha='center',va='center')

ax.text(6.5,4,'Total sequences',ha='right',va='center')
ax.text(6.5,2,'Sequences entering\n final third',ha='right',va='center')
ax.text(6.5,0,'Sequences entering\n box',ha='right',va='center')
ax.text(6.5,-2,'Sequences resulting\n in shot',ha='right',va='center')
ax.text(6.5,-4,'Sequences resulting\n in goal',ha='right',va='center')

ax.text(8.5,4,str(len(first_quarter_df)),ha='center',va='center')
ax.text(8.5,2,str(len(first_quarter_df.loc[first_quarter_df['final third entrance'] == 'yes'])),ha='center',va='center')
ax.text(8.5,0,str(len(first_quarter_df.loc[first_quarter_df['box entrance'] == 'yes'])),ha='center',va='center')
ax.text(8.5,-2,str(len(first_quarter_df.loc[(first_quarter_df['ending action'] == 'shot') | (first_quarter_df['ending action'] == 'free kick shot')])),ha='center',va='center')
ax.text(8.5,-4,str(len(first_quarter_df.loc[first_quarter_df['shot result'] == 'goal'])),ha='center',va='center')

ax.text(11.5,4,str(len(second_quarter_df)),ha='center',va='center')
ax.text(11.5,2,str(len(second_quarter_df.loc[second_quarter_df['final third entrance'] == 'yes'])),ha='center',va='center')
ax.text(11.5,0,str(len(second_quarter_df.loc[second_quarter_df['box entrance'] == 'yes'])),ha='center',va='center')
ax.text(11.5,-2,str(len(second_quarter_df.loc[(second_quarter_df['ending action'] == 'shot') | (second_quarter_df['ending action'] == 'free kick shot')])),ha='center',va='center')
ax.text(11.5,-4,str(len(second_quarter_df.loc[second_quarter_df['shot result'] == 'goal'])),ha='center',va='center')

ax.text(14.5,4,str(len(third_quarter_df)),ha='center',va='center')
ax.text(14.5,2,str(len(third_quarter_df.loc[third_quarter_df['final third entrance'] == 'yes'])),ha='center',va='center')
ax.text(14.5,0,str(len(third_quarter_df.loc[third_quarter_df['box entrance'] == 'yes'])),ha='center',va='center')
ax.text(14.5,-2,str(len(third_quarter_df.loc[(third_quarter_df['ending action'] == 'shot') | (third_quarter_df['ending action'] == 'free kick shot')])),ha='center',va='center')
ax.text(14.5,-4,str(len(third_quarter_df.loc[third_quarter_df['shot result'] == 'goal'])),ha='center',va='center')

ax.text(17.5,4,str(len(fourth_quarter_df)),ha='center',va='center')
ax.text(17.5,2,str(len(fourth_quarter_df.loc[fourth_quarter_df['final third entrance'] == 'yes'])),ha='center',va='center')
ax.text(17.5,0,str(len(fourth_quarter_df.loc[fourth_quarter_df['box entrance'] == 'yes'])),ha='center',va='center')
ax.text(17.5,-2,str(len(fourth_quarter_df.loc[(fourth_quarter_df['ending action'] == 'shot') | (fourth_quarter_df['ending action'] == 'free kick shot')])),ha='center',va='center')
ax.text(17.5,-4,str(len(fourth_quarter_df.loc[fourth_quarter_df['shot result'] == 'goal'])),ha='center',va='center')

plt.axis('off')
ax.set_aspect('equal')
plt.title('Sequence Break Down',size=18,pad=5)
fig.savefig(game + '/seq_table.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
######################## Opportunity Sequence Summary #########################
###############################################################################

seq_summary_df['start to goal distance'] = np.sqrt((seq_summary_df['x start']-120)**2 + 
                                                   (seq_summary_df['y start']-40)**2)
seq_summary_df['end to goal distance'] = np.sqrt((seq_summary_df['x end']-120)**2 + 
                                                   (seq_summary_df['y end']-40)**2)

seq_summary_df.loc[seq_summary_df['start to goal distance']*0.5 >= seq_summary_df['end to goal distance'],'progressive'] = 'yes'
seq_summary_df.loc[(seq_summary_df['progressive'] == 'yes') & (seq_summary_df['completed passes'] <= 4),'transitional'] = 'yes'

opportunity_seq_df = seq_summary_df.loc[(seq_summary_df['box entrance'] == 'yes') | 
                                       (seq_summary_df['ending action'] == 'shot') | 
                                       (seq_summary_df['ending action'] == 'free kick shot')]

(fig,ax) = plt.subplots()

ax.plot([0,10],[0,0],color='grey')
ax.plot([0,10],[1,1],color='grey')
ax.plot([0,10],[2,2],color='grey')
ax.plot([0,10],[3,3],color='grey')
ax.plot([0,10],[4,4],color='grey')
ax.plot([0,10],[5,5],color='grey')

ax.plot([0,0],[0,5],color='grey')
ax.plot([4,4],[0,5],color='grey')
ax.plot([7,7],[0,5],color='grey')
ax.plot([10,10],[0,5],color='grey')

ax.text(3.7,3.5,'Avg number of\n successful actions',ha='right',va='center')
ax.text(3.7,2.5,'Avg number of\n complete passes',ha='right',va='center')
ax.text(3.7,1.5,'Progressive',ha='right',va='center')
ax.text(3.7,0.5,'Transitional',ha='right',va='center')

ax.text(5.5,4.5,'All Seq',ha='center',va='center')
ax.text(8.5,4.5,'Opp Seq',ha='center',va='center')

ax.text(5.5,3.5,str(round(seq_summary_df['successful actions'].mean(),2)),ha='center',va='center')
ax.text(5.5,2.5,str(round(seq_summary_df['completed passes'].mean(),2)),ha='center',va='center')
ax.text(5.5,1.5,str(len(seq_summary_df.loc[seq_summary_df['progressive'] == 'yes'])),ha='center',va='center')
ax.text(5.5,0.5,str(len(seq_summary_df.loc[seq_summary_df['transitional'] == 'yes'])),ha='center',va='center')

ax.text(8.5,3.5,str(round(opportunity_seq_df['successful actions'].mean(),2)),ha='center',va='center')
ax.text(8.5,2.5,str(round(opportunity_seq_df['completed passes'].mean(),2)),ha='center',va='center')
ax.text(8.5,1.5,str(len(opportunity_seq_df.loc[opportunity_seq_df['progressive'] == 'yes'])),ha='center',va='center')
ax.text(8.5,0.5,str(len(opportunity_seq_df.loc[opportunity_seq_df['transitional'] == 'yes'])),ha='center',va='center')

plt.axis('off')
ax.set_aspect('equal')
fig.savefig(game + '/oppsummarytable.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
########################### Heat Map ##########################################
###############################################################################

seq_df = dict()
x = dict()
y = dict()

seq_num =df['seq'].max()

for k in range(0,seq_num):
    seq_df[str(k)] = df.loc[df['seq'] == (k+1)]
    seq_df[str(k)] = seq_df[str(k)].reset_index(drop=True) 

for i in range(0,seq_num):
    x[i] = np.zeros(len(seq_df[str(i)]))
    y[i] = np.zeros(len(seq_df[str(i)]))
    for j,action in seq_df[str(i)].iterrows():
        x[i][j] = action['x1']
        y[i][j] = action['y1']

def floorgrid(x,a):
    return int(a*np.floor(x/a))
def ceilgrid(x,a):
    return int(a*np.ceil(x/a))

def f(x,y,x1,x2,y1,y2):
    return (x2-x1)*(y-y1)-(y2-y1)*(x-x1)

def linecrossings(x1,x2,y1,y2):
    xmin = np.minimum(x1,x2)
    xmax = np.maximum(x1,x2)
    ymin = np.minimum(y1,y2)
    ymax = np.maximum(y1,y2)
    # compute the bounds of the line segement 
    x_lower = floorgrid(xmin,5)
    x_upper = ceilgrid(xmax,5)
    y_lower = floorgrid(ymin,5)
    y_upper = ceilgrid(ymax,5)
    # create a matrix to store where the crossings happen
    crossing_counts = np.zeros((24,16))
    # checks each box
    for x in range(x_lower,x_upper,5):
        for y in range(y_lower,y_upper,5):
            bottom_left = f(x,y,x1,x2,y1,y2)
            bottom_right = f(x+5,y,x1,x2,y1,y2)
            top_left = f(x,y+5,x1,x2,y1,y2)
            top_right = f(x+5,y+5,x1,x2,y1,y2)
            if ((bottom_left > 0) or (bottom_right > 0) or (top_left > 0) or (top_right > 0)) and ((bottom_left < 0) or (bottom_right < 0) or (top_left < 0) or (top_right < 0)):
                crossing_counts[int(x/5),int(y/5)] += 1
    return crossing_counts

total_crossings = np.zeros((24,16))

for i in range(0,seq_num):
    if len(x[i]) > 1:
        for k in range(0,len(x[i])-1):
            total_crossings += linecrossings(x[i][k],x[i][k+1],y[i][k],y[i][k+1])

                
xbins = np.arange(0,125,5)
ybins = np.arange(0,85,5)
crossing_hist = np.transpose(total_crossings)[::-1]

(fig,ax) = createZonalPitch(120,80,'yards','black')

pos = ax.imshow((crossing_hist)**(2/3), extent=[0,120,0,80],
                cmap=plt.cm.YlOrRd,aspect='auto',interpolation='bessel')
ax.set_title('Heat Map')
plt.xlim((-1,120))
plt.ylim((-1,80))
plt.tight_layout()
plt.gca().set_aspect('equal', adjustable='box')
fig.savefig(game + '/heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

###############################################################################
############# Passes by Type Tables, Total Passes & Completion % ##############
###############################################################################

#for all types of passes, bin into zones
hist_short_passes, xedges_short, yedges_short = np.histogram2d(df_short_pass['y1'], df_short_pass['x1'], bins=(3, 4), range=[[0, 80], [0, 120]])
hist_medium_passes, xedges_medium, yedges_medium = np.histogram2d(df_medium_pass['y1'], df_medium_pass['x1'], bins=(3, 4), range=[[0, 80], [0, 120]])
hist_long_passes, xedges_long, yedges_long = np.histogram2d(df_long_pass['y1'], df_long_pass['x1'], bins=(3, 4), range=[[0, 80], [0, 120]])

#code for tables
# Table to accompany histogram.
(fig,ax) = plt.subplots()

#draws line from [x1, x2] to [y1, y2]
#draw horizontal lines
ax.plot([0, 10], [6, 6], color='grey') #top line
ax.plot([0, 10], [4.25, 4.25], color='grey') 
ax.plot([0, 10], [2.5, 2.5], color='grey')   
ax.plot([0, 10], [0.75, 0.75], color='grey')  
ax.plot([0, 10], [-1, -1], color='grey') #bottom line

#draw vertical lines
ax.plot([0, 0], [6, -1], color='grey')   
ax.plot([3.33, 3.33], [6, -1], color='grey')
ax.plot([6.66, 6.66], [6, -1], color='grey')   
ax.plot([10, 10], [6, -1], color='grey')      

#adding first row labels (total and completion)
ax.text(5.0, 5.125, 'Total Passes\n(Completed)', ha='center', va='center')
ax.text(8.33, 5.125,'Completion %',ha='center',va='center')

#populating the first column (pass type names)
ax.text(1.665, 3.375,'short passes',ha='center',va='center')
ax.text(1.665, 1.625,'medium passes',ha='center',va='center')
ax.text(1.665, -0.125,'long passes',ha='center',va='center')

short_attempt = len(df_short_pass)
medium_attempt = len(df_medium_pass)
long_attempt = len(df_long_pass)

short_complete = len(df_short_pass.loc[df_short_pass['detail'] == 'complete'])
medium_complete = len(df_medium_pass.loc[df_medium_pass['detail'] == 'complete'])
long_complete = len(df_long_pass.loc[df_long_pass['detail'] == 'complete'])

#populating the second column (total passes)
ax.text(5.0, 3.375, str(int(short_attempt)) + ' (' + str(int(short_complete)) + ')', ha='center', va='center')
ax.text(5.0, 1.625, str(int(medium_attempt)) + ' (' + str(int(medium_complete)) + ')', ha='center', va='center')
ax.text(5.0, -0.125, str(int(long_attempt)) + ' (' + str(int(long_complete)) + ')', ha='center', va='center')

#populate the third column (completion percentage). rounded to 3 decimals.
#left these as decimals but can change to percent if its easier to read
ax.text(8.33, 3.375, round(short_complete/short_attempt*100, 3), ha='center', va='center')
ax.text(8.33, 1.625, round(medium_complete/medium_attempt*100, 3), ha='center', va='center')
ax.text(8.33, -0.125, round(long_complete/long_attempt*100, 3), ha='center', va='center')

plt.axis('off')
fig.savefig(game + '/pass_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

