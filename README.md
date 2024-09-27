# Introduction

The file GameReport.py, is a file used to analyze soccer match data. Soccer match data is collected manually so it only tracks the one teams performance (not opponent performance). Please note at the present this project is not self contained. It needs a local file called FCPython.py to run (see [Field Pitch Maps](#field-pitch-maps) below). 

Please feel free to use the code presented here if you have wandered on to this page somehow. 

# Game Data File 

The code is designed to import data tables where the data is compiled using [this game tagger](https://mshea-08.github.io/). The tagger comes from [this](https://splxgoal-jkomar.pythonanywhere.com/?trk=article-ssr-frontend-pulse_little-text-block) *much nicer* game tagger developed by Chaitanya Jadhav and John Komar, where I changed the pitch image and the button options to meet the needs of this project. Big thanks to their work for making this project possible. Generally, data tables should have the following set up...

| event | detail | surface | x1 | y1 | x2 | y2 | half |
| ----- | ------ | ------- | -- | -- | -- | -- | ---- |

Where possible events, details, and surfaces can be seen in the tagger. 

# Field Pitch Maps

The pitch visuals are created using code by [FC Pythons](https://fcpython.com/). This code has been modified slightly in instances to fit the needs of this project. For example, createZonalPitch() adds zone labels to the field map based on the direction of play being from left to right. 

# File Data and Visuals 

Here I will go through the specific components of the file. 
