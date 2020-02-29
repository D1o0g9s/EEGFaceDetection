###############
### Imports ###
###############

# Allow Python 2 to run this code.
from __future__ import absolute_import, division

# psychopy imports
from psychopy.hardware import keyboard
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import random as rd # shuffle etc

# import numpy as np # numpy imports for eeg
import os  # handy system and path functions
import sys  # to get file system encoding
import csv # To save experiment info into csv

# String constants for markers
from PsychoPyConstants import *

# For LSL marker stream
from pylsl import StreamInfo, StreamOutlet

## These will be migrated to the PsychoPy file
# import threading # possibly won't need this. 
# from pyOpenBCI import OpenBCICyton

sys.path.append('../')

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

#############################
### Initialize variables ###
#############################

NUM_IMAGES_TO_SHOW = 10 # Total number of images to show
NUM_SECONDS_TO_SHOW_IMAGE = 2 # seconds

FACE = 0
LANDSCAPE = 1
# Get all the files in the folders
images_faces_path = "./images/faces"
all_faces_filenames = os.listdir(images_faces_path)
faces_filenames = [os.path.join(images_faces_path, all_faces_filenames[i]) for i in range(len(all_faces_filenames))]
rd.shuffle(faces_filenames)
NUM_FACE_IMAGES = len(faces_filenames)


images_landscape_path = "./images/landscape"
all_landscape_filenames = os.listdir(images_landscape_path)
landscape_filenames = [os.path.join(images_landscape_path, all_landscape_filenames[i]) for i in range(len(all_landscape_filenames))]
rd.shuffle(landscape_filenames)
NUM_LANDSCAPE_IMAGES = len(landscape_filenames)

# Instructions string
instructions_text = "Calibration\nYou will be shown a series of images. \nPlease keep as still as possible to reduce noise in the data. \n\nReady?"


class DiscriminationExperiment: 
    
    def __init__(self): 

        # Elements
        self.__win = None
        self.__routineTimer = None
        self.__kb = None
        self.__image_stim = None
        self.__image_filename = None

        self.__marker_outlet = None

        self.__current_face = 0
        self.__current_landscape = 0
        self.__current_type = 0 # 0 means face, 1 means landscape

        self.__endExpNow = False

    def __getNextImage(self) : 
        # Gets the next 
        if self.__current_type == FACE: 
            self.__current_type = LANDSCAPE
            next_image_idx = self.__current_face
            self.__current_face = (1 + self.__current_face) % NUM_FACE_IMAGES
            next_image_filename = faces_filenames[next_image_idx]
        else : 
            self.__current_type = FACE
            next_image_idx = self.__current_landscape
            self.__current_landscape = (1 + self.__current_landscape) % NUM_LANDSCAPE_IMAGES
            next_image_filename = landscape_filenames[next_image_idx]
        next_image = self.__getImageStim(next_image_filename)
        return next_image


    def __getTextStim(self, text, location=(0,0), height=0.05): 
        return visual.TextStim(win=self.__win, name='textStim',
            text=text,
            font='Arial',
            pos=location, height=height, wrapWidth=None, ori=0,
            color='white', colorSpace='rgb', opacity=1,
            languageStyle='LTR',
            depth=0.0)
        

    def __getImageStim(self, filename): 
        self.__image_stim = visual.ImageStim(
            win=self.__win, name='image',
            image=filename, mask=None,
            ori=0, units='norm', pos=(0, 0), size=(1, 1.5),
            color=[1,1,1], colorSpace='rgb', opacity=1,
            flipHoriz=False, flipVert=False,
            texRes=128, interpolate=True, depth=0.0)
        self.__image_filename = filename
        return self.__image_stim


    def __startRoutine(self, components) :
        for thisComponent in components:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

    def __setDrawOn(self, components) :
        for stim in components :
            if stim.status == NOT_STARTED:
                stim.status = STARTED
                stim.setAutoDraw(True)
    
    def __endRoutine(self, components) :
        for thisComponent in components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
                thisComponent.status = NOT_STARTED 

    
    def __getDatafilenameAndSetupWindow(self): 
        #################
        ### Start Box ###
        #################
        psychopyVersion = '3.0.5'
        expName = 'Discrimination Experiment'  # from the Builder filename that created this script
        expInfo = {'participant': '', 'session': '001'}
        dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
        if dlg.OK == False:
            core.quit()  # user pressed cancel
        expInfo['date'] = data.getDateStr()  # add a simple timestamp
        expInfo['expName'] = expName
        expInfo['psychopyVersion'] = psychopyVersion
        expInfo['numImages'] = NUM_IMAGES_TO_SHOW
        expInfo['numSecondsBetweenImages'] = NUM_SECONDS_TO_SHOW_IMAGE

        # Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
        filename = _thisDir + os.sep + u'data/%s_%s_%s_%s' % (expInfo['participant'], expInfo['session'], expName, expInfo['date'])
        
        # Save the experiment meta data to a csv file
        with open(filename+'.csv', 'w') as f:  # Just use 'w' mode in 3.x
            w = csv.writer(f)
            w.writerow(expInfo.keys())
            w.writerow(expInfo.values())

        # An ExperimentHandler isn't essential but helps with data saving
        thisExp = data.ExperimentHandler(name=expName, version='',
            extraInfo=expInfo, runtimeInfo=None,
            savePickle=True, saveWideText=True,
            dataFileName=filename)

        ####################
        ### Window Setup ###
        ####################
        self.__win = visual.Window(
            size=(1430, 870), fullscr=False, screen=0,
            allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            blendMode='avg', useFBO=True,
            units='height')
        # store frame rate of monitor if we can measure it
        expInfo['frameRate'] = self.__win.getActualFrameRate()
        if expInfo['frameRate'] != None:
            frameDur = 1.0 / round(expInfo['frameRate'])
        else:
            frameDur = 1.0 / 60.0  # could not measure, so guess
        
        return filename, thisExp, expInfo
    
    def __createMarkerStream(self) : 
        info = StreamInfo(name='PsychoPy Markers', 
            type='Markers', channel_count=1, nominal_srate=0, 
            channel_format='string', source_id='psychopy_thread')
        outlet = StreamOutlet(info)
        return outlet


    def __showTextWithSpaceExit(self, text, location=(0, 0), add_instr=True, height=0.05): 
        
        stim = self.__getTextStim(text + ("\n\n>> Press Space to advance." if add_instr else ""), location=location, height=height)
        components = [stim]
        self.__startRoutine(components)

        continueRoutine = True
        while continueRoutine:

            # Draw components! 
            self.__setDrawOn(components)

            # Check if space pressed
            if 'space' in self.__kb.getKeys(['space'], waitRelease=True): 
                continueRoutine = False

            # Check for ESC quit 
            if self.__endExpNow or 'escape' in self.__kb.getKeys(['escape'], waitRelease=True):
                core.quit()
                sys.exit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                self.__win.flip()

        self.__endRoutine(components)


    def __showTimedText(self, text, seconds): 
        
        stim = self.__getTextStim(text)
        components = [stim]
        self.__startRoutine(components)

        # Initalize timer
        self.__routineTimer.reset()
        self.__routineTimer.add(seconds)

        continueRoutine = True
        while continueRoutine and self.__routineTimer.getTime() > 0:

            self.__setDrawOn(components)

            # Check for ESC quit
            if self.__endExpNow or 'escape' in self.__kb.getKeys(['escape'], waitRelease=True):
                # self.__board.stop_stream()
                core.quit()
                sys.exit()


            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                self.__win.flip()

        self.__endRoutine(components)


    def runPsychopy(self):
        # make the marker stream. Must do before the window setup to be able to start Lab Recorder
        self.__marker_outlet = self.__createMarkerStream()
        
        # Get experiement details and filename
        filename, thisExp, expInfo = self.__getDatafilenameAndSetupWindow()

        # save a log file for detail verbose info
        # logFile = logging.LogFile(filename+'.log', level=logging.EXP)
        logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

        ### ESC flag ###
        self.__endExpNow = False  # flag for 'escape' or other condition => quit the exp

        # Create some handy timers
        globalClock = core.Clock()  # to track the time since experiment started
        self.__routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine
        self.__kb = keyboard.Keyboard()


        # Flag the start of the Psychopy experiment
        self.__marker_outlet.push_sample([RECORDING_START_MARKER])
        self.__showTextWithSpaceExit(instructions_text)
        self.__marker_outlet.push_sample([CALIBRATION_START_MARKER])

        
        for i in range(NUM_IMAGES_TO_SHOW) : 

            # Reset the timers
            self.__routineTimer.reset()
            self.__kb.clock.reset() 
            time_shown = 0

            self.__marker_outlet.push_sample([NEW_IMAGE_START_MARKER])
            self.__marker_outlet.push_sample([FACE_IMAGE_MARKER] if self.__current_type == FACE else [LANDSCAPE_IMAGE_MARKER])

            self.__getNextImage()
            self.__startRoutine([self.__image_stim])
            self.__marker_outlet.push_sample([self.__image_filename])
            print(self.__image_filename)

            self.__setDrawOn([self.__image_stim])
            self.__showTimedText("", NUM_SECONDS_TO_SHOW_IMAGE)
            self.__endRoutine([self.__image_stim])

            self.__marker_outlet.push_sample([NEW_IMAGE_END_MARKER])


        # Flag the end of the Psychopy experiment
        self.__marker_outlet.push_sample([CALIBRATION_END_MARKER])
        self.__marker_outlet.push_sample([RECORDING_END_MARKER])

        logging.flush()
        # make sure everything is closed down
        thisExp.abort()  # or data files will save again on exit
        self.__win.close()

    
myExperiment = DiscriminationExperiment() 
myExperiment.runPsychopy()
