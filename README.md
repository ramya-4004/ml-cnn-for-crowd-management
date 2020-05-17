# ml-cnn-for-crowd-management

Project For DevJam
Team - Infiltrators

## Table Of Contents
* [Info](#Info)
* [Requirements](#Requirements)
* [Overview of code](#Overview)
* [How To Run](#How-to-run)
* [Timeline](#Timeline)

## Info
To ensure Social Distancing is being followed , We have two approaches:-

1. Queue Management in - front of retail shops.
2.Crowd counting and liminting the no. of people at a time the stores.

If either of the two is being unfollowed , we plan to send a message to the shop owner / manager.

This is the whole idea around which around project revovles.

## Requirements
This project is being developed using
* Python 3.6
* OpenCV
* imutils
* Numpy

## Overview

### 1. Queue Management

We implemented a tracking + detection system for controlling the Queue and Crowd outside the store.

For detection we are using the pre-built model present in OpenCV:
`hog = cv2.HOGDescriptor()`
`hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())`

The tracking algorithm consists of Centroid POint tracking And ID Managements.
The centroidtracker.py file in OUTSIDE_STORE has all the further details.

There is a video also in OUTSIDE_STORE which is used as a demo for the basic working of the aforementioned scripts.
The full functionality of this is still a Work In Progress.

![Snapshot](Demo.png)

### 2. Crowd Counting


## How-to-run

```
Directory_where_the_project_is>cd OUTSIDE_STORE
>python queue.py

```

## Timeline

17 - 05 - 2020
	
	Crowd checking is being done thru the files in the OUTSIDE_STORE.
	For testing we have used the video also included in the folder.
	Position Markers - Work In Progress(Not Reacting to the number of people in frame)
	So on temporary basis , have included some custom markers.
	END.