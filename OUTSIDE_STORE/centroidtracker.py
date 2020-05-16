# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
	def __init__(self, maxDisappear=10):
		# available ID
		self.nextOID = 0
		# list of objects
		# key - OID , value - centroid coordinates
		self.objects = OrderedDict()
		# how many times each object has disappeared/lost in frames
		# key - OID , value - lost count
		self.disappear = OrderedDict()

		# we need to keep track how many times the object has disappeared from frames consecutively.
		# if above a limit we can say that we no longer need to keep track of that object.
		self.maxDisappear = maxDisappear

	def insert(self, centroid):
		# keeping record of the objects we need to detect/track.
		self.objects[self.nextOID] = centroid
		self.disappear[self.nextOID] = 0
		self.nextOID += 1

	def delete(self, OID):
		# deleting the object from our records of tracking
		del self.objects[OID]
		del self.disappear[OID]

	def update(self, rects):
		# rects - boxes coordinates returned by our detector
		if len(rects) == 0:
			# means that the screen is empty
			# so for existing objects that are being tracked have been lost/disappeared.
			for OID in list(self.disappear.keys()):
				self.disappear[OID] += 1
				# checking the limit of disappearance
				if self.disappeared[OID] > self.maxDisappeared:
					self.delete(OID)

			# there is no tracking info to update
			return self.objects

		# ELSE
		# input centroids for the current frame
		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (sX, sY, eX, eY)) in enumerate(rects):
			cX = int((sX + eX) / 2.0)
			cY = int((sY + eY) / 2.0)
			inputCentroids[i] = (cX, cY)

		# if no objects are being tracked
		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.insert(inputCentroids[i])

		# otherwise, are are currently tracking objects.
		# here our logic kicks in
		# we calculate the pair wise distance between the existing points being tracked and
		# the new points we have taken in the frame.
		# then we choose the smallest distance w.r.t the existing points.
		# the inputted points corresponding to that are actually on the same object as that of existing points.
		# so they do not need a new id.
		# for the remaining ones (if any) we assign them new ids and start to track them also.

		else:
			# the data for the already being tracked objects' points.
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			# in order to perform this matching we must (1) find the
			# smallest value in each row and then (2) sort the row
			# indexes based on their minimum values so that the row
			# with the smallest value as at the *front* of the index
			# list
			rows = D.min(axis=1).argsort()

			# next, we perform a similar process on the columns by
			# finding the smallest value in each column and then
			# sorting using the previously computed row index list
			cols = D.argmin(axis=1)[rows]

			# this keeps track of the pairs of (row , col) already being used.
			usedRows = set()
			usedCols = set()

			# this is looping over for the existing tracking points
			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue
				# ELSE
				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappear[objectID] = 0

				usedRows.add(row)
				usedCols.add(col)

			# both the row and column index we have NOT yet used.
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# if the number of already tracked points is equal or greater than the number of inputted points
			# there is a chance that objects have disappeared.
			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappear[objectID] += 1
					# again checking for total loss according to the limit.
					if self.disappear[objectID] > self.maxDisappear:
						self.delete(objectID)

			# this means we have new objects to be inserted and tracked.
			else:
				for col in unusedCols:
					self.insert(inputCentroids[col])

		# return the set of trackable objects
		return self.objects