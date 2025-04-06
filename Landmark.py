class Landmark:
    def __init__(self, diameter, center, r=0, theta=0):
        self.diameter = diameter
        self.centerx, self.centery = center
        self.r = r
        self.theta = theta

    def __eq__(self, other):
        if isinstance(other, Landmark):
            return abs(self.diameter - other.diameter) < 1e-2 and abs(self.center - other.center) < 1e-2
        return False

    def update(self, r, theta):
        self.r = r
        self.theta = theta

    def getInfo(self):
        return self.diameter, (self.centerx, self.centery)

    def getDistance(self):
        return self.r, self.theta



class Landmarks:
    def __init__(self):
        self.landmarks: [Landmark] = []

    def add(self, landmark):
        if landmark not in self.landmarks:
            self.landmarks.append(landmark)
            index = len(self.landmarks)-1
            return index, landmark
        else:
            index = self.landmarks.index(landmark)
            self.landmarks[index] = landmark
            return index, landmark

    def __getitem__(self, item):
        return item, self.landmarks[item]
