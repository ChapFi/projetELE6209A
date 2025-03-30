import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def parse_sensor_management(filepath):
    """
    Parses a sensor management text file with the specified structure.

    Args:
        filepath (str): The path to the sensor management text file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 3:
                    try:
                        time = float(parts[0])
                        sensor = int(parts[1])
                        index = int(parts[2])
                        data.append({'time': time, 'sensor': sensor, 'index': index})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_laser_data(filepath):
    """
    Parses a laser sensor data file with the specified structure.

    Args:
        filepath (str): The path to the laser sensor data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split(',')
                if len(parts) >= 362:  # Time + 361 laser points
                    try:
                        time_laser = float(parts[0])
                        laser_values = [float(val) for val in parts[1:]]  # Convert laser values to floats
                        data.append({'time_laser': time_laser, 'laser_values': laser_values})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line.
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_odometry_data(filepath):
    """
    Parses an odometry sensor data file with the specified structure.

    Args:
        filepath (str): The path to the odometry sensor data file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row in the file.
              Returns None if the file cannot be opened or parsed.
    """
    try:
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if not lines:
            return []  # Return empty list if file is empty

        # Skip the header line
        data = []
        for line in lines[1:]:
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split()
                if len(parts) == 3:
                    try:
                        time_vs = float(parts[0])
                        velocity = float(parts[1])
                        steering = float(parts[2])
                        data.append({'time_vs': time_vs, 'velocity': velocity, 'steering': steering})
                    except ValueError:
                        print(f"Warning: Invalid data in line: {line}")
                        # Optionally, you can choose to raise an exception or skip the line.
                else:
                    print(f"Warning: Incorrect number of columns in line: {line}")
                    #Optionally, you can choose to raise an exception or skip the line.

        return data

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

L = 2.83  # Wheelbase
H = 0.76
b = 0.5
nbTrees = 0

def find_circle_center_least_squares(points):
    """
    Calculates the center of a circle using least squares fitting.

    Args:
        points: A list of tuples, each representing a point (x, y).

    Returns:
        A tuple (h, k) representing the center of the circle.
    """
    n = len(points)
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    A = np.column_stack((-2 * x, -2 * y, np.ones(n)))
    b = -(x**2 + y**2)

    h, k, _ = np.linalg.lstsq(A, b, rcond=None)[0]

    return (h, k)

def g(mesureT, estimatet1,dt, N):
    vt = mesureT[0,0]
    wt = mesureT[1,0]
    Dmu = np.array([[vt*(-np.sin(estimatet1[3])+np.sin(estimatet1[3]+wt*dt))/wt],
                    [vt*(np.cos(estimatet1[3])-np.cos(estimatet1[3]+wt*dt))/wt],
                    [wt*dt]])
    Fx = np.zeros([3, 2*N+3])
    Fx[0:3,0:3] = np.eye(3)
    newmu = np.add(estimatet1, np.matmul(np.transpose(Fx), Dmu))
    return newmu

def covMatrix(vt, wt, theta, dt,N, sigma):
    Gx = np.array([[0, 0, vt*(-np.cos(theta)+np.cos(theta[3]+wt*dt))/wt],
                   [0, 0, vt*(-np.sin(theta)+np.sin(theta+wt*dt))/wt],
                   [0, 0, 0]])
    Fx = np.zeros((3, 2*N+3))
    Fx[0:3,0:3] = np.eye(3)

    Rx = np.zeros((3,3))
    Rx[0,0] = 0.5*0.5
    Rx[1,1] = 0.5*0.5
    Rx[2,2] = 0.5*np.pi/180*0.5*np.pi/180

    G = np.add(np.eye(3+2*N), np.matmul(np.matmul(np.transpose(Fx), Gx), Fx))
    C1 = np.matmul(np.matmul(G, sigma), np.transpose(G))
    C2 = np.matmul(np.matmul(np.transpose(Fx), Rx), Fx)
    covM = np.add(C1, C2)
    return covM

def compute_data_association(z):
    cost_matrix = np.zeros((len(Trees), len(z)))

    # Calculate the cost matrix (e.g., Euclidean distance)
    for i, existing in enumerate(Trees):
        for j, new in enumerate(z):
            cost = np.sqrt((existing[0] - new[0])**2 + (existing[1] - new[1])**2 + (existing[2] - new[2])**2)
            cost_matrix[i, j] = cost

    # Apply the Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create the associations
    associations = list(zip(row_ind, col_ind))

    for 

def extendedKalman(state, u, sigma, measure, dt):
    N = (len(state)-3)/3
    currenTime = currenTime + dt
    estState = g(u, state, dt)
    estSigma = covMatrix( u['velocity'], u['steering'], state[3,0],dt, N, sigma)
    
    #Correction
    sr2 = 1 #To DO trouver les valeurs
    sp2 = 1
    Qt = np.array([[sr2,0],[sp2,0]])
    for z in measure:
        
        j = 3#TO DO

        if np.norm(estState[j:j+2]) == 0:
            estState[2*j:2*j+2]  = np.add(estState[0:2], np.array([[z[0,0]*np.cos(z[1,0] + estState[2,0])]
                                                               ,[z[0,0]*np.sin(z[1,0] + estState[2,0])]]))
        delta = np.subtract(estState[0:2], estState[j:j+2])
        q = np.mamtmul(np.transpose(delta), delta)
        zest = np.array([[np.sqrt(q)],
                         [np.arctan2(delta[1,0], delta[0,0])-estState[2,0]] ])
        err = np.subtract(z, zest)
        Fxj = np.zeros((5, 3+2*N))
        Fxj[0:3,0:3] = np.eye(3)
        Fxj[3:5, 2*j:2*j+2] = np.eye(2)

        H = np.array([[-np.sqrt(q)*delta[0,0], -np.sqrt(q)*delta[1,0], 0, np.sqrt(q)*delta[0,0], np.sqrt(q)*delta[1,0]],
                      [delta[1,0], -delta[0,0], -q, -delta[1,0], delta[0,0]]])
        Hi = np.matmul(H, Fxj)
        A = np.add(np.matmul(np.matmul(Hi, estSigma), np.transpose(Hi)), Qt)
        b = np.matmul(estSigma, np.transpose(Hi))
        Ki = np.linalg.solve(np.transpose(A), np.transpose(b))
        Ki = np.transpose(Ki)
        estState = np.add(estState, np.matmul(Ki, ))
        estSigma = np.matmul(np.subtract(np.eye(3+2*N), np.matmul(Ki, Hi)), estSigma)

    newState = estState
    newSigma = estSigma
    return newState, newSigma

def EKFSlam(rowOdom, rowLaser, sensorManager):

    #Initiale state
    nbLandmark = 500
    X = np.zeros([3+2*nbLandmark,1])
    sigma = np.zeros(3+2*nbLandmark)
    cTime = 0

    i = 0
    for sensor in sensorManager:
        if sensor['sensor'] == 2 and sensorManager[i+1] == 3:
            # Odom mesurement
            idu = int(sensor['index'])-3
            u = rowOdom[idu]

            #Time
            dt = u['time_vs'] - cTime
            cTime = dt + cTime

            #Laser mesurement
            idlaser = int(sensor['index'])-2
            laserMesures = rowLaser[idlaser]
            laser = laserMesures['laser_values']
            angles = np.linspace(-np.pi/2, np.pi/2, 361)

            #get the cluster point
            ranges =[]
            current_range =[]
            for angle, range in zip(angles, laser):
                if 3 <= range < 40:
                    current_range.append((angle,range))
                else:
                    if current_range:  # If there are measurements in the current range
                        ranges.append(current_range)
                        current_range = []  # Reset for the next range

            if current_range: #capture the last range if it doesn't end with the out of range value
                ranges.append(current_range)

            z = []
            for dataRange in ranges:
                if len(dataRange) >= 8:
                    points = [(X[0,0]+pair[1]*np.cos(pair[0]+X[2,0]), X[1,0]+pair[1]*np.sin(pair[0]+X[2,0])) for pair in dataRange]
                    (h,k) = find_circle_center_least_squares(points) #obtain the center of the cluster
                    deltaB = dataRange[len(dataRange)-1][0]-dataRange[0][0]
                    diam = deltaB*sum([pair[1] for pair in dataRange])/len(dataRange) #obtain the diameter
                    z.append((h,k, diam))
            
            #Use the EKF for estmation
            state, sigma = extendedKalman(state, u, sigma, z, dt)
        if i == len(sensorManager-1):
            break
        i +=1
    return state
            
def displayRowData(rowOdom, rowLaser, sensorManager):
    X = np.array([[0],[0], [0]])
    Odom = np.zeros((3, len(rowOdom)+1))
    landMark = np.zeros((2,17233))
    landtree = np.zeros((2,17233))
    i = 0
    j = 0
    time =0
    for sensor in sensorManager:
        if sensor['sensor'] == 2:
            id = int(sensor['index'])-3
            mesure = rowOdom[id]
            dt = mesure['time_vs']-time
            time = dt + time
            vt = mesure['velocity']
            wt = mesure['steering']
            theta = X[2,0]
            Dmu = np.array([[vt*(-np.sin(theta)+np.sin(theta+wt*dt))/wt],
                        [vt*(np.cos(theta)-np.cos(theta+wt*dt))/wt],[wt*dt]])
            X = np.add(X, Dmu)
            Odom[:,i] = X[:,0]
            i += 1
        elif sensor['sensor'] == 3:
            id = int(sensor['index'])-3
            mesures = rowLaser[id]
            mesures = mesures['laser_values']
            angles = np.linspace(-np.pi/2, np.pi/2, 361)
            ranges =[]
            current_range =[]
            for angle, range in zip(angles, mesures):
                if range < 81:
                    current_range.append((angle,range))
                else:
                    if current_range:  # If there are measurements in the current range
                        ranges.append(current_range)
                        current_range = []  # Reset for the next range

            if current_range: #capture the last range if it doesn't end with the out of range value
                ranges.append(current_range)

            for dataRange in ranges:
                if len(dataRange) >= 8:

                    points = [(X[0,0]+pair[1]*np.cos(pair[0]+X[2,0]), X[1,0]+pair[1]*np.sin(pair[0]+X[2,0])) for pair in dataRange]
                    deltaB = dataRange[len(dataRange)-1][0]-dataRange[0][0]
                    (h,k) = find_circle_center_least_squares(points)
                    landtree[0,j] = h
                    landtree[1,j] = k

                    nbRange = len(dataRange)
                    angle = sum([pair[0] for pair in dataRange])/nbRange
                    range = sum([pair[1] for pair in dataRange])/nbRange
                    lm = np.array([[X[0,0]+range*np.cos(angle+X[2,0])],
                                            [X[1,0]+range*np.sin(angle+X[2,0])]])
                    landMark[:,j] = lm[:,0]
                    j += 1
    print(j)

    # Plot trajectory and landmark
    plt.figure()
    plt.scatter(landMark[0, :], landMark[1, :], marker='+', label='landmarks')
    plt.scatter(landtree[0, :], landtree[1, :], marker='.', label='landmarks')
    plt.plot(Odom[0, :], Odom[1,:], label='row trajectory', linestyle='--', color='r')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.title('EKF SLAM')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    dataManagement = parse_sensor_management("dataset/Sensors_manager.txt")
    laserData = parse_laser_data('dataset/LASER.txt')
    drsData = parse_odometry_data('dataset/DRS.txt')    
    xf = displayRowData(drsData, laserData, dataManagement)
    print(xf)

